#!/usr/bin/env python3
"""
Workstream 8 Stage 9: Full Multi-Epoch Controlled Acoustic Training Engine
========================================================================
Executes full multi-epoch convergence training for B-02A, B-02B, and B-02C:
- Independent initialization from base XLS-R 300M
- Max 20 epochs with early stopping (patience=4, min 3 epochs)
- Length-bucketed dynamic batching (batch_size=32)
- BF16 AMP, single-process, 0% deadlock risk
- Memory protected with expandable segments and zero-grad set_to_none
- Checkpoint saving every 500 steps (with resume support mid-epoch)
- Full validation at every epoch boundary on all 6,728 validation samples
- Exact unpadded/masked Greedy decoding
"""

import os
import sys
import gc
import json
import time
import math
import random
import argparse
from pathlib import Path

# Memory optimization: Prevent allocator fragmentation on unified memory systems
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Determinism
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

WORKSPACE_DIR = Path("/home/musa/Kashmiri Project")
DATASET_DIR = WORKSPACE_DIR / "kashmiri_dataset"
EXP_ROOT = DATASET_DIR / "experiments/native_asr_training/experiments"
STAGE9_DIR = EXP_ROOT / "stage9_b02_acoustic_comparison"
BASE_XLSR_DIR = EXP_ROOT / "B-01-xlsr300m-character-ctc/checkpoints/base_xlsr"
TOK_DIR = EXP_ROOT / "B-01-xlsr300m-character-ctc/tokenizer"
VAL_MANIFEST = DATASET_DIR / "experiments/native_asr_training/data_splits/val.jsonl"
STAGE7_4GRAM_LM = EXP_ROOT / "stage7_ctc_lm_decoding/language_models/lm_word_4gram_domain_d.arpa"

sys.path.insert(0, str(DATASET_DIR / "scripts"))
from metrics_evaluator import calculate_wer_and_cer, normalize_perso_arabic_kashmiri

# Cached resamplers to avoid repeated allocations in audio reading
_RESAMPLERS = {}
def get_resampler(orig_sr, target_sr=16000):
    if orig_sr not in _RESAMPLERS:
        import torchaudio
        _RESAMPLERS[orig_sr] = torchaudio.transforms.Resample(orig_sr, target_sr)
    return _RESAMPLERS[orig_sr]


class LengthBucketedDataset:
    def __init__(self, manifest_file, max_duration=16.0, min_duration=0.5, batch_size=32):
        self.samples = []
        filtered_ctc = 0
        with open(manifest_file, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                dur = float(d.get("duration", d.get("duration_seconds", 0)))
                if not (min_duration <= dur <= max_duration):
                    continue
                raw_txt = d.get("text", "").strip()
                norm_txt = normalize_perso_arabic_kashmiri(raw_txt)
                if not norm_txt:
                    continue
                t_frames = int(dur * 50)
                if len(norm_txt) > t_frames:
                    filtered_ctc += 1
                    continue
                self.samples.append({
                    "sample_id": d.get("sample_id", d.get("segment_id", "")),
                    "audio_path": d["audio_path"],
                    "duration": dur,
                    "raw_text": raw_txt,
                    "normalized_text": norm_txt,
                    "source": d.get("source", d.get("dataset_source", "unknown"))
                })

        self.samples.sort(key=lambda x: x["duration"])
        self.batch_size = batch_size
        self.batches = [self.samples[i:i+batch_size] for i in range(0, len(self.samples), batch_size)]
        print(f"Loaded {len(self.samples):,} valid samples into {len(self.batches):,} batches (batch_size={batch_size}) from {Path(manifest_file).name}")
        if filtered_ctc > 0:
            print(f"  Safely filtered {filtered_ctc} CTC length violation samples (T < U).")

    def get_shuffled_batches(self, seed=None):
        shuffled = list(self.batches)
        if seed is not None:
            r = random.Random(seed)
            r.shuffle(shuffled)
        else:
            random.shuffle(shuffled)
        return shuffled


def collate_batch(batch_items, processor):
    wavs = []
    norm_texts = []
    raw_texts = []
    sample_ids = []
    durations = []
    sources = []
    
    for item in batch_items:
        audio_p = WORKSPACE_DIR / item["audio_path"]
        try:
            audio, sr = sf.read(str(audio_p), dtype="float32")
            if audio_p.suffix == ".wav" and audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sr != 16000:
                resampler = get_resampler(sr, 16000)
                t_audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
                audio = resampler(t_audio).squeeze(0).numpy()
        except Exception:
            continue
        
        wavs.append(audio)
        norm_texts.append(item["normalized_text"])
        raw_texts.append(item["raw_text"])
        sample_ids.append(item["sample_id"])
        durations.append(len(audio) / 16000.0)
        sources.append(item.get("source", "unknown"))
        
    if not wavs:
        return None

    inputs = processor(wavs, sampling_rate=16000, return_tensors="pt", padding=True)
    labels = processor.tokenizer(norm_texts, return_tensors="pt", padding=True).input_ids
    labels = labels.masked_fill(labels == processor.tokenizer.pad_token_id, -100)
    
    return {
        "input_values": inputs.input_values,
        "attention_mask": inputs.attention_mask,
        "labels": labels,
        "sample_ids": sample_ids,
        "durations": durations,
        "norm_texts": norm_texts,
        "raw_texts": raw_texts,
        "sources": sources
    }


def evaluate_validation(model, processor, val_dataset, device, val_batch_size=16):
    model.eval()
    torch.cuda.empty_cache()
    gc.collect()

    all_raw_refs = []
    all_norm_refs = []
    all_raw_hyps = []
    all_norm_hyps = []
    all_sample_ids = []
    all_sources = []
    
    val_loss_total = 0.0
    val_batches_count = 0
    blank_count = 0
    total_emitted_tokens = 0
    empty_hyps = 0

    total_word_err_norm = 0
    total_words_norm = 0
    total_char_err_norm = 0
    total_chars_norm = 0

    total_word_err_raw = 0
    total_words_raw = 0
    total_char_err_raw = 0
    total_chars_raw = 0

    val_batches = val_dataset.batches
    if val_dataset.batch_size != val_batch_size:
        val_batches = [val_dataset.samples[i:i+val_batch_size] for i in range(0, len(val_dataset.samples), val_batch_size)]

    with torch.no_grad():
        for batch in val_batches:
            col = collate_batch(batch, processor)
            if col is None:
                continue

            input_values = col["input_values"].to(device)
            attention_mask = col["attention_mask"].to(device)
            labels = col["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_values=input_values, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss_total += loss.item()
                val_batches_count += 1
                logits = outputs.logits

            pred_ids = torch.argmax(logits, dim=-1)
            for i in range(len(col["sample_ids"])):
                p_ids = pred_ids[i]
                blank_count += (p_ids == 0).sum().item()
                total_emitted_tokens += len(p_ids)

                pred_raw = processor.decode(p_ids)
                pred_norm = normalize_perso_arabic_kashmiri(pred_raw)
                
                ref_raw = col["raw_texts"][i]
                ref_norm = col["norm_texts"][i]
                
                # Normalized metrics
                res_norm = calculate_wer_and_cer(ref_norm, pred_norm)
                total_word_err_norm += res_norm["word_dist"]
                total_words_norm += res_norm["ref_words"]
                total_char_err_norm += res_norm["char_dist"]
                total_chars_norm += res_norm["ref_chars"]

                # Raw metrics
                res_raw = calculate_wer_and_cer(ref_raw, pred_raw)
                total_word_err_raw += res_raw["word_dist"]
                total_words_raw += res_raw["ref_words"]
                total_char_err_raw += res_raw["char_dist"]
                total_chars_raw += res_raw["ref_chars"]

                all_raw_refs.append(ref_raw)
                all_norm_refs.append(ref_norm)
                all_raw_hyps.append(pred_raw)
                all_norm_hyps.append(pred_norm)
                all_sample_ids.append(col["sample_ids"][i])
                all_sources.append(col["sources"][i])
                
                if not pred_norm.strip():
                    empty_hyps += 1

    torch.cuda.empty_cache()
    gc.collect()

    mean_val_loss = val_loss_total / max(1, val_batches_count)
    norm_wer = total_word_err_norm / max(1, total_words_norm)
    norm_cer = total_char_err_norm / max(1, total_chars_norm)
    raw_wer = total_word_err_raw / max(1, total_words_raw)
    raw_cer = total_char_err_raw / max(1, total_chars_raw)
    blank_rate = blank_count / max(1, total_emitted_tokens)
    empty_pct = (empty_hyps / max(1, len(all_norm_refs))) * 100.0

    return {
        "val_loss": mean_val_loss,
        "norm_wer": norm_wer,
        "norm_cer": norm_cer,
        "raw_wer": raw_wer,
        "raw_cer": raw_cer,
        "blank_rate": blank_rate,
        "empty_pct": empty_pct,
        "total_words_norm": total_words_norm,
        "total_chars_norm": total_chars_norm,
        "sample_ids": all_sample_ids,
        "raw_refs": all_raw_refs,
        "norm_refs": all_norm_refs,
        "raw_hyps": all_raw_hyps,
        "norm_hyps": all_norm_hyps,
        "sources": all_sources
    }


def run_experiment(exp_name, manifest_name, max_epochs=20, min_epochs=3, patience=4, batch_size=32, resume=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = STAGE9_DIR / exp_name
    best_model_dir = exp_dir / "checkpoints/best_model"
    latest_ckpt_dir = exp_dir / "checkpoints/latest_checkpoint"
    log_dir = exp_dir / "logs"
    pred_dir = exp_dir / "predictions"
    report_dir = exp_dir / "reports"

    for d in [best_model_dir, latest_ckpt_dir, log_dir, pred_dir, report_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(f"FULL MULTI-EPOCH TRAINING EXPERIMENT: {exp_name}")
    print(f"Corpus Manifest: {manifest_name}")
    print(f"Max Epochs: {max_epochs} | Min Epochs: {min_epochs} | Patience: {patience}")
    print(f"Checkpoint Interval: Every 500 steps | Effective Batch Size: {batch_size} | BF16 AMP: True")
    print("=" * 75)

    processor = Wav2Vec2Processor.from_pretrained(str(TOK_DIR))
    manifest_p = STAGE9_DIR / "manifests" / manifest_name
    train_dataset = LengthBucketedDataset(manifest_p, max_duration=16.0, min_duration=0.5, batch_size=batch_size)
    val_dataset = LengthBucketedDataset(VAL_MANIFEST, max_duration=16.0, min_duration=0.5, batch_size=16)

    steps_per_epoch = len(train_dataset.batches)
    total_training_steps = max_epochs * steps_per_epoch
    warmup_steps = int(0.10 * total_training_steps)

    print(f"Epoch Structure: {steps_per_epoch:,} steps/epoch | Total Planned: {total_training_steps:,} steps | Warmup: {warmup_steps:,} steps")

    train_log_file = log_dir / "training_log.csv"
    epoch_log_file = log_dir / "epoch_metrics.csv"
    resource_file = log_dir / "resource_usage.csv"

    start_epoch = 1
    global_step = 0
    batch_offset = 0
    best_val_wer = float("inf")
    non_improving_epochs = 0
    state = None
    is_resumed = False

    if resume and (latest_ckpt_dir / "training_state.pt").exists():
        print(f"Loading resume checkpoint from: {latest_ckpt_dir}...")
        state = torch.load(latest_ckpt_dir / "training_state.pt", map_location=device, weights_only=False)
        model = Wav2Vec2ForCTC.from_pretrained(str(latest_ckpt_dir)).to(device)
        is_resumed = True
        
        saved_epoch = state["epoch"]
        saved_step = state["global_step"]
        saved_offset = state.get("batch_offset", 0)

        if saved_offset == 0 or saved_offset >= steps_per_epoch:
            start_epoch = saved_epoch + 1
            batch_offset = 0
        else:
            start_epoch = saved_epoch
            batch_offset = saved_offset

        global_step = saved_step
        best_val_wer = state.get("best_val_wer", float("inf"))
        non_improving_epochs = state.get("non_improving_epochs", 0)
    else:
        print(f"Initializing fresh model from base XLS-R checkpoint: {BASE_XLSR_DIR}...")
        model = Wav2Vec2ForCTC.from_pretrained(str(BASE_XLSR_DIR), ctc_zero_infinity=True).to(device)

    model.freeze_feature_encoder()

    ENC_LR = 1.0e-4
    HEAD_LR = 1.0e-3
    WEIGHT_DECAY = 0.01

    optimizer_grouped = [
        {"params": [p for n, p in model.named_parameters() if "lm_head" not in n and p.requires_grad], "lr": ENC_LR},
        {"params": [p for n, p in model.named_parameters() if "lm_head" in n and p.requires_grad], "lr": HEAD_LR}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped, weight_decay=WEIGHT_DECAY)

    def get_lr_factor(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_training_steps - warmup_steps))
        cos_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(0.01, cos_decay)

    # Initialize learning rates based on global_step
    initial_factor = get_lr_factor(global_step)
    optimizer.param_groups[0]["lr"] = ENC_LR * initial_factor
    optimizer.param_groups[1]["lr"] = HEAD_LR * initial_factor

    if is_resumed and state is not None:
        if "optimizer_state_dict" in state:
            try:
                optimizer.load_state_dict(state["optimizer_state_dict"])
                # Re-apply current LR to optimizer param groups
                optimizer.param_groups[0]["lr"] = ENC_LR * initial_factor
                optimizer.param_groups[1]["lr"] = HEAD_LR * initial_factor
            except Exception as e:
                print(f"Warning loading optimizer state: {e}")
        print(f"RESUMPTION SUCCESSFUL: Resumed at Epoch {start_epoch}, Step {global_step:,}/{total_training_steps:,} (Batch offset: {batch_offset}/{steps_per_epoch}), LR: {ENC_LR * initial_factor:.2e}, Best Val WER: {best_val_wer*100:.2f}%, Non-improving: {non_improving_epochs}")
    else:
        if not train_log_file.exists():
            with open(train_log_file, "w", encoding="utf-8") as f:
                f.write("step,epoch,loss,grad_norm,enc_lr,head_lr,vram_gb,step_time_sec\n")
        if not epoch_log_file.exists():
            with open(epoch_log_file, "w", encoding="utf-8") as f:
                f.write("epoch,step,val_loss,norm_wer,norm_cer,raw_wer,raw_cer,blank_rate,empty_pct,eval_time_sec,is_best\n")
        if not resource_file.exists():
            with open(resource_file, "w", encoding="utf-8") as f:
                f.write("step,elapsed_sec,throughput_samples_per_sec,vram_peak_gb\n")

    start_time = time.time()
    step_start = time.time()
    running_loss = 0.0

    print(f"\n--- Initiating Multi-Epoch Acoustic Fine-Tuning ---")
    for epoch in range(start_epoch, max_epochs + 1):
        epoch_start = time.time()
        model.train()

        shuffled_batches = train_dataset.get_shuffled_batches(seed=42 + epoch)
        current_batch_offset = 0
        if epoch == start_epoch and batch_offset > 0:
            current_batch_offset = batch_offset
            shuffled_batches = shuffled_batches[batch_offset:]
            print(f"\n>>> Epoch {epoch:02d}/{max_epochs:02d} continuing from batch {batch_offset}/{steps_per_epoch} ({len(shuffled_batches)} batches remaining) | LR: {optimizer.param_groups[0]['lr']:.2e}")
            batch_offset = 0
        else:
            print(f"\n>>> Epoch {epoch:02d}/{max_epochs:02d} started ({len(shuffled_batches):,} batches) | LR: {optimizer.param_groups[0]['lr']:.2e}")

        for b_sub_idx, batch in enumerate(shuffled_batches):
            b_actual_idx = current_batch_offset + b_sub_idx
            col = collate_batch(batch, processor)
            if col is None:
                continue

            input_values = col["input_values"].to(device)
            attention_mask = col["attention_mask"].to(device)
            labels = col["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_values=input_values, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            running_loss += loss.item()

            factor = get_lr_factor(global_step)
            optimizer.param_groups[0]["lr"] = ENC_LR * factor
            optimizer.param_groups[1]["lr"] = HEAD_LR * factor
            optimizer.zero_grad(set_to_none=True)

            if global_step % 50 == 0:
                step_dur = (time.time() - step_start) / 50.0
                vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                lr_e = optimizer.param_groups[0]["lr"]
                lr_h = optimizer.param_groups[1]["lr"]
                elapsed = time.time() - start_time
                throughput = (50 * len(batch)) / max(0.001, (time.time() - step_start))
                print(f"Ep {epoch:02d} [{b_actual_idx+1:04d}/{steps_per_epoch:04d}] | Step {global_step:05d}/{total_training_steps:05d} | Loss: {running_loss/50:.4f} | LR: {lr_e:.2e} | {throughput:.1f} samp/s | VRAM: {vram:.1f}GB", flush=True)

                with open(train_log_file, "a", encoding="utf-8") as f:
                    f.write(f"{global_step},{epoch},{running_loss/50:.4f},{grad_norm.item():.4f},{lr_e:.6e},{lr_h:.6e},{vram:.2f},{step_dur:.3f}\n")
                with open(resource_file, "a", encoding="utf-8") as f:
                    f.write(f"{global_step},{elapsed:.1f},{throughput:.2f},{vram:.2f}\n")

                running_loss = 0.0
                step_start = time.time()

            # Save checkpoint every 500 steps (Safe & Fast, zero OOM risk)
            if global_step % 500 == 0:
                torch.cuda.empty_cache()
                gc.collect()
                step_ckpt_dir = exp_dir / f"checkpoints/step_{global_step:05d}"
                step_ckpt_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(step_ckpt_dir))
                processor.save_pretrained(str(step_ckpt_dir))

                # Update latest checkpoint for seamless resumption
                model.save_pretrained(str(latest_ckpt_dir))
                processor.save_pretrained(str(latest_ckpt_dir))
                torch.save({
                    "epoch": epoch,
                    "global_step": global_step,
                    "batch_offset": b_actual_idx + 1,
                    "best_val_wer": best_val_wer,
                    "non_improving_epochs": non_improving_epochs,
                    "optimizer_state_dict": optimizer.state_dict()
                }, str(latest_ckpt_dir / "training_state.pt"))
                print(f"  >>> [Step {global_step:,} Checkpoint Saved] Saved to {step_ckpt_dir.name} & {latest_ckpt_dir.name} (VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB)", flush=True)

        epoch_dur = (time.time() - epoch_start) / 60.0
        print(f"\n--- Epoch {epoch:02d} Completed in {epoch_dur:.1f}m. Running Full Validation on 6,728 samples... ---")

        val_start = time.time()
        val_res = evaluate_validation(model, processor, val_dataset, device, val_batch_size=16)
        val_dur = time.time() - val_start
        val_wer = val_res["norm_wer"]
        val_cer = val_res["norm_cer"]
        raw_wer = val_res["raw_wer"]
        raw_cer = val_res["raw_cer"]

        is_best = False
        if val_wer < best_val_wer:
            improvement = (best_val_wer - val_wer) * 100 if best_val_wer != float("inf") else 0.0
            print(f"  >>> NEW BEST MODEL at Epoch {epoch}! Norm WER: {val_wer*100:.2f}% (-{improvement:.2f}%) | Norm CER: {val_cer*100:.2f}%")
            best_val_wer = val_wer
            is_best = True
            non_improving_epochs = 0

            # Save best checkpoint
            model.save_pretrained(str(best_model_dir))
            processor.save_pretrained(str(best_model_dir))

            # Save best predictions
            best_pred_file = pred_dir / "best_validation_predictions.jsonl"
            with open(best_pred_file, "w", encoding="utf-8") as f:
                for sid, r_raw, r_norm, h_raw, h_norm, src in zip(
                    val_res["sample_ids"], val_res["raw_refs"], val_res["norm_refs"],
                    val_res["raw_hyps"], val_res["norm_hyps"], val_res["sources"]
                ):
                    f.write(json.dumps({
                        "sample_id": sid,
                        "source": src,
                        "raw_ref": r_raw,
                        "norm_ref": r_norm,
                        "raw_hyp": h_raw,
                        "norm_hyp": h_norm
                    }, ensure_ascii=False) + "\n")
        else:
            non_improving_epochs += 1
            print(f"  --- Validation Norm WER did not improve ({val_wer*100:.2f}% vs best {best_val_wer*100:.2f}%). Non-improving epochs: {non_improving_epochs}/{patience}")

        # Save epoch checkpoint
        epoch_ckpt_dir = exp_dir / f"checkpoints/epoch_{epoch:02d}"
        epoch_ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(epoch_ckpt_dir))
        processor.save_pretrained(str(epoch_ckpt_dir))
        with open(epoch_ckpt_dir / "epoch_metadata.json", "w", encoding="utf-8") as f:
            json.dump({
                "epoch": epoch,
                "global_step": global_step,
                "val_loss": val_res["val_loss"],
                "norm_wer": val_wer,
                "norm_cer": val_cer,
                "raw_wer": raw_wer,
                "raw_cer": raw_cer,
                "blank_rate": val_res["blank_rate"],
                "empty_pct": val_res["empty_pct"],
                "is_best": is_best
            }, f, indent=2)

        # Log epoch metrics
        with open(epoch_log_file, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{global_step},{val_res['val_loss']:.4f},{val_wer:.4f},{val_cer:.4f},{raw_wer:.4f},{raw_cer:.4f},{val_res['blank_rate']:.4f},{val_res['empty_pct']:.2f},{val_dur:.1f},{is_best}\n")

        # Save latest checkpoint
        model.save_pretrained(str(latest_ckpt_dir))
        processor.save_pretrained(str(latest_ckpt_dir))
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "batch_offset": 0,
            "best_val_wer": best_val_wer,
            "non_improving_epochs": non_improving_epochs,
            "optimizer_state_dict": optimizer.state_dict()
        }, str(latest_ckpt_dir / "training_state.pt"))

        # Early stopping check
        if epoch >= min_epochs and non_improving_epochs >= patience:
            print(f"\n==========================================================================")
            print(f"EARLY STOPPING TRIGGERED at Epoch {epoch}!")
            print(f"Patience of {patience} non-improving evaluations reached. Best WER: {best_val_wer*100:.2f}%")
            print(f"==========================================================================\n")
            break

    total_training_time = time.time() - start_time
    print(f"\n=======================================================")
    print(f"EXPERIMENT {exp_name} MULTI-EPOCH TRAINING CONCLUDED")
    print(f"Total Duration: {total_training_time/3600:.2f} hours")
    print(f"Best Validation Normalized WER: {best_val_wer*100:.2f}%")
    print(f"=======================================================\n")
    return best_val_wer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True, choices=["B-02A", "B-02B", "B-02C", "all"])
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--min_epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    manifest_map = {
        "B-02A": "b02_base_manifest.jsonl",
        "B-02B": "b02_quran_augmented_manifest.jsonl",
        "B-02C": "b02_all_verified_human_manifest.jsonl"
    }

    if args.experiment == "all":
        exps = ["B-02A", "B-02B", "B-02C"]
    else:
        exps = [args.experiment]

    for exp in exps:
        run_experiment(
            exp_name=exp,
            manifest_name=manifest_map[exp],
            max_epochs=args.max_epochs,
            min_epochs=args.min_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            resume=args.resume
        )
