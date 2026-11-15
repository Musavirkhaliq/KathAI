#!/usr/bin/env python3
"""
EduCasheer Kashmiri AI — Stage 10: Continual Acoustic Adaptation Engine
========================================================================
Performs targeted continual fine-tuning from our best checkpoint (B-02B)
on previously unused, strictly non-test, non-val Kashmiri speech:
- Source checkpoint: B-02B best model (29.32% greedy WER)
- Dataset: 45,031 utterances (~88.83 hours) of previously unseen data
- Guardrails: 100% exclusion of val.jsonl and test.jsonl
- Optimization: Conservative fine-tuning LR (3e-5 encoder, 1e-4 head) with cosine decay
- Memory: BF16 AMP + expandable segments + batch_size=16 for low VRAM coexistence
- Validation: Full validation on all 6,728 validation samples at each epoch boundary
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
STAGE10_DIR = EXP_ROOT / "stage10_continual_training"
BEST_B02B_DIR = EXP_ROOT / "stage9_b02_acoustic_comparison/B-02B/checkpoints/best_model"
TOK_DIR = EXP_ROOT / "B-01-xlsr300m-character-ctc/tokenizer"
VAL_MANIFEST = DATASET_DIR / "experiments/native_asr_training/data_splits/val.jsonl"
ADAPTATION_MANIFEST = STAGE10_DIR / "unused_adaptation_manifest.jsonl"

sys.path.insert(0, str(DATASET_DIR / "scripts"))
from metrics_evaluator import calculate_wer_and_cer, normalize_perso_arabic_kashmiri

_RESAMPLERS = {}
def get_resampler(orig_sr, target_sr=16000):
    if orig_sr not in _RESAMPLERS:
        import torchaudio
        _RESAMPLERS[orig_sr] = torchaudio.transforms.Resample(orig_sr, target_sr)
    return _RESAMPLERS[orig_sr]


class LengthBucketedDataset:
    def __init__(self, manifest_file, max_duration=16.0, min_duration=0.5, batch_size=16):
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
            print(f"  Filtered {filtered_ctc} CTC length violations (T < U).")

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
        audio_p = Path(item["audio_path"])
        if not audio_p.is_absolute():
            audio_p = WORKSPACE_DIR / audio_p
        try:
            audio, sr = sf.read(str(audio_p), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                audio_t = torch.from_numpy(audio)
                resampler = get_resampler(sr, 16000)
                audio = resampler(audio_t).numpy()
            wavs.append(audio)
            norm_texts.append(item["normalized_text"])
            raw_texts.append(item["raw_text"])
            sample_ids.append(item["sample_id"])
            durations.append(item["duration"])
            sources.append(item["source"])
        except Exception:
            continue

    if not wavs:
        return None

    inputs = processor(wavs, sampling_rate=16000, return_tensors="pt", padding=True)
    with processor.as_target_processor():
        labels_batch = processor(norm_texts, return_tensors="pt", padding=True)

    labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

    return {
        "input_values": inputs.input_values,
        "attention_mask": inputs.attention_mask,
        "labels": labels,
        "norm_texts": norm_texts,
        "raw_texts": raw_texts,
        "sample_ids": sample_ids,
        "durations": durations,
        "sources": sources
    }


def evaluate_validation(model, processor, val_dataset, device):
    model.eval()
    val_loss_total = 0.0
    val_batches = 0
    
    all_raw_refs = []
    all_norm_refs = []
    all_raw_hyps = []
    all_norm_hyps = []
    all_sample_ids = []
    all_sources = []
    
    total_blank_frames = 0
    total_frames = 0
    empty_hyps = 0

    with torch.no_grad():
        for batch_items in val_dataset.batches:
            batch = collate_batch(batch_items, processor)
            if batch is None:
                continue

            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_values=input_values, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

            val_loss_total += loss.item()
            val_batches += 1

            pred_ids = torch.argmax(logits, dim=-1)
            pad_id = processor.tokenizer.pad_token_id

            for i in range(pred_ids.size(0)):
                mask_len = attention_mask[i].sum().item()
                subsampled_len = model._get_feat_extract_output_lengths(mask_len)
                valid_preds = pred_ids[i, :subsampled_len]

                total_blank_frames += (valid_preds == pad_id).sum().item()
                total_frames += subsampled_len

                pred_str = processor.decode(valid_preds.tolist(), group_tokens=True)
                norm_hyp = normalize_perso_arabic_kashmiri(pred_str)

                if not norm_hyp.strip():
                    empty_hyps += 1

                all_sample_ids.append(batch["sample_ids"][i])
                all_raw_refs.append(batch["raw_texts"][i])
                all_norm_refs.append(batch["norm_texts"][i])
                all_raw_hyps.append(pred_str)
                all_norm_hyps.append(norm_hyp)
                all_sources.append(batch["sources"][i])

    avg_val_loss = val_loss_total / max(1, val_batches)
    blank_rate = total_blank_frames / max(1, total_frames)
    empty_pct = empty_hyps / max(1, len(all_norm_hyps))

    # Standard metrics
    metrics = calculate_wer_and_cer(all_norm_refs, all_norm_hyps)
    raw_metrics = calculate_wer_and_cer(all_raw_refs, all_raw_hyps)

    return {
        "val_loss": avg_val_loss,
        "norm_wer": metrics["wer"],
        "norm_cer": metrics["cer"],
        "raw_wer": raw_metrics["wer"],
        "raw_cer": raw_metrics["cer"],
        "blank_rate": blank_rate,
        "empty_pct": empty_pct,
        "sample_ids": all_sample_ids,
        "norm_refs": all_norm_refs,
        "norm_hyps": all_norm_hyps,
        "sources": all_sources
    }


def run_continual_training(max_epochs=5, min_epochs=2, patience=2, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt_dir = STAGE10_DIR / "checkpoints"
    best_model_dir = ckpt_dir / "best_model"
    latest_ckpt_dir = ckpt_dir / "latest_checkpoint"
    log_dir = STAGE10_DIR / "logs"
    pred_dir = STAGE10_DIR / "predictions"
    report_dir = STAGE10_DIR / "reports"

    for d in [best_model_dir, latest_ckpt_dir, log_dir, pred_dir, report_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STAGE 10: CONTINUAL ACOUSTIC ADAPTATION ON UNUSED KASHMIRI SPEECH")
    print(f"Base Checkpoint: {BEST_B02B_DIR}")
    print(f"Adaptation Manifest: {ADAPTATION_MANIFEST}")
    print(f"Max Epochs: {max_epochs} | Min Epochs: {min_epochs} | Patience: {patience}")
    print(f"Batch Size: {batch_size} | BF16 AMP: True | Device: {device}")
    print("=" * 80)

    processor = Wav2Vec2Processor.from_pretrained(str(TOK_DIR))
    train_dataset = LengthBucketedDataset(ADAPTATION_MANIFEST, max_duration=16.0, min_duration=0.5, batch_size=batch_size)
    val_dataset = LengthBucketedDataset(VAL_MANIFEST, max_duration=16.0, min_duration=0.5, batch_size=16)

    steps_per_epoch = len(train_dataset.batches)
    total_training_steps = max_epochs * steps_per_epoch
    warmup_steps = int(0.10 * total_training_steps)

    print(f"Dataset Structure: {len(train_dataset.samples):,} samples -> {steps_per_epoch:,} steps/epoch | Total Planned: {total_training_steps:,} steps")

    train_log_file = log_dir / "training_log.csv"
    epoch_log_file = log_dir / "epoch_metrics.csv"
    resource_file = log_dir / "resource_usage.csv"

    if not train_log_file.exists():
        with open(train_log_file, "w", encoding="utf-8") as f:
            f.write("step,epoch,train_loss,grad_norm,enc_lr,head_lr,vram_used_gb,step_time_sec\n")

    if not epoch_log_file.exists():
        with open(epoch_log_file, "w", encoding="utf-8") as f:
            f.write("epoch,step,val_loss,norm_wer,norm_cer,raw_wer,raw_cer,blank_rate,empty_pct,eval_time_sec,is_best\n")

    if not resource_file.exists():
        with open(resource_file, "w", encoding="utf-8") as f:
            f.write("timestamp,epoch,step,gpu_util,vram_used_mb,vram_total_mb\n")

    print(f"Loading weights from best B-02B model checkpoint: {BEST_B02B_DIR}...")
    model = Wav2Vec2ForCTC.from_pretrained(str(BEST_B02B_DIR), ctc_zero_infinity=True).to(device)
    model.freeze_feature_encoder()

    # Conservative learning rates for continual adaptation
    ENC_LR = 3.0e-5
    HEAD_LR = 1.0e-4
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
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    print("\nRunning initial zero-step validation evaluation on full dev set...")
    t0_eval = time.time()
    init_eval = evaluate_validation(model, processor, val_dataset, device)
    eval_dur = time.time() - t0_eval
    print(f"Initial Baseline Checkpoint Score: Val Loss: {init_eval['val_loss']:.4f} | WER: {init_eval['norm_wer']:.2f}% | CER: {init_eval['norm_cer']:.2f}% (Eval time: {eval_dur:.1f}s)")

    best_val_wer = init_eval["norm_wer"]
    non_improving_epochs = 0
    global_step = 0

    print("\nStarting Continual Adaptation Training...")
    
    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        model.train()
        shuffled_batches = train_dataset.get_shuffled_batches(seed=42 + epoch)
        running_loss = 0.0
        running_steps = 0
        
        for batch_idx, batch_items in enumerate(shuffled_batches):
            step_start = time.time()
            batch = collate_batch(batch_items, processor)
            if batch is None:
                continue

            input_values = batch["input_values"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_values=input_values, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[Warning] NaN/Inf loss encountered at step {global_step}. Skipping batch.")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            running_loss += loss.item()
            running_steps += 1
            step_time = time.time() - step_start

            # Log every 50 steps
            if global_step % 50 == 0:
                avg_loss = running_loss / max(1, running_steps)
                enc_lr = optimizer.param_groups[0]["lr"]
                head_lr = optimizer.param_groups[1]["lr"]
                vram_gb = torch.cuda.memory_allocated() / 1e9

                with open(train_log_file, "a", encoding="utf-8") as f:
                    f.write(f"{global_step},{epoch},{avg_loss:.4f},{grad_norm:.4f},{enc_lr:.6e},{head_lr:.6e},{vram_gb:.2f},{step_time:.3f}\n")

                print(f"Epoch {epoch:2d}/{max_epochs} | Step {global_step:5d}/{total_training_steps} | Loss: {avg_loss:.4f} | Grad: {grad_norm:.2f} | LR: {enc_lr:.2e} | VRAM: {vram_gb:.1f}GB | {step_time:.2f}s/step")
                running_loss = 0.0
                running_steps = 0

            # Periodic checkpoint saving every 500 steps
            if global_step % 500 == 0:
                print(f"Saving checkpoint at step {global_step} to {latest_ckpt_dir}...")
                model.save_pretrained(str(latest_ckpt_dir))
                processor.save_pretrained(str(latest_ckpt_dir))
                torch.save({
                    "epoch": epoch,
                    "global_step": global_step,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_wer": best_val_wer,
                    "non_improving_epochs": non_improving_epochs
                }, latest_ckpt_dir / "training_state.pt")

        # Epoch Boundary Evaluation
        epoch_dur = time.time() - epoch_start
        print(f"\nCompleted Epoch {epoch} in {epoch_dur/60:.1f} minutes. Running full validation evaluation...")
        t0_eval = time.time()
        val_res = evaluate_validation(model, processor, val_dataset, device)
        eval_dur = time.time() - t0_eval

        val_wer = val_res["norm_wer"]
        val_cer = val_res["norm_cer"]
        is_best = val_wer < best_val_wer

        print(f"--> Epoch {epoch} Val WER: {val_wer:.2f}% (CER: {val_cer:.2f}%, Loss: {val_res['val_loss']:.4f}) | Best: {best_val_wer:.2f}% | New Best: {is_best}")

        with open(epoch_log_file, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{global_step},{val_res['val_loss']:.4f},{val_wer:.4f},{val_cer:.4f},{val_res['raw_wer']:.4f},{val_res['raw_cer']:.4f},{val_res['blank_rate']:.4f},{val_res['empty_pct']:.4f},{eval_dur:.1f},{is_best}\n")

        # Save epoch checkpoint
        epoch_ckpt = ckpt_dir / f"epoch_{epoch:02d}"
        epoch_ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(epoch_ckpt))
        processor.save_pretrained(str(epoch_ckpt))

        if is_best:
            best_val_wer = val_wer
            non_improving_epochs = 0
            print(f"*** Saving new BEST MODEL to {best_model_dir} (WER: {best_val_wer:.2f}%) ***")
            model.save_pretrained(str(best_model_dir))
            processor.save_pretrained(str(best_model_dir))
            # Save predictions
            with open(pred_dir / f"val_preds_epoch_{epoch:02d}.jsonl", "w", encoding="utf-8") as f:
                for sid, r, h, src in zip(val_res["sample_ids"], val_res["norm_refs"], val_res["norm_hyps"], val_res["sources"]):
                    f.write(json.dumps({"sample_id": sid, "norm_ref": r, "norm_hyp": h, "source": src}, ensure_ascii=False) + "\n")
        else:
            non_improving_epochs += 1
            print(f"No improvement for {non_improving_epochs} epoch(s). (Patience: {patience})")

        if epoch >= min_epochs and non_improving_epochs >= patience:
            print(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
            break

    print("\n" + "=" * 80)
    print(f"STAGE 10 CONTINUAL ADAPTATION COMPLETE! Best Validation WER: {best_val_wer:.2f}%")
    print(f"Best model stored at: {best_model_dir}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--min_epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    run_continual_training(
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        patience=args.patience,
        batch_size=args.batch_size
    )
