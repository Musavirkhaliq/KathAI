#!/usr/bin/env python3
"""
ASR Acoustic Fine-Tuning: Continuation from Champion E22-01 on All Clean Discovered Data

Initialization:
  - Model: E22-01/best_model (Wav2Vec2-XLSR-300M + 161-dim Char CTC)
Dataset:
  - Clean Master Training Mixture: 83,613 utterances (~160.3 hours)
  - Strict Exclusion: All test sets, validation sets, benchmark datasets, and colliding speakers are strictly removed.
Training Setup:
  - Batch size: 16 per device, Gradient Accumulation: 2 (Effective Batch: 32)
  - Epochs: 1 full epoch (~2,612 steps)
  - Learning rate: 3e-6 with Cosine decay and 100 warmup steps
  - Mixed Precision: bfloat16
  - Feature Encoder: Frozen
  - Evaluation: Every 250 steps on Dev B (WAAZ Val) and Dev A Sample (Legacy Val)
  - Checkpoint: Saves best model on combined validation score and periodic checkpoints
  - Final Evaluation: Comprehensive comparison of E22-01 vs New Best Checkpoint on Dev B and Dev A
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import json
import math
import time
import shutil
import random
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import jiwer
from pyctcdecode import build_ctcdecoder

WORKSPACE = Path("/home/musa/Kashmiri Project")
sys.path.insert(0, str(WORKSPACE))

from kashmiri_dataset.scripts.metrics_evaluator import normalize_perso_arabic_kashmiri as norm_strict

INIT_CHECKPOINT = WORKSPACE / "kashmiri_dataset/experiments/stage22_data_expanded_training/checkpoints/E22-01/best_model"
TRAIN_MANIFEST = WORKSPACE / "kashmiri_dataset/experiments/all_data_expanded_training/manifests/data_all_clean_verified.jsonl"
EXP_DIR = WORKSPACE / "kashmiri_dataset/experiments/all_data_expanded_training"
CHECKPOINTS_DIR = EXP_DIR / "checkpoints"
BEST_MODEL_DIR = CHECKPOINTS_DIR / "best_model"

DEV_A_PATH = WORKSPACE / "kashmiri_dataset/releases/asr_training_v2_waaz/validation_manifests/dev_a_legacy_val.jsonl"
DEV_B_PATH = WORKSPACE / "kashmiri_dataset/releases/asr_training_v2_waaz/validation_manifests/dev_b_waaz_val.jsonl"
LM_PATH = WORKSPACE / "kashmiri_dataset/experiments/native_asr_training/experiments/stage7_ctc_lm_decoding/language_models/lm_word_4gram_domain_d.arpa"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)

class SpeechASRDataset(Dataset):
    def __init__(self, manifest_path, processor, max_duration=25.0, min_duration=0.5):
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.records = []
        
        print(f"[Dataset] Loading manifest {manifest_path}...")
        with open(manifest_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                d = json.loads(line.strip())
                dur = float(d.get("duration") or d.get("duration_seconds") or 0.0)
                if dur < min_duration or dur > max_duration:
                    continue
                text = d.get("text") or d.get("normalized_text") or ""
                norm_t = norm_strict(text)
                if not norm_t.strip():
                    continue
                
                ap = d.get("audio_path") or d.get("processed_file_path") or d.get("path")
                if not os.path.isabs(ap):
                    ap = str(WORKSPACE / ap)
                
                tokens = processor.tokenizer(norm_t).input_ids
                if not tokens:
                    continue

                self.records.append({
                    "sample_id": d.get("sample_id", f"s_{idx}"),
                    "audio_path": ap,
                    "labels": tokens,
                    "source": d.get("source_dataset") or d.get("source", "unknown"),
                    "duration": dur
                })
        print(f"[Dataset] Loaded {len(self.records)} valid training records with pre-tokenized targets.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            wav, sr = sf.read(rec["audio_path"], dtype="float32")
            if wav.ndim > 1:
                wav = np.mean(wav, axis=1)
            if sr != 16000:
                import torchaudio.functional as F_audio
                w_tensor = torch.from_numpy(wav).float()
                w_tensor = F_audio.resample(w_tensor, orig_freq=sr, new_freq=16000)
                wav = w_tensor.numpy()
        except Exception:
            wav = np.zeros(16000, dtype=np.float32)

        return {
            "wav": wav,
            "labels": rec["labels"],
            "source": rec["source"],
            "sample_id": rec["sample_id"]
        }

class CollateCTC:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        batch = [b for b in batch if len(b["labels"]) > 0 and len(b["wav"]) > 160]
        if not batch:
            return None

        wavs = [b["wav"] for b in batch]
        features = self.processor(wavs, sampling_rate=16000, padding=True, return_tensors="pt")

        label_tensors = [torch.tensor(b["labels"], dtype=torch.long) for b in batch]
        target_lengths = torch.tensor([len(b["labels"]) for b in batch], dtype=torch.long)
        targets = torch.cat(label_tensors)

        return {
            "input_values": features.input_values,
            "attention_mask": features.get("attention_mask", None),
            "targets": targets,
            "target_lengths": target_lengths,
            "sources": [b["source"] for b in batch],
            "sample_ids": [b["sample_id"] for b in batch]
        }

def evaluate_fast_val(model, processor, records, device="cuda"):
    model.eval()
    total_loss = 0.0
    total_ref_words = 0
    total_errs = 0
    evaluated_count = 0

    with torch.no_grad():
        for rec in records:
            ap = rec.get("audio_path") or rec.get("processed_file_path") or rec.get("path")
            if not os.path.isabs(ap):
                ap = str(WORKSPACE / ap)

            try:
                wav, sr = sf.read(ap, dtype="float32")
                if wav.ndim > 1:
                    wav = np.mean(wav, axis=1)
                if sr != 16000:
                    import torchaudio.functional as F_audio
                    w_tensor = torch.from_numpy(wav).float()
                    w_tensor = F_audio.resample(w_tensor, orig_freq=sr, new_freq=16000)
                    wav = w_tensor.numpy()
            except Exception:
                continue

            ref_norm = norm_strict(rec.get("text") or rec.get("normalized_text") or "")
            ref_words = len(ref_norm.split())
            if ref_words == 0 or len(wav) < 1600:
                continue

            inputs = processor(wav, sampling_rate=16000, return_tensors="pt").input_values.to(device)
            ref_tokens = processor.tokenizer(ref_norm).input_ids
            if not ref_tokens:
                continue

            with torch.amp.autocast(device, dtype=torch.bfloat16):
                logits = model(inputs).logits
                log_probs = torch.log_softmax(logits, dim=-1).permute(1, 0, 2)
                tgt_tensor = torch.tensor(ref_tokens, dtype=torch.long, device=device)
                in_lens = torch.tensor([logits.shape[1]], dtype=torch.long, device=device)
                tgt_lens = torch.tensor([len(ref_tokens)], dtype=torch.long, device=device)
                loss = F.ctc_loss(log_probs, tgt_tensor, in_lens, tgt_lens, blank=0, zero_infinity=True)

            p_ids = torch.argmax(logits[0], dim=-1)
            hyp = norm_strict(processor.decode(p_ids))
            res = jiwer.process_words(ref_norm, hyp)
            err = res.substitutions + res.deletions + res.insertions

            total_loss += loss.item()
            total_errs += err
            total_ref_words += ref_words
            evaluated_count += 1

    model.train()
    model.freeze_feature_encoder()
    avg_loss = total_loss / max(1, evaluated_count)
    wer = (total_errs / max(1, total_ref_words)) * 100.0 if total_ref_words > 0 else 0.0
    return avg_loss, wer, evaluated_count

def full_benchmark_eval(model_dir, kenlm_decoder=None, max_dev_a_greedy=1000, max_dev_a_kenlm=150):
    print(f"\n[Benchmark Eval] Loading model from {model_dir} for evaluation...")
    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(model_dir).to("cuda")
    model.eval()

    splits = {
        "dev_b_waaz_val": (DEV_B_PATH, 127, 127),
        "dev_a_legacy_val": (DEV_A_PATH, max_dev_a_greedy, max_dev_a_kenlm)
    }

    results = {}
    with torch.no_grad():
        for split_name, (split_path, max_items_greedy, max_items_kenlm) in splits.items():
            records = []
            with open(split_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line.strip()))
            
            random.Random(42).shuffle(records)
            records_greedy = records[:max_items_greedy]

            greedy_errs, kenlm_errs = 0, 0
            ref_words_greedy, ref_words_kenlm = 0, 0
            all_refs_g, all_hyps_g = [], []
            kenlm_count = 0

            for idx, rec in enumerate(records_greedy):
                ap = rec.get("audio_path") or rec.get("path")
                if not os.path.isabs(ap):
                    ap = str(WORKSPACE / ap)

                try:
                    wav, sr = sf.read(ap, dtype="float32")
                    if wav.ndim > 1:
                        wav = np.mean(wav, axis=1)
                    if sr != 16000:
                        import torchaudio.functional as F_audio
                        w_tensor = torch.from_numpy(wav).float()
                        w_tensor = F_audio.resample(w_tensor, orig_freq=sr, new_freq=16000)
                        wav = w_tensor.numpy()
                except Exception:
                    continue

                ref = norm_strict(rec.get("text") or rec.get("normalized_text") or "")
                rw = len(ref.split())
                if rw == 0 or len(wav) < 1600:
                    continue

                inputs = processor(wav, sampling_rate=16000, return_tensors="pt").input_values.to("cuda")
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(inputs).logits[0]

                # Greedy decode
                pred_ids = torch.argmax(logits, dim=-1)
                hyp_greedy = norm_strict(processor.decode(pred_ids))
                res_g = jiwer.process_words(ref, hyp_greedy)
                greedy_errs += (res_g.substitutions + res_g.deletions + res_g.insertions)
                ref_words_greedy += rw
                all_refs_g.append(ref)
                all_hyps_g.append(hyp_greedy)

                # KenLM decode
                if kenlm_decoder is not None and kenlm_count < max_items_kenlm:
                    logits_np = logits.float().cpu().numpy()
                    hyp_kenlm = norm_strict(kenlm_decoder.decode(logits_np))
                    res_k = jiwer.process_words(ref, hyp_kenlm)
                    kenlm_errs += (res_k.substitutions + res_k.deletions + res_k.insertions)
                    ref_words_kenlm += rw
                    kenlm_count += 1

            g_wer = (greedy_errs / ref_words_greedy) * 100.0 if ref_words_greedy > 0 else 0.0
            k_wer = (kenlm_errs / ref_words_kenlm) * 100.0 if ref_words_kenlm > 0 else g_wer

            # CER
            res_cer = jiwer.process_characters(all_refs_g, all_hyps_g)
            cer = (res_cer.substitutions + res_cer.deletions + res_cer.insertions) / max(1, sum(len(r) for r in all_refs_g)) * 100.0

            results[split_name] = {
                "total_utterances": len(all_refs_g),
                "total_ref_words": ref_words_greedy,
                "greedy_wer": round(g_wer, 2),
                "kenlm_wer": round(k_wer, 2),
                "kenlm_utts_evaluated": kenlm_count,
                "greedy_cer": round(cer, 2)
            }
            print(f"  [{split_name}] utts={len(all_refs_g)} | Greedy WER={g_wer:.2f}% | KenLM WER={k_wer:.2f}% (on {kenlm_count} utts) | CER={cer:.2f}%")

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per device")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-6, help="Learning rate")
    parser.add_argument("--eval_every", type=int, default=250, help="Evaluation step frequency")
    parser.add_argument("--max_steps", type=int, default=None, help="Override total steps")
    args = parser.parse_args()

    set_seed(42)
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ALL-DATA ASR ACOUSTIC FINE-TUNING CONTINUATION FROM E22-01")
    print(f"Initialization: {INIT_CHECKPOINT}")
    print(f"Training Manifest: {TRAIN_MANIFEST}")
    print(f"Checkpoints Root: {CHECKPOINTS_DIR}")
    print(f"Learning Rate: {args.lr} | Batch Size: {args.batch_size} * {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print("=" * 80)

    # 1. Load Model & Processor
    print("\n[1/5] Loading Processor & E22-01 Model...")
    processor = Wav2Vec2Processor.from_pretrained(INIT_CHECKPOINT)
    model = Wav2Vec2ForCTC.from_pretrained(INIT_CHECKPOINT).to("cuda")
    model.train()
    model.freeze_feature_encoder()
    assert model.config.vocab_size == 161, f"Vocab mismatch: {model.config.vocab_size} != 161"
    print(f"[Model] Feature encoder frozen. Vocab size: {model.config.vocab_size}. Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # Initial copy of INIT_CHECKPOINT to BEST_MODEL_DIR so best_model always exists
    if not (BEST_MODEL_DIR / "model.safetensors").exists():
        print(f"[Init] Initializing {BEST_MODEL_DIR} with baseline E22-01...")
        model.save_pretrained(BEST_MODEL_DIR)
        processor.save_pretrained(BEST_MODEL_DIR)

    # 2. Setup KenLM decoder for evaluations
    print("\n[2/5] Building Calibrated KenLM CTC Decoder (alpha=0.20, beta=-0.50)...")
    vocab_dict = processor.tokenizer.get_vocab()
    sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
    labels = [token if token != "|" else " " for token, idx in sorted_vocab]

    try:
        kenlm_decoder = build_ctcdecoder(
            labels=labels,
            kenlm_model_path=str(LM_PATH),
            alpha=0.20,
            beta=-0.50,
            unk_score_offset=-10.0
        )
        print("[KenLM] Decoder built successfully.")
    except Exception as e:
        print(f"[KenLM] Warning: Could not build KenLM decoder: {e}. Falling back to greedy only.")
        kenlm_decoder = None

    # 3. Load Validation Sets
    print("\n[3/5] Loading Validation Sets...")
    dev_b_records = []
    with open(DEV_B_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dev_b_records.append(json.loads(line.strip()))

    dev_a_records = []
    with open(DEV_A_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dev_a_records.append(json.loads(line.strip()))
    random.Random(42).shuffle(dev_a_records)
    dev_a_sample = dev_a_records[:200]
    print(f"[Validation] Dev B size: {len(dev_b_records)} | Dev A sample size: {len(dev_a_sample)}")

    # Pre-train baseline evaluation on E22-01
    print("\n[Pre-Train Check] Evaluating starting E22-01 model...")
    init_b_loss, init_b_wer, _ = evaluate_fast_val(model, processor, dev_b_records)
    init_a_loss, init_a_wer, _ = evaluate_fast_val(model, processor, dev_a_sample)
    init_score = 0.5 * init_b_wer + 0.5 * init_a_wer
    print(f"[Baseline E22-01] Dev B WER: {init_b_wer:.2f}% (loss: {init_b_loss:.4f}) | Dev A Sample WER: {init_a_wer:.2f}% (loss: {init_a_loss:.4f}) | Combined: {init_score:.2f}")

    # 4. Build Dataset & DataLoader
    print("\n[4/5] Building Training DataLoader...")
    train_dataset = SpeechASRDataset(TRAIN_MANIFEST, processor, max_duration=25.0, min_duration=0.5)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=CollateCTC(processor),
        num_workers=0,
        pin_memory=True
    )

    steps_per_epoch = len(train_dataset) // (args.batch_size * args.grad_accum)
    total_steps = args.max_steps if args.max_steps else steps_per_epoch * args.epochs
    warmup_steps = min(100, int(0.05 * total_steps))

    print(f"[Training] Dataset items: {len(train_dataset)} | Steps/Epoch: {steps_per_epoch} | Total Steps: {total_steps} | Warmup: {warmup_steps}")

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.01
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr_ratio=0.1
    )

    # 5. Training Loop
    print("\n[5/5] Starting Fine-Tuning Loop...")
    step = 0
    epoch = 0
    best_score = init_score
    best_step = 0
    accum_loss = 0.0
    accum_steps = 0
    training_history = []
    t_start = time.time()

    optimizer.zero_grad()

    while step < total_steps:
        epoch += 1
        print(f"\n--- Epoch {epoch} / {args.epochs} ---")
        for batch in train_loader:
            if batch is None:
                continue

            input_values = batch["input_values"].to("cuda")
            targets = batch["targets"].to("cuda")
            target_lengths = batch["target_lengths"].to("cuda")

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_values)
                logits = outputs.logits
                log_probs = torch.log_softmax(logits, dim=-1).permute(1, 0, 2)

                if batch.get("attention_mask") is not None:
                    attention_mask = batch["attention_mask"].to("cuda")
                    input_lengths = model._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
                else:
                    input_lengths = torch.full((input_values.shape[0],), logits.shape[1], dtype=torch.long, device="cuda")

                loss = F.ctc_loss(
                    log_probs,
                    targets,
                    input_lengths,
                    target_lengths,
                    blank=0,
                    reduction="mean",
                    zero_infinity=True
                )
                scaled_loss = loss / args.grad_accum

            scaled_loss.backward()
            accum_loss += loss.item()
            accum_steps += 1

            if accum_steps % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                # Step logging
                if step % 25 == 0 or step <= 5:
                    avg_step_loss = accum_loss / accum_steps
                    curr_lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t_start
                    steps_per_sec = step / max(1e-5, elapsed)
                    eta_mins = (total_steps - step) / max(1e-5, steps_per_sec) / 60.0
                    print(f"Step {step}/{total_steps} | Loss: {avg_step_loss:.4f} | LR: {curr_lr:.2e} | Speed: {steps_per_sec:.2f} step/s | ETA: {eta_mins:.1f}m")
                    accum_loss = 0.0
                    accum_steps = 0

                # Validation Evaluation
                if step % args.eval_every == 0 or step == total_steps:
                    b_loss, b_wer, _ = evaluate_fast_val(model, processor, dev_b_records)
                    a_loss, a_wer, _ = evaluate_fast_val(model, processor, dev_a_sample)
                    comb_score = 0.5 * b_wer + 0.5 * a_wer
                    is_best = comb_score < best_score

                    print(f"\n>>> [EVAL @ STEP {step}] Dev B: Loss={b_loss:.4f}, WER={b_wer:.2f}% | Dev A Sample: Loss={a_loss:.4f}, WER={a_wer:.2f}% | Combined Score={comb_score:.2f} (Best={best_score:.2f}) {'*** NEW BEST ***' if is_best else ''}")

                    training_history.append({
                        "step": step,
                        "epoch": epoch,
                        "train_loss": avg_step_loss if 'avg_step_loss' in locals() else 0.0,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "dev_b_wer": b_wer,
                        "dev_b_loss": b_loss,
                        "dev_a_wer": a_wer,
                        "dev_a_loss": a_loss,
                        "combined_score": comb_score,
                        "is_best": is_best
                    })

                    if is_best:
                        best_score = comb_score
                        best_step = step
                        print(f"[Checkpoint] Saving new best model to {BEST_MODEL_DIR}...")
                        model.save_pretrained(BEST_MODEL_DIR)
                        processor.save_pretrained(BEST_MODEL_DIR)
                        with open(BEST_MODEL_DIR / "best_metrics.json", "w", encoding="utf-8") as f:
                            json.dump({
                                "best_step": best_step,
                                "best_combined_score": best_score,
                                "dev_b_wer": b_wer,
                                "dev_a_wer": a_wer,
                                "dev_b_loss": b_loss,
                                "dev_a_loss": a_loss,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
                            }, f, indent=2)

                # Periodic Checkpoint Saving
                if step % 500 == 0 or step == total_steps:
                    ckpt_dir = CHECKPOINTS_DIR / f"checkpoint-{step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    print(f"[Checkpoint] Saving checkpoint to {ckpt_dir}...")
                    model.save_pretrained(ckpt_dir)
                    processor.save_pretrained(ckpt_dir)

            if step >= total_steps:
                break

    # Save full training history
    with open(EXP_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=2)

    total_training_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE in {total_training_time/60:.1f} minutes!")
    print(f"Best Checkpoint: Step {best_step} with Combined Score: {best_score:.2f}")
    print("=" * 80)

    # 6. Comprehensive Final Comparative Evaluation
    print("\n" + "=" * 80)
    print("RUNNING FINAL BENCHMARK EVALUATION (BASELINE E22-01 vs NEW BEST MODEL)")
    print("=" * 80)

    print("\n--- Evaluating Baseline E22-01 ---")
    baseline_results = full_benchmark_eval(INIT_CHECKPOINT, kenlm_decoder, max_dev_a_greedy=1000, max_dev_a_kenlm=150)

    print(f"\n--- Evaluating New Best Model (from {BEST_MODEL_DIR}) ---")
    best_results = full_benchmark_eval(BEST_MODEL_DIR, kenlm_decoder, max_dev_a_greedy=1000, max_dev_a_kenlm=150)

    # Save comparison report
    final_report = {
        "baseline_e22_01": baseline_results,
        "new_best_model": best_results,
        "best_step": best_step,
        "best_combined_score": best_score,
        "training_time_minutes": round(total_training_time / 60.0, 2),
        "total_steps": step
    }
    with open(EXP_DIR / "final_comparison_report.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY COMPARISON TABLE")
    print("=" * 80)
    for split in ["dev_b_waaz_val", "dev_a_legacy_val"]:
        base_g = baseline_results[split]["greedy_wer"]
        new_g = best_results[split]["greedy_wer"]
        delta_g = new_g - base_g
        base_k = baseline_results[split]["kenlm_wer"]
        new_k = best_results[split]["kenlm_wer"]
        delta_k = new_k - base_k
        print(f"Split: {split}")
        print(f"  Greedy WER: {base_g:.2f}% -> {new_g:.2f}% ({delta_g:+.2f} pp)")
        print(f"  KenLM  WER: {base_k:.2f}% -> {new_k:.2f}% ({delta_k:+.2f} pp)")
        print(f"  CER:        {baseline_results[split]['greedy_cer']:.2f}% -> {best_results[split]['greedy_cer']:.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()
