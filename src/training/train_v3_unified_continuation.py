#!/usr/bin/env python3
"""
V3 Unified All-Data ASR Acoustic Training Continuation
Trains the best checkpoint from previous run across the complete canonical V3.0
clean dataset (101,173 utterances / 199.93 hours) for 2 full epochs.

Initialization:
  - Checkpoint: kashmiri_dataset/experiments/all_data_expanded_training/checkpoints/best_model
Dataset:
  - Manifest: kashmiri_dataset/releases/asr_v3_unified_splits/train_unified_v3.jsonl
Validation:
  - WAAZ Community Val: kashmiri_dataset/releases/asr_v3_unified_splits/val_community_waaz_v3.jsonl (309 utts)
  - Standard Dev Val: kashmiri_dataset/releases/asr_v3_unified_splits/val_standard_v3.jsonl (1,000 utts)
Testing:
  - WAAZ Community Test: kashmiri_dataset/releases/asr_v3_unified_splits/test_community_waaz_v3.jsonl (114 utts)
  - Standard Test Suite: kashmiri_dataset/releases/asr_v3_unified_splits/test_standard_v3.jsonl (1,500 utts)
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import json
import math
import time
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

INIT_CHECKPOINT = WORKSPACE / "kashmiri_dataset/experiments/all_data_expanded_training/checkpoints/best_model"
TRAIN_MANIFEST = WORKSPACE / "kashmiri_dataset/releases/asr_v3_unified_splits/train_unified_v3.jsonl"
EXP_DIR = WORKSPACE / "kashmiri_dataset/experiments/v3_unified_all_data_training"
CHECKPOINTS_DIR = EXP_DIR / "checkpoints"
BEST_MODEL_DIR = CHECKPOINTS_DIR / "best_model"

VAL_WAAZ_PATH = WORKSPACE / "kashmiri_dataset/releases/asr_v3_unified_splits/val_community_waaz_v3.jsonl"
VAL_STD_PATH = WORKSPACE / "kashmiri_dataset/releases/asr_v3_unified_splits/val_standard_v3.jsonl"
TEST_WAAZ_PATH = WORKSPACE / "kashmiri_dataset/releases/asr_v3_unified_splits/test_community_waaz_v3.jsonl"
TEST_STD_PATH = WORKSPACE / "kashmiri_dataset/releases/asr_v3_unified_splits/test_standard_v3.jsonl"
LM_PATH = WORKSPACE / "kashmiri_dataset/experiments/native_asr_training/experiments/stage7_ctc_lm_decoding/language_models/lm_word_4gram_domain_d.arpa"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.05):
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

def full_benchmark_eval(model_dir, kenlm_decoder=None, max_std_greedy=1500, max_std_kenlm=150):
    print(f"\n[Benchmark Eval] Loading model from {model_dir} for evaluation...")
    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(model_dir).to("cuda")
    model.eval()

    splits = {
        "val_community_waaz_v3": (VAL_WAAZ_PATH, 309, 100),
        "val_standard_v3": (VAL_STD_PATH, max_std_greedy, max_std_kenlm),
        "test_community_waaz_v3": (TEST_WAAZ_PATH, 114, 114),
        "test_standard_v3": (TEST_STD_PATH, max_std_greedy, max_std_kenlm)
    }

    results = {}
    with torch.no_grad():
        for split_name, (split_path, max_items_greedy, max_items_kenlm) in splits.items():
            if not split_path.exists():
                continue

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
            print(f"  [{split_name:24s}] utts={len(all_refs_g):4d} | Greedy WER={g_wer:5.2f}% | KenLM WER={k_wer:5.2f}% | CER={cer:5.2f}%")

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to train (default: 2)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per device")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2.0e-6, help="Learning rate (default: 2e-6)")
    parser.add_argument("--eval_every", type=int, default=250, help="Evaluation step frequency")
    parser.add_argument("--max_steps", type=int, default=None, help="Override total steps")
    args = parser.parse_args()

    set_seed(42)
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("V3 UNIFIED ALL-DATA ACOUSTIC TRAINING CONTINUATION")
    print(f"Initialization: {INIT_CHECKPOINT}")
    print(f"Training Manifest: {TRAIN_MANIFEST}")
    print(f"Checkpoints Root: {CHECKPOINTS_DIR}")
    print(f"Epochs: {args.epochs} | LR: {args.lr} | Effective Batch Size: {args.batch_size * args.grad_accum}")
    print("=" * 80)

    # 1. Load Model & Processor
    print("\n[1/5] Loading Processor & Best Model from Previous Run...")
    processor = Wav2Vec2Processor.from_pretrained(INIT_CHECKPOINT)
    model = Wav2Vec2ForCTC.from_pretrained(INIT_CHECKPOINT).to("cuda")
    model.train()
    model.freeze_feature_encoder()
    assert model.config.vocab_size == 161, f"Vocab mismatch: {model.config.vocab_size} != 161"
    print(f"[Model] Feature encoder frozen. Vocab size: {model.config.vocab_size}. Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # Initialize BEST_MODEL_DIR with starting weights
    if not (BEST_MODEL_DIR / "model.safetensors").exists():
        print(f"[Init] Initializing {BEST_MODEL_DIR} with baseline model weights...")
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
    val_waaz_records = []
    with open(VAL_WAAZ_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                val_waaz_records.append(json.loads(line.strip()))

    val_std_records = []
    with open(VAL_STD_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                val_std_records.append(json.loads(line.strip()))
    random.Random(42).shuffle(val_std_records)
    val_std_sample = val_std_records[:250]
    print(f"[Validation] WAAZ Community Val size: {len(val_waaz_records)} | Standard Val monitoring sample: {len(val_std_sample)}")

    # Pre-train baseline evaluation
    print("\n[Pre-Train Check] Evaluating starting model on V3 validation sets...")
    init_waaz_loss, init_waaz_wer, _ = evaluate_fast_val(model, processor, val_waaz_records)
    init_std_loss, init_std_wer, _ = evaluate_fast_val(model, processor, val_std_sample)
    init_score = 0.5 * init_waaz_wer + 0.5 * init_std_wer
    print(f"[Baseline Model] WAAZ Val WER: {init_waaz_wer:.2f}% (loss: {init_waaz_loss:.4f}) | Standard Val Sample WER: {init_std_wer:.2f}% (loss: {init_std_loss:.4f}) | Combined: {init_score:.2f}")

    # 4. Build Dataset & DataLoader
    print("\n[4/5] Building Training DataLoader on Canonical V3 Unified Dataset...")
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
    warmup_steps = min(150, int(0.03 * total_steps))

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
        min_lr_ratio=0.05
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
                    w_loss, w_wer, _ = evaluate_fast_val(model, processor, val_waaz_records)
                    s_loss, s_wer, _ = evaluate_fast_val(model, processor, val_std_sample)
                    comb_score = 0.5 * w_wer + 0.5 * s_wer
                    is_best = comb_score < best_score

                    print(f"\n>>> [EVAL @ STEP {step}] WAAZ Val: Loss={w_loss:.4f}, WER={w_wer:.2f}% | Standard Val Sample: Loss={s_loss:.4f}, WER={s_wer:.2f}% | Combined={comb_score:.2f} (Best={best_score:.2f}) {'*** NEW BEST ***' if is_best else ''}")

                    training_history.append({
                        "step": step,
                        "epoch": epoch,
                        "train_loss": avg_step_loss if 'avg_step_loss' in locals() else 0.0,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "waaz_wer": w_wer,
                        "waaz_loss": w_loss,
                        "std_wer": s_wer,
                        "std_loss": s_loss,
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
                                "waaz_wer": w_wer,
                                "std_wer": s_wer,
                                "waaz_loss": w_loss,
                                "std_loss": s_loss,
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

    # 6. Comprehensive Final Benchmark Evaluation on V3 Test Suites
    print("\n" + "=" * 80)
    print("RUNNING FINAL BENCHMARK EVALUATION (BASELINE vs V3 BEST MODEL)")
    print("=" * 80)

    print("\n--- Evaluating Baseline Model ---")
    baseline_results = full_benchmark_eval(INIT_CHECKPOINT, kenlm_decoder)

    print(f"\n--- Evaluating New V3 Best Model (from {BEST_MODEL_DIR}) ---")
    best_results = full_benchmark_eval(BEST_MODEL_DIR, kenlm_decoder)

    final_report = {
        "baseline_model": baseline_results,
        "v3_best_model": best_results,
        "best_step": best_step,
        "best_combined_score": best_score,
        "training_time_minutes": round(total_training_time / 60.0, 2),
        "total_steps": step
    }
    with open(EXP_DIR / "final_v3_comparison_report.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print("\n" + "=" * 80)
    print("V3 FINAL SUMMARY COMPARISON TABLE")
    print("=" * 80)
    for split_key in baseline_results:
        base_g = baseline_results[split_key]["greedy_wer"]
        new_g = best_results[split_key]["greedy_wer"]
        delta_g = new_g - base_g
        base_k = baseline_results[split_key]["kenlm_wer"]
        new_k = best_results[split_key]["kenlm_wer"]
        delta_k = new_k - base_k
        print(f"Split: {split_key}")
        print(f"  Greedy WER: {base_g:.2f}% -> {new_g:.2f}% ({delta_g:+.2f} pp)")
        print(f"  KenLM  WER: {base_k:.2f}% -> {new_k:.2f}% ({delta_k:+.2f} pp)")
        print(f"  CER:        {baseline_results[split_key]['greedy_cer']:.2f}% -> {best_results[split_key]['greedy_cer']:.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()
