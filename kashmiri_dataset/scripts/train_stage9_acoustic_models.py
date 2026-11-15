#!/usr/bin/env python3
"""
Stage 9: Controlled Acoustic Training Engine (B-02A, B-02B, B-02C)
================================================================
Robust, single-process, length-bucketed training with BF16 AMP:
- Zero multiprocessing deadlock risks
- Matched optimization step budget
- Independent base XLS-R 300M initialization
- Exact unpadded/masked validation decoding on 6,728 development samples
"""

import os
import sys
import json
import time
import math
import random
import argparse
from pathlib import Path

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

WORKSPACE_DIR = Path("/home/musa/Kashmiri Project")
DATASET_DIR = WORKSPACE_DIR / "kashmiri_dataset"
EXP_ROOT = DATASET_DIR / "experiments/native_asr_training/experiments"
STAGE9_DIR = EXP_ROOT / "stage9_b02_acoustic_comparison"
BASE_XLSR_DIR = EXP_ROOT / "B-01-xlsr300m-character-ctc/checkpoints/base_xlsr"
TOK_DIR = EXP_ROOT / "B-01-xlsr300m-character-ctc/tokenizer"
VAL_MANIFEST = DATASET_DIR / "experiments/native_asr_training/data_splits/val.jsonl"

sys.path.insert(0, str(DATASET_DIR / "scripts"))
from normalize_transcripts import normalize_kashmiri_text

def levenshtein_distance(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y), dtype=int)
    for x in range(size_x): matrix[x, 0] = x
    for y in range(size_y): matrix[0, y] = y
    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix[x, y] = matrix[x-1, y-1]
            else:
                matrix[x, y] = min(matrix[x-1, y] + 1, matrix[x-1, y-1] + 1, matrix[x, y-1] + 1)
    return matrix[size_x - 1, size_y - 1]

def calculate_wer_and_cer(references, hypotheses):
    total_words, word_errors = 0, 0
    total_chars, char_errors = 0, 0
    for ref, hyp in zip(references, hypotheses):
        ref_words = ref.strip().split()
        hyp_words = hyp.strip().split()
        word_errors += levenshtein_distance(ref_words, hyp_words)
        total_words += len(ref_words)

        ref_chars = list(ref.replace(" ", ""))
        hyp_chars = list(hyp.replace(" ", ""))
        char_errors += levenshtein_distance(ref_chars, hyp_chars)
        total_chars += len(ref_chars)

    wer = (word_errors / max(1, total_words))
    cer = (char_errors / max(1, total_chars))
    return wer, cer, word_errors, total_words, char_errors, total_chars

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
                txt = d.get("normalized_text", d.get("text", "")).strip()
                if not txt:
                    continue
                t_frames = int(dur * 50)
                if len(txt) > t_frames:
                    filtered_ctc += 1
                    continue
                self.samples.append({
                    "sample_id": d.get("sample_id", d.get("segment_id", "")),
                    "audio_path": d["audio_path"],
                    "duration": dur,
                    "text": d.get("text", ""),
                    "normalized_text": txt
                })

        self.samples.sort(key=lambda x: x["duration"])
        self.batch_size = batch_size
        self.batches = [self.samples[i:i+batch_size] for i in range(0, len(self.samples), batch_size)]
        print(f"Loaded {len(self.samples):,} valid samples into {len(self.batches):,} batches (batch_size={batch_size}) from {Path(manifest_file).name}")
        if filtered_ctc > 0:
            print(f"  Safely filtered {filtered_ctc} CTC length violation samples.")

    def get_shuffled_batches(self):
        shuffled = list(self.batches)
        random.shuffle(shuffled)
        return shuffled

def collate_batch(batch_items, processor):
    wavs = []
    norm_texts = []
    sample_ids = []
    durations = []
    
    for item in batch_items:
        audio_p = WORKSPACE_DIR / item["audio_path"]
        try:
            audio, sr = sf.read(str(audio_p))
            if sr != 16000:
                import torchaudio
                t_audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
                audio = torchaudio.transforms.Resample(sr, 16000)(t_audio).squeeze(0).numpy()
        except Exception as e:
            continue
        
        wavs.append(audio)
        norm_texts.append(item["normalized_text"])
        sample_ids.append(item["sample_id"])
        durations.append(len(audio) / 16000.0)
        
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
        "norm_texts": norm_texts
    }

def evaluate_validation(model, processor, val_dataset, device):
    model.eval()
    all_refs = []
    all_hyps = []
    all_sample_ids = []
    val_loss_total = 0.0
    val_batches_count = 0
    blank_count = 0
    total_emitted_tokens = 0
    empty_hyps = 0

    with torch.no_grad():
        for batch in val_dataset.batches:
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

                pred_str = processor.decode(p_ids)
                ref_norm = col["norm_texts"][i]
                all_refs.append(ref_norm)
                all_hyps.append(pred_str)
                all_sample_ids.append(col["sample_ids"][i])
                if not pred_str.strip():
                    empty_hyps += 1

    mean_val_loss = val_loss_total / max(1, val_batches_count)
    wer, cer, w_err, total_w, c_err, total_c = calculate_wer_and_cer(all_refs, all_hyps)
    blank_rate = blank_count / max(1, total_emitted_tokens)
    empty_pct = (empty_hyps / max(1, len(all_refs))) * 100.0

    return {
        "val_loss": mean_val_loss,
        "norm_wer": wer,
        "norm_cer": cer,
        "blank_rate": blank_rate,
        "empty_pct": empty_pct,
        "word_errors": w_err,
        "total_words": total_w,
        "char_errors": c_err,
        "total_chars": total_c,
        "sample_ids": all_sample_ids,
        "refs": all_refs,
        "hyps": all_hyps
    }

def run_experiment(exp_name, manifest_name, max_steps=1000, eval_interval=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = STAGE9_DIR / exp_name
    best_model_dir = exp_dir / "checkpoints/best_model"
    latest_ckpt_dir = exp_dir / "checkpoints/latest_checkpoint"
    log_dir = exp_dir / "logs"
    pred_dir = exp_dir / "predictions"
    report_dir = exp_dir / "reports"

    for d in [best_model_dir, latest_ckpt_dir, log_dir, pred_dir, report_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"STARTING FULL TRAINING EXPERIMENT: {exp_name}")
    print(f"Target Manifest: {manifest_name}")
    print(f"Matched Optimization Budget: {max_steps} steps | Effective Batch Size: 32")
    print("=" * 70)

    processor = Wav2Vec2Processor.from_pretrained(str(TOK_DIR))
    manifest_p = STAGE9_DIR / "manifests" / manifest_name
    train_dataset = LengthBucketedDataset(manifest_p, max_duration=16.0, min_duration=0.5, batch_size=32)
    val_dataset = LengthBucketedDataset(VAL_MANIFEST, max_duration=16.0, min_duration=0.5, batch_size=32)

    print(f"Initializing fresh model from {BASE_XLSR_DIR} (ctc_zero_infinity=True)...")
    model = Wav2Vec2ForCTC.from_pretrained(str(BASE_XLSR_DIR), ctc_zero_infinity=True)
    model.freeze_feature_encoder()
    model.to(device)

    ENC_LR = 1.0e-4
    HEAD_LR = 1.0e-3
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = int(max_steps * 0.10)

    optimizer_grouped = [
        {"params": [p for n, p in model.named_parameters() if "lm_head" not in n and p.requires_grad], "lr": ENC_LR},
        {"params": [p for n, p in model.named_parameters() if "lm_head" in n and p.requires_grad], "lr": HEAD_LR}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped, weight_decay=WEIGHT_DECAY)

    def get_lr_factor(step):
        if step < WARMUP_STEPS:
            return float(step) / float(max(1, WARMUP_STEPS))
        progress = float(step - WARMUP_STEPS) / float(max(1, max_steps - WARMUP_STEPS))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_factor)

    train_log_file = log_dir / "training_log.csv"
    val_log_file = log_dir / "validation_metrics.csv"
    resource_file = log_dir / "resource_usage.csv"

    with open(train_log_file, "w", encoding="utf-8") as f:
        f.write("step,loss,grad_norm,enc_lr,head_lr,vram_gb,step_time_sec\n")
    with open(val_log_file, "w", encoding="utf-8") as f:
        f.write("step,val_loss,norm_wer,norm_cer,blank_rate,empty_pct,eval_time_sec\n")
    with open(resource_file, "w", encoding="utf-8") as f:
        f.write("step,elapsed_sec,throughput_samples_per_sec,vram_peak_gb\n")

    global_step = 0
    best_val_wer = float("inf")
    running_loss = 0.0
    start_time = time.time()
    step_start = time.time()

    model.train()
    shuffled_batches = train_dataset.get_shuffled_batches()
    batch_idx = 0

    print(f"\n--- Training Progress: 0 -> {max_steps} steps ---")
    while global_step < max_steps:
        if batch_idx >= len(shuffled_batches):
            shuffled_batches = train_dataset.get_shuffled_batches()
            batch_idx = 0

        batch = shuffled_batches[batch_idx]
        batch_idx += 1

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
            optimizer.zero_grad()
            continue

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        global_step += 1
        running_loss += loss.item()

        if global_step % 50 == 0:
            step_dur = (time.time() - step_start) / 50.0
            vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            lr_e = scheduler.get_last_lr()[0]
            lr_h = scheduler.get_last_lr()[1]
            elapsed = time.time() - start_time
            throughput = (50 * len(batch)) / (time.time() - step_start)
            print(f"Step {global_step:04d}/{max_steps} | Loss: {running_loss/50:.4f} | LR: {lr_e:.2e} | {throughput:.1f} samp/s | VRAM: {vram:.1f}GB | Elapsed: {elapsed/60:.1f}m")

            with open(train_log_file, "a", encoding="utf-8") as f:
                f.write(f"{global_step},{running_loss/50:.4f},{grad_norm.item():.4f},{lr_e:.6e},{lr_h:.6e},{vram:.2f},{step_dur:.3f}\n")
            with open(resource_file, "a", encoding="utf-8") as f:
                f.write(f"{global_step},{elapsed:.1f},{throughput:.2f},{vram:.2f}\n")

            running_loss = 0.0
            step_start = time.time()

        # Validation at eval_interval or final step
        if global_step % eval_interval == 0 or global_step == max_steps:
            val_start = time.time()
            print(f"\n--- Running Full Validation on 6,728 samples (Step {global_step}) ---")
            val_res = evaluate_validation(model, processor, val_dataset, device)
            val_dur = time.time() - val_start
            val_wer = val_res["norm_wer"]
            val_cer = val_res["norm_cer"]

            print(f"Step {global_step} Val | Loss: {val_res['val_loss']:.4f} | Norm WER: {val_wer*100:.2f}% | Norm CER: {val_cer*100:.2f}% | Blank Rate: {val_res['blank_rate']*100:.2f}% | Val Time: {val_dur:.1f}s")
            with open(val_log_file, "a", encoding="utf-8") as f:
                f.write(f"{global_step},{val_res['val_loss']:.4f},{val_wer:.4f},{val_cer:.4f},{val_res['blank_rate']:.4f},{val_res['empty_pct']:.2f},{val_dur:.1f}\n")

            if val_wer < best_val_wer:
                improvement = (best_val_wer - val_wer) * 100 if best_val_wer != float("inf") else 0.0
                print(f"  >>> BEST MODEL! WER decreased to {val_wer*100:.2f}% (-{improvement:.2f}%)")
                best_val_wer = val_wer
                model.save_pretrained(str(best_model_dir))
                processor.save_pretrained(str(best_model_dir))

                best_pred_file = pred_dir / "best_validation_predictions.jsonl"
                with open(best_pred_file, "w", encoding="utf-8") as f:
                    for sid, ref, hyp in zip(val_res["sample_ids"], val_res["refs"], val_res["hyps"]):
                        f.write(json.dumps({"sample_id": sid, "ref": ref, "hyp": hyp}, ensure_ascii=False) + "\n")

            model.train()
            step_start = time.time()

    # Save final latest checkpoint
    model.save_pretrained(str(latest_ckpt_dir))
    processor.save_pretrained(str(latest_ckpt_dir))
    torch.save({
        "global_step": global_step,
        "best_val_wer": best_val_wer,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict()
    }, str(latest_ckpt_dir / "training_state.pt"))

    total_training_time = time.time() - start_time
    print(f"\n=======================================================")
    print(f"EXPERIMENT {exp_name} COMPLETED IN {total_training_time/60:.1f} MINUTES!")
    print(f"Best Validation Normalized WER: {best_val_wer*100:.2f}%")
    print(f"=======================================================\n")
    return best_val_wer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True, choices=["B-02A", "B-02B", "B-02C", "all"])
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=500)
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
        run_experiment(exp, manifest_map[exp], max_steps=args.max_steps, eval_interval=args.eval_interval)
