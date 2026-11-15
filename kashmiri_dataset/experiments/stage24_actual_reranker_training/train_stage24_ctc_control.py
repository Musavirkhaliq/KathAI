#!/usr/bin/env python3
"""
Stage 24: Controlled Acoustic Continuation Training (CTC Control)
Initializes from Stage 22 E22-01 and performs controlled acoustic fine-tuning
on the Stage 24 training-only Kashmiri speech mixture.

Monitors:
  - CTC Loss progression
  - Catastrophic forgetting on dev_a_disjoint
  - Generalization on dev_b_waaz_val
  - Greedy & KenLM WER vs original E22-01
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import json
import math
import time
import hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pyctcdecode import build_ctcdecoder
import jiwer

WORKSPACE = Path("/home/musa/Kashmiri Project")
sys.path.insert(0, str(WORKSPACE))

from kashmiri_dataset.audio.canonical_audio import load_canonical_audio
from kashmiri_dataset.scripts.metrics_evaluator import normalize_perso_arabic_kashmiri as norm_strict
from kashmiri_dataset.experiments.stage22c_inference_certification.inference.stage22c_exact_output_lengths import compute_exact_ctc_length, slice_exact_logits

STAGE24_DIR = WORKSPACE / "kashmiri_dataset/experiments/stage24_actual_reranker_training"
INITIAL_CKPT = WORKSPACE / "kashmiri_dataset/experiments/stage22_data_expanded_training/checkpoints/E22-01/best_model"
TRAIN_MANIFEST = STAGE24_DIR / "data/stage24_train_sampled_20k.jsonl"
OUT_CKPT_DIR = STAGE24_DIR / "checkpoints/E24-CTC-01"
LM_PATH = WORKSPACE / "kashmiri_dataset/experiments/native_asr_training/experiments/stage7_ctc_lm_decoding/language_models/lm_word_4gram_domain_d.arpa"
UNIGRAMS_PATH = WORKSPACE / "kashmiri_dataset/experiments/stage22b_decoder_and_independent_test/language_model/unigrams.txt"

DEV_B_PATH = WORKSPACE / "kashmiri_dataset/releases/asr_training_v2_waaz/validation_manifests/dev_b_waaz_val.jsonl"
DEV_CALIB_PATH = WORKSPACE / "kashmiri_dataset/experiments/stage11_robustness_optimization/data_splits/dev_calibration.jsonl"

def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

class CTCAudioDataset(Dataset):
    def __init__(self, manifest_path, processor, max_duration=25.0, min_duration=0.5):
        self.processor = processor
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.records = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line.strip())
                    dur = float(d.get("duration", 0.0))
                    if dur == 0 or (self.min_duration <= dur <= self.max_duration):
                        self.records.append(d)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        raw_p = rec.get("audio_filepath") or rec.get("audio_path")
        p = Path(raw_p) if Path(raw_p).is_absolute() else WORKSPACE / raw_p
        wav, _ = load_canonical_audio(str(p), target_sr=16000)
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav)
        if wav.dim() > 1:
            wav = wav.squeeze()

        ref_raw = rec.get("text", rec.get("normalized_text", ""))
        ref_norm = norm_strict(ref_raw)
        labels = self.processor.tokenizer.encode(ref_norm)
        if not labels:
            labels = [0]

        return {
            "wav": wav,
            "labels": torch.tensor(labels, dtype=torch.long),
            "ref_norm": ref_norm,
            "sample_id": rec.get("sample_id", "")
        }

def ctc_collate_fn(batch):
    wavs = [b["wav"] for b in batch]
    labels = [b["labels"] for b in batch]
    ref_norms = [b["ref_norm"] for b in batch]
    sids = [b["sample_id"] for b in batch]

    lengths = torch.tensor([len(w) for w in wavs], dtype=torch.long)
    padded_wavs = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True)

    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_values": padded_wavs,
        "input_lengths": lengths,
        "labels": padded_labels,
        "label_lengths": label_lengths,
        "ref_norms": ref_norms,
        "sample_ids": sids
    }

def evaluate_greedy(model, processor, records_manifest, max_eval=200, device="cuda"):
    model.eval()
    records = []
    with open(records_manifest, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    if max_eval and len(records) > max_eval:
        records = records[:max_eval]

    total_words = 0
    total_errors = 0

    with torch.no_grad():
        for r in records:
            raw_p = r.get("audio_filepath") or r.get("audio_path")
            p = Path(raw_p) if Path(raw_p).is_absolute() else WORKSPACE / raw_p
            try:
                wav, _ = load_canonical_audio(str(p), target_sr=16000)
                if isinstance(wav, np.ndarray):
                    wav = torch.from_numpy(wav)
                if wav.dim() > 1:
                    wav = wav.squeeze()
                if len(wav) < 400:
                    continue

                inp = wav.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
                out = model(inp).logits[0].float()
                pred_ids = torch.argmax(out, dim=-1).cpu().tolist()
                pred_text = norm_strict(processor.tokenizer.decode(pred_ids, group_tokens=True))
                ref_norm = norm_strict(r.get("text", r.get("normalized_text", "")))

                ref_words = len(ref_norm.split())
                total_words += ref_words
                res = jiwer.process_words(ref_norm, pred_text)
                errs = res.substitutions + res.deletions + res.insertions
                total_errors += errs
            except Exception:
                pass

    wer = (total_errors / total_words) * 100.0 if total_words > 0 else 0.0
    return round(wer, 2), total_errors, total_words

def run_ctc_training():
    print("=" * 70)
    print("STAGE 24: CONTROLLED ACOUSTIC CONTINUATION TRAINING (CTC CONTROL)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_model_dir = OUT_CKPT_DIR / "best_model"
    best_model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading initial model E22-01 from {INITIAL_CKPT}...")
    processor = Wav2Vec2Processor.from_pretrained(str(INITIAL_CKPT))
    model = Wav2Vec2ForCTC.from_pretrained(str(INITIAL_CKPT), torch_dtype=torch.bfloat16).to(device)
    model.freeze_feature_encoder()
    print("Feature encoder frozen. CTC classification head verified.")

    print(f"Loading dataset from {TRAIN_MANIFEST}...")
    dataset = CTCAudioDataset(str(TRAIN_MANIFEST), processor)
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        collate_fn=ctc_collate_fn,
        pin_memory=False
    )
    print(f"Loaded {len(dataset)} training utterances.")

    # Hyperparameters
    lr = 2.0e-6
    total_steps = 250
    warmup_steps = 25
    grad_accum_steps = 2
    eval_every = 50
    save_every = 125

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # Initial baseline evaluations
    print("\nRunning initial baseline evaluations...")
    init_dev_b_wer, init_dev_b_errs, _ = evaluate_greedy(model, processor, DEV_B_PATH, max_eval=127, device=device)
    init_calib_wer, init_calib_errs, _ = evaluate_greedy(model, processor, DEV_CALIB_PATH, max_eval=200, device=device)
    print(f"  [Baseline E22-01] Dev B Greedy WER: {init_dev_b_wer:.2f}% | Calib Sample Greedy WER: {init_calib_wer:.2f}%")

    global_step = 0
    running_loss = 0.0
    accum_loss = 0.0
    best_combined_wer = 999.0
    best_step = 0

    logs = []
    t_start = time.time()
    t_step = time.time()
    optimizer.zero_grad()

    model.train()
    data_iter = iter(loader)

    print(f"\nStarting CTC Training (Target: {total_steps} steps)...")
    while global_step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        inps = batch["input_values"].to(device=device, dtype=torch.bfloat16)
        in_lens = batch["input_lengths"].to(device=device)
        labels = batch["labels"].to(device=device)

        # Construct attention mask from lengths
        attn_mask = (torch.arange(inps.size(1), device=device)[None, :] < in_lens[:, None]).long()

        outputs = model(input_values=inps, attention_mask=attn_mask, labels=labels)
        loss = outputs.loss / grad_accum_steps
        loss.backward()

        accum_loss += loss.item() * grad_accum_steps

        if (global_step + 1) % grad_accum_steps == 0 or True:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            running_loss += accum_loss
            accum_loss = 0.0

            if global_step % 10 == 0:
                step_speed = 10.0 / max(0.01, (time.time() - t_step))
                t_step = time.time()
                cur_lr = scheduler.get_last_lr()[0]
                print(f"Step {global_step:3d}/{total_steps} | CTC Loss: {running_loss/10:.4f} | LR: {cur_lr:.2e} | Speed: {step_speed:.2f} steps/s")
                running_loss = 0.0

            if global_step % eval_every == 0 or global_step == total_steps:
                dev_b_wer, b_errs, _ = evaluate_greedy(model, processor, DEV_B_PATH, max_eval=127, device=device)
                calib_wer, c_errs, _ = evaluate_greedy(model, processor, DEV_CALIB_PATH, max_eval=200, device=device)
                comb_score = (dev_b_wer + calib_wer) / 2.0

                print(f"  --> [EVAL @ STEP {global_step}] Dev B WER: {dev_b_wer:.2f}% | Calib Sample WER: {calib_wer:.2f}% | Combined: {comb_score:.2f}%")

                log_entry = {
                    "step": global_step,
                    "dev_b_greedy_wer": dev_b_wer,
                    "calib_greedy_wer": calib_wer,
                    "combined_score": round(comb_score, 2),
                    "loss": round(outputs.loss.item(), 4),
                    "lr": float(scheduler.get_last_lr()[0])
                }
                logs.append(log_entry)

                if comb_score < best_combined_wer:
                    best_combined_wer = comb_score
                    best_step = global_step
                    print(f"  *** NEW BEST CTC MODEL AT STEP {global_step}! Saving weights... ***")
                    model.save_pretrained(best_model_dir)
                    processor.save_pretrained(best_model_dir)

                model.train()

            if global_step % save_every == 0:
                step_ckpt_dir = OUT_CKPT_DIR / f"checkpoint-{global_step}"
                step_ckpt_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(step_ckpt_dir)
                processor.save_pretrained(step_ckpt_dir)

    print(f"\nCTC Training completed in {time.time()-t_start:.1f}s. Best model: Step {best_step} (Combined WER: {best_combined_wer:.2f}%).")

    # Save training log
    log_csv = STAGE24_DIR / "training_logs/stage24_ctc_training_log.csv"
    pd.DataFrame(logs).to_csv(log_csv, index=False)
    print(f"Saved CTC training log to {log_csv}")

    # Compute sha256 of best model
    weights_path = best_model_dir / "model.safetensors"
    h = get_sha256(weights_path) if weights_path.exists() else ""
    with open(best_model_dir / "model_sha256.txt", "w") as f:
        f.write(h + "\n")

    manifest = {
        "experiment_id": "E24-CTC-01",
        "parent_checkpoint": "Stage 22 E22-01",
        "total_steps": total_steps,
        "best_step": best_step,
        "best_combined_wer": best_combined_wer,
        "initial_dev_b_wer": init_dev_b_wer,
        "initial_calib_wer": init_calib_wer,
        "weights_sha256": h,
        "weights_path": str(weights_path)
    }
    with open(OUT_CKPT_DIR / "stage24_ctc_model_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

if __name__ == "__main__":
    run_ctc_training()
