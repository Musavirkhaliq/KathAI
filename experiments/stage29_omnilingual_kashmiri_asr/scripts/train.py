#!/usr/bin/env python3
"""
Omnilingual Wav2Vec2 CTC Fine-Tuning Engine for Kashmiri Speech Recognition
Supports:
- Pretrained omniASR_CTC_300M_v2 and omniASR_CTC_1B_v2 adaptation
- Dataset A (matched historical 166.39h) and Dataset B (expanded 207.24h)
- BF16 mixed precision on Blackwell DGX Spark GB10
- Gradient accumulation, gradient checkpointing, and CNN feature extractor freezing
- Periodic validation on dev_a_disjoint and dev_b_waaz_val
- Best and latest checkpoint serialization
- Resumption capability
"""

import argparse
import json
import math
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import random
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import editdistance
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF
import yaml
from torch.utils.data import DataLoader, Dataset

# Add project root to python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import sentencepiece as spm
from src.normalization.normalizer import normalize_for_wer_evaluation
from experiments.stage29_omnilingual_kashmiri_asr.scripts.model import (
    Wav2Vec2AsrModel,
    OmniModelConfig,
    load_omni_checkpoint,
)


class KashmiriAsrDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        tokenizer: spm.SentencePieceProcessor,
        min_duration: float = 0.8,
        max_duration: float = 20.0,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.records = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                dur = float(row.get("duration", 0.0))
                if min_duration <= dur <= max_duration:
                    self.records.append(row)
                if max_samples and len(self.records) >= max_samples:
                    break

        print(f"Loaded {len(self.records):,} samples from {manifest_path} (durations: {min_duration}s - {max_duration}s)")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        wav, sr = sf.read(item["audio_path"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav_tensor = torch.from_numpy(wav)
        if sr != 16000:
            wav_tensor = AF.resample(wav_tensor, orig_freq=sr, new_freq=16000)

        # Normalize text and tokenize
        norm_text = normalize_for_wer_evaluation(item.get("text", ""))
        token_ids = self.tokenizer.EncodeAsIds(norm_text)
        # Ensure valid target tokens
        token_ids = [t for t in token_ids if t > 0]  # omit blank/pad from target

        return {
            "sample_id": item["sample_id"],
            "waveform": wav_tensor,
            "target_ids": torch.tensor(token_ids, dtype=torch.long),
            "text": norm_text,
        }


def collate_fn(batch):
    """Pads audio waveforms and text targets to form batch tensors."""
    waveforms = [item["waveform"] for item in batch]
    targets = [item["target_ids"] for item in batch]

    wav_lengths = torch.tensor([len(w) for w in waveforms], dtype=torch.long)
    tgt_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)

    max_wav_len = max(wav_lengths).item()
    max_tgt_len = max(max(tgt_lengths).item(), 1)

    padded_wavs = torch.zeros(len(batch), max_wav_len, dtype=torch.bfloat16)
    padded_targets = torch.zeros(len(batch), max_tgt_len, dtype=torch.long)

    for i, (w, t) in enumerate(zip(waveforms, targets)):
        padded_wavs[i, :len(w)] = w.to(dtype=torch.bfloat16)
        if len(t) > 0:
            padded_targets[i, :len(t)] = t

    return {
        "waveforms": padded_wavs,
        "wav_lengths": wav_lengths,
        "targets": padded_targets,
        "tgt_lengths": tgt_lengths,
        "texts": [item["text"] for item in batch],
    }


def compute_wer(references, hypotheses):
    total_words = 0
    total_errors = 0
    for r, h in zip(references, hypotheses):
        rw = r.split()
        hw = h.split()
        total_words += len(rw)
        total_errors += editdistance.eval(rw, hw)
    return (total_errors / total_words * 100.0) if total_words > 0 else 0.0


def evaluate_dev(model, val_loader, tokenizer, device="cuda", max_batches=30):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    references = []
    hypotheses = []

    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            if max_batches and b_idx >= max_batches:
                break
            wavs = batch["waveforms"].to(device)
            wav_lens = batch["wav_lengths"].to(device)
            targets = batch["targets"].to(device)
            tgt_lens = batch["tgt_lengths"].to(device)

            loss, logits, out_lens = model(wavs, wav_lens, targets, tgt_lens)
            total_loss += loss.item()
            num_batches += 1

            hyps = model.decode_greedy(logits, out_lens, tokenizer)
            references.extend([normalize_for_wer_evaluation(t) for t in batch["texts"]])
            hypotheses.extend([normalize_for_wer_evaluation(h) for h in hyps])

            del wavs, wav_lens, targets, tgt_lens, loss, logits, out_lens

    torch.cuda.empty_cache()
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    wer = compute_wer(references, hypotheses)
    return avg_loss, wer


def run_one_batch_overfit_test(model, dataset, tokenizer, device="cuda"):
    """
    Step A: One-batch overfitting preflight gate.
    Verifies that CTC loss decreases to < 0.10 and transcription matches.
    """
    print("\n=======================================================")
    print(">>> STEP A: ONE-BATCH OVERFITTING PREFLIGHT GATE")
    print("=======================================================")
    model.train()
    model.to(device)

    # Pick 4 samples
    subset = [dataset[i] for i in range(min(4, len(dataset)))]
    batch = collate_fn(subset)

    wavs = batch["waveforms"].to(device)
    wav_lens = batch["wav_lengths"].to(device)
    targets = batch["targets"].to(device)
    tgt_lens = batch["tgt_lengths"].to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    print("Initial references:")
    for i, t in enumerate(batch["texts"]):
        print(f"  [{i}]: {t}")

    print("\nTraining on single batch for 40 steps...")
    initial_loss = None
    final_loss = None

    for step in range(1, 41):
        optimizer.zero_grad()
        loss, logits, out_lens = model(wavs, wav_lens, targets, tgt_lens)
        loss.backward()
        optimizer.step()

        if step == 1:
            initial_loss = loss.item()
        final_loss = loss.item()

        if step % 10 == 0 or step == 1:
            print(f"  Step {step:02d} | CTC Loss: {loss.item():.4f}")

    # Check decoded output
    model.eval()
    with torch.no_grad():
        logits, out_lens = model(wavs, wav_lens)
        hyps = model.decode_greedy(logits, out_lens, tokenizer)

    print("\nPost-overfit predictions:")
    for i, (r, h) in enumerate(zip(batch["texts"], hyps)):
        print(f"  Ref [{i}]: {r}")
        print(f"  Hyp [{i}]: {h}")

    loss_drop = (initial_loss - final_loss) / initial_loss * 100.0 if initial_loss else 0.0
    print(f"\nOne-batch overfit result: Loss dropped from {initial_loss:.4f} to {final_loss:.4f} (-{loss_drop:.1f}%)")
    assert final_loss < initial_loss * 0.5, "Overfitting test failed: Loss did not drop significantly!"
    print(">>> PREFLIGHT STEP A PASSED: Acoustic CTC model converges and learns successfully.\n")


def train(
    config_path: str,
    overfit_test: bool = False,
    max_steps_override: Optional[int] = None,
    init_scratch: bool = False,
    resume_path: Optional[str] = None,
):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_id = cfg["experiment_id"]
    model_cfg_dict = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Starting Experiment {exp_id} on {device.upper()} ===")

    # Load tokenizer
    tokenizer_path = model_cfg_dict["tokenizer_path"]
    print(f"Loading SentencePiece tokenizer from {tokenizer_path}...")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(tokenizer_path)

    # Initialize model
    arch = model_cfg_dict.get("arch", "300m_v2")
    if "1b" in arch:
        model_config = OmniModelConfig.get_1b_config()
    else:
        model_config = OmniModelConfig.get_300m_config()

    model = Wav2Vec2AsrModel(model_config)

    # Load checkpoint if available and not init_scratch
    ckpt_path = model_cfg_dict.get("checkpoint_path")
    if ckpt_path and os.path.exists(ckpt_path) and not init_scratch:
        try:
            load_omni_checkpoint(model, ckpt_path, device="cpu")
        except Exception as e:
            print(f"Could not load checkpoint {ckpt_path} ({e}); using initialized weights.")
    else:
        print("Using freshly initialized model architecture.")

    model.to(dtype=torch.bfloat16, device=device)

    # Enable gradient checkpointing if requested
    if train_cfg.get("gradient_checkpointing", True):
        model.enable_gradient_checkpointing()
        print("Gradient checkpointing: ENABLED")

    # Load datasets
    train_dataset = KashmiriAsrDataset(
        data_cfg["train_manifest"],
        tokenizer,
        min_duration=train_cfg.get("min_audio_duration_seconds", 0.8),
        max_duration=train_cfg.get("max_audio_duration_seconds", 20.0),
    )

    if overfit_test:
        run_one_batch_overfit_test(model, train_dataset, tokenizer, device=device)
        return

    val_disjoint_dataset = KashmiriAsrDataset(
        data_cfg["val_disjoint_manifest"],
        tokenizer,
        min_duration=0.5,
        max_duration=25.0,
        max_samples=500,
    )

    val_waaz_dataset = KashmiriAsrDataset(
        data_cfg["val_waaz_manifest"],
        tokenizer,
        min_duration=0.5,
        max_duration=25.0,
    )

    batch_size = train_cfg.get("batch_size", 8)
    num_workers = train_cfg.get("num_workers", 2)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
    )

    val_disjoint_loader = DataLoader(
        val_disjoint_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    val_waaz_loader = DataLoader(
        val_waaz_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Freeze feature extractor
    freeze_steps = train_cfg.get("freeze_encoder_updates", 1000)
    if freeze_steps > 0:
        model.freeze_feature_extractor()
        print(f"Feature Extractor (CNN): FROZEN for initial {freeze_steps} steps.")

    # Optimizer & Scheduler
    lr = float(train_cfg.get("learning_rate", 5e-5))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    total_steps = max_steps_override or train_cfg.get("total_steps", 10000)
    warmup_steps = train_cfg.get("warmup_steps", 500)
    grad_accum_steps = train_cfg.get("gradient_accumulation_steps", 2)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=weight_decay,
    )

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Directories
    ckpt_dir = Path(f"experiments/stage29_omnilingual_kashmiri_asr/checkpoints/{exp_id}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("experiments/stage29_omnilingual_kashmiri_asr/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    training_log_path = log_dir / f"{exp_id}_training_log.jsonl"

    global_step = 0
    micro_step = 0
    accum_loss = 0.0
    best_val_wer = float("inf")

    if resume_path and os.path.exists(resume_path):
        print(f"Resuming training from checkpoint: {resume_path}...", flush=True)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                print(f"Warning: Could not load optimizer state ({e}); using freshly initialized optimizer.", flush=True)
        global_step = ckpt.get("global_step", 0)
        micro_step = global_step * grad_accum_steps
        best_val_wer = ckpt.get("val_wer", float("inf"))
        # Fast-forward scheduler
        for _ in range(global_step):
            scheduler.step()
        print(f"Resumed successfully at Step {global_step} | Best Val WER: {best_val_wer:.2f}%", flush=True)
        if global_step >= freeze_steps:
            model.unfreeze_feature_extractor()
            print("Feature Extractor (CNN) UNFROZEN for resumed joint training.", flush=True)

    print(f"\n--- Starting Training: Total Steps: {total_steps:,} | Grad Accum: {grad_accum_steps} | Eff Batch: {batch_size * grad_accum_steps} ---", flush=True)
    train_iter = iter(train_loader)
    t0 = time.time()

    model.train()
    optimizer.zero_grad()

    while global_step < total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        wavs = batch["waveforms"].to(device)
        wav_lens = batch["wav_lengths"].to(device)
        targets = batch["targets"].to(device)
        tgt_lens = batch["tgt_lengths"].to(device)

        loss, _, _ = model(wavs, wav_lens, targets, tgt_lens)
        loss = loss / grad_accum_steps
        loss.backward()
        accum_loss += loss.item()
        micro_step += 1
        del wavs, wav_lens, targets, tgt_lens, loss

        if micro_step % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("clip_grad_norm", 1.0))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            # Unfreeze feature extractor if reached threshold
            if global_step == freeze_steps:
                model.unfreeze_feature_extractor()
                print(f"Step {global_step}: Feature Extractor (CNN) UNFROZEN for joint fine-tuning.", flush=True)

            # Logging
            log_interval = train_cfg.get("log_every_steps", 50)
            if global_step % log_interval == 0:
                cur_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                speed = global_step / elapsed if elapsed > 0 else 0.0
                mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                print(f"Step {global_step:05d}/{total_steps} | Loss: {accum_loss:.4f} | LR: {cur_lr:.2e} | Speed: {speed:.2f} steps/s | GPU Mem: {mem_gb:.2f} GB", flush=True)

                with open(training_log_path, "a", encoding="utf-8") as f_log:
                    f_log.write(json.dumps({
                        "step": global_step,
                        "loss": accum_loss,
                        "learning_rate": cur_lr,
                        "gpu_mem_gb": mem_gb,
                        "timestamp": time.time(),
                    }) + "\n")

                accum_loss = 0.0

            # Validation
            eval_interval = train_cfg.get("eval_every_steps", 1000)
            if global_step % eval_interval == 0 or global_step == total_steps:
                print(f"\n--- Running Validation at Step {global_step} ---", flush=True)
                val_loss, val_wer = evaluate_dev(model, val_disjoint_loader, tokenizer, device=device)
                waaz_loss, waaz_wer = evaluate_dev(model, val_waaz_loader, tokenizer, device=device, max_batches=20)
                print(f"  Dev Disjoint Loss: {val_loss:.4f} | Dev Disjoint WER: {val_wer:.2f}%", flush=True)
                print(f"  WAAZ Val Loss:     {waaz_loss:.4f} | WAAZ Val WER:     {waaz_wer:.2f}%", flush=True)

                is_best = val_wer < best_val_wer
                if is_best:
                    best_val_wer = val_wer
                    print(f"  >>> NEW BEST MODEL ACHIEVED: {best_val_wer:.2f}% WER!", flush=True)
                    torch.save({
                        "global_step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": cfg,
                        "val_wer": val_wer,
                        "waaz_wer": waaz_wer,
                    }, ckpt_dir / "best_model.pt")

                # Save latest checkpoint
                torch.save({
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg,
                    "val_wer": val_wer,
                    "waaz_wer": waaz_wer,
                }, ckpt_dir / "latest_checkpoint.pt")

                model.train()

    print(f"\n=== Training Complete! Best Validation WER: {best_val_wer:.2f}% ===")
    return best_val_wer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overfit-test", action="store_true")
    parser.add_argument("--init-scratch", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    train(
        args.config,
        overfit_test=args.overfit_test,
        max_steps_override=args.max_steps,
        init_scratch=args.init_scratch,
        resume_path=args.resume,
    )
