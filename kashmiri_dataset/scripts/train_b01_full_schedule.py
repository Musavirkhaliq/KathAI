import os
import sys
import json
import time
import math
import random
import unicodedata
from collections import Counter
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

# Paths
BASE_DIR = Path("/home/musa/Kashmiri Project")
EXP_DIR = BASE_DIR / "kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc"
SPLITS_DIR = BASE_DIR / "kashmiri_dataset/experiments/native_asr_training/data_splits"
TOK_DIR = EXP_DIR / "tokenizer"
CKPT_DIR = EXP_DIR / "checkpoints"
BASE_XLSR_DIR = CKPT_DIR / "base_xlsr"
BEST_MODEL_DIR = EXP_DIR / "best_model"
LATEST_CKPT_DIR = EXP_DIR / "latest_checkpoint"
STEP_CKPT_DIR = EXP_DIR / "step_checkpoint"

BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
LATEST_CKPT_DIR.mkdir(parents=True, exist_ok=True)
STEP_CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Metrics calculation
def levenshtein_distance(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y), dtype=int)
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix[x, y] = matrix[x-1, y-1]
            else:
                matrix[x, y] = min(
                    matrix[x-1, y] + 1,     # deletion
                    matrix[x-1, y-1] + 1,   # substitution
                    matrix[x, y-1] + 1      # insertion
                )
    return matrix[size_x - 1, size_y - 1]

def normalize_perso_arabic_kashmiri(text):
    if not text:
        return ""
    import re
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u0643", "\u06a9")
    text = text.replace("\u0647", "\u06c1")
    text = text.replace("\u064a", "\u06cc")
    text = re.sub(r'[\u064b-\u0652\u0653\u0654\u0670]', '', text)
    text = re.sub(r'[۔،؛!؟\.,:;?"\'\(\)\[\]\{\}\-\_]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_metrics(ref, hyp):
    ref_words = ref.strip().split()
    hyp_words = hyp.strip().split()
    if len(ref_words) == 0:
        word_dist = len(hyp_words)
        ref_words_count = 0
    else:
        word_dist = levenshtein_distance(ref_words, hyp_words)
        ref_words_count = len(ref_words)

    ref_chars = list(ref.replace(" ", ""))
    hyp_chars = list(hyp.replace(" ", ""))
    if len(ref_chars) == 0:
        char_dist = len(hyp_chars)
        ref_chars_count = 0
    else:
        char_dist = levenshtein_distance(ref_chars, hyp_chars)
        ref_chars_count = len(ref_chars)

    return {
        "word_dist": word_dist,
        "ref_words": ref_words_count,
        "char_dist": char_dist,
        "ref_chars": ref_chars_count
    }

class LengthBucketedDataset:
    def __init__(self, manifest_file, max_duration=16.0, min_duration=0.5, batch_size=16):
        self.samples = []
        filtered_ctc_violations = 0
        with open(manifest_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                dur = item.get("duration", 0)
                if not (min_duration <= dur <= max_duration):
                    continue
                # Enforce CTC feasibility: acoustic frames T = int(dur * 50) must be >= target text length
                t_frames = int(dur * 50)
                target_len = len(unicodedata.normalize("NFKC", item.get("text", "")))
                if target_len >= t_frames or target_len == 0:
                    filtered_ctc_violations += 1
                    continue
                self.samples.append(item)
                    
        # Sort by duration for length-bucketing
        self.samples.sort(key=lambda x: x["duration"])
        self.batch_size = batch_size
        self.batches = [self.samples[i:i+batch_size] for i in range(0, len(self.samples), batch_size)]
        print(f"Loaded {len(self.samples):,} valid samples into {len(self.batches):,} length-bucketed batches (batch_size={batch_size}) from {manifest_file.name}")
        if filtered_ctc_violations > 0:
            print(f"  Safely filtered {filtered_ctc_violations} pathological CTC length violation samples.")

    def get_shuffled_batches(self):
        shuffled = list(self.batches)
        random.shuffle(shuffled)
        return shuffled

def load_batch_tensors(batch_items, processor):
    wavs = []
    texts = []
    sample_ids = []
    durations = []
    raw_texts = []
    
    for item in batch_items:
        wav_path = BASE_DIR / item["audio_path"]
        try:
            audio, sr = sf.read(wav_path)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
        except Exception:
            audio = np.zeros(16000, dtype=np.float32)
            
        text = item["text"]
        norm_text = unicodedata.normalize("NFKC", text).replace(" ", "|")
        wavs.append(audio)
        texts.append(norm_text)
        sample_ids.append(item["sample_id"])
        durations.append(len(audio) / 16000.0)
        raw_texts.append(text)
        
    inputs = processor(wavs, sampling_rate=16000, return_tensors="pt", padding=True)
    labels = processor.tokenizer(texts, return_tensors="pt", padding=True).input_ids
    labels = labels.masked_fill(labels == processor.tokenizer.pad_token_id, -100)
    
    return {
        "input_values": inputs.input_values,
        "labels": labels,
        "sample_ids": sample_ids,
        "durations": durations,
        "raw_texts": raw_texts
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Workstream 8B.1: Stage 4 Full-Schedule Training on {device} (ctc_zero_infinity=True) ===")

    # 1. Load Processor
    processor = Wav2Vec2Processor.from_pretrained(str(TOK_DIR))
    vocab_size = len(processor.tokenizer)
    print(f"Loaded Processor. Tokenizer Vocab Size: {vocab_size}")

    # 2. Datasets
    train_dataset = LengthBucketedDataset(SPLITS_DIR / "train.jsonl", max_duration=16.0, batch_size=16)
    val_dataset = LengthBucketedDataset(SPLITS_DIR / "val.jsonl", max_duration=16.0, batch_size=16)

    # 3. Model Initialization
    # Check if a step checkpoint or latest checkpoint exists to resume
    resumed = False
    state_file = STEP_CKPT_DIR / "training_state.pt"
    if not state_file.exists():
        state_file = LATEST_CKPT_DIR / "training_state.pt"

    if state_file.exists():
        print(f"Found existing training state at {state_file}! Resuming...")
        model = Wav2Vec2ForCTC.from_pretrained(str(state_file.parent))
        model.to(device)
        state = torch.load(state_file, map_location=device)
        start_epoch = state["epoch"]
        start_step = state["global_step"]
        best_val_norm_wer = state.get("best_val_norm_wer", float("inf"))
        patience_counter = state.get("patience_counter", 0)
        resumed = True
        print(f"  Resumed from Epoch {start_epoch}, Step {start_step}, Best WER: {best_val_norm_wer*100:.2f}%")
    else:
        print("Loading base XLS-R 300M model with ctc_zero_infinity=True...")
        model = Wav2Vec2ForCTC.from_pretrained(str(BASE_XLSR_DIR), ctc_zero_infinity=True)
        model.freeze_feature_encoder()
        model.to(device)
        start_epoch = 1
        start_step = 0
        best_val_norm_wer = float("inf")
        patience_counter = 0

    # 4. Hyperparameters
    MAX_EPOCHS = 20
    GRAD_ACCUM_STEPS = 2
    ENC_LR = 1.0e-4
    HEAD_LR = 1.0e-3
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.10
    MAX_GRAD_NORM = 1.0
    PATIENCE = 4

    batches_per_epoch = len(train_dataset.batches)
    total_optimization_steps = (batches_per_epoch // GRAD_ACCUM_STEPS) * MAX_EPOCHS
    warmup_steps = int(total_optimization_steps * WARMUP_RATIO)
    print(f"Batches per epoch: {batches_per_epoch} | Steps per epoch: {batches_per_epoch // GRAD_ACCUM_STEPS}")
    print(f"Total optimization steps: {total_optimization_steps} | Warmup steps: {warmup_steps}")

    # Differential Optimizer
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if "lm_head" not in n and p.requires_grad], "lr": ENC_LR},
        {"params": [p for n, p in model.named_parameters() if "lm_head" in n and p.requires_grad], "lr": HEAD_LR}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=WEIGHT_DECAY)

    # Cosine learning rate scheduler with warmup
    def get_lr_factor(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_optimization_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_factor)

    if resumed:
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])

    # Logging files
    train_log_file = EXP_DIR / "full_training_log.csv"
    epoch_val_file = EXP_DIR / "epoch_validation_metrics.csv"
    resource_file = EXP_DIR / "resource_usage.csv"

    if not resumed or not train_log_file.exists():
        with open(train_log_file, "w", encoding="utf-8") as f:
            f.write("epoch,step,loss,grad_norm,enc_lr,head_lr,vram_gb\n")
    if not resumed or not epoch_val_file.exists():
        with open(epoch_val_file, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_loss,raw_wer,norm_wer,raw_cer,norm_cer,blank_rate,empty_pct,time_sec\n")
    if not resumed or not resource_file.exists():
        with open(resource_file, "w", encoding="utf-8") as f:
            f.write("epoch,duration_sec,audio_hours,throughput_x_rt,vram_peak_gb,gpu_device\n")

    global_step = start_step
    total_audio_sec_processed = 0

    print("\n--- Commencing Stage 4 Full Training Schedule (Robust & Bucketed) ---")
    sys.stdout.flush()
    training_start_time = time.time()

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        epoch_start_time = time.time()
        model.train()
        epoch_train_loss = 0.0
        train_steps_in_epoch = 0
        epoch_audio_sec = 0
        optimizer.zero_grad()

        train_batches = train_dataset.get_shuffled_batches()

        for step, batch_items in enumerate(train_batches):
            batch = load_batch_tensors(batch_items, processor)
            inputs = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            epoch_audio_sec += sum(batch["durations"])

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(inputs, labels=labels)
                loss = outputs.loss / GRAD_ACCUM_STEPS

            # Safety check: skip batch if loss is NaN or Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: NaN/Inf loss encountered at step {step}. Skipping batch.")
                optimizer.zero_grad()
                continue

            loss.backward()
            epoch_train_loss += loss.item() * GRAD_ACCUM_STEPS

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                
                # Safety check: skip step if gradients are NaN or Inf
                if torch.isnan(norm) or torch.isinf(norm):
                    print(f"WARNING: NaN gradient norm encountered at step {step}. Skipping step.")
                    optimizer.zero_grad()
                    continue

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                train_steps_in_epoch += 1

                # Log training progress periodically
                if global_step % 10 == 0:
                    current_enc_lr = scheduler.get_last_lr()[0]
                    current_head_lr = scheduler.get_last_lr()[1]
                    vram_cur = torch.cuda.memory_allocated() / 1e9
                    print(f"Epoch {epoch:2d}/{MAX_EPOCHS:2d} | Step {global_step:5d}/{total_optimization_steps:5d} | "
                          f"Loss: {loss.item()*GRAD_ACCUM_STEPS:.4f} | Grad Norm: {norm.item():.4f} | "
                          f"Enc LR: {current_enc_lr:.2e} | Head LR: {current_head_lr:.2e} | VRAM: {vram_cur:.2f}GB")
                    sys.stdout.flush()
                    with open(train_log_file, "a", encoding="utf-8") as f:
                        f.write(f"{epoch},{global_step},{loss.item()*GRAD_ACCUM_STEPS:.4f},"
                                f"{norm.item():.4f},{current_enc_lr:.6e},{current_head_lr:.6e},{vram_cur:.2f}\n")
                        f.flush()

                # Step checkpoint every 100 steps for safety
                if global_step % 20 == 0:
                    model.save_pretrained(str(STEP_CKPT_DIR))
                    processor.save_pretrained(str(STEP_CKPT_DIR))
                    torch.save({
                        "epoch": epoch,
                        "global_step": global_step,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_val_norm_wer": best_val_norm_wer,
                        "patience_counter": patience_counter
                    }, STEP_CKPT_DIR / "training_state.pt")

        avg_train_loss = epoch_train_loss / max(train_steps_in_epoch, 1)
        epoch_time = time.time() - epoch_start_time
        total_audio_sec_processed += epoch_audio_sec
        audio_hours = epoch_audio_sec / 3600.0
        throughput_x_rt = epoch_audio_sec / max(epoch_time, 1)
        peak_vram = torch.cuda.max_memory_allocated() / 1e9

        print(f"\n[EPOCH {epoch} FINISHED] Avg Train Loss: {avg_train_loss:.4f} | "
              f"Time: {epoch_time/60:.2f}m | Processed: {audio_hours:.2f}h audio ({throughput_x_rt:.1f}x RT) | "
              f"Peak VRAM: {peak_vram:.2f}GB")
        sys.stdout.flush()

        # Resource logging
        with open(resource_file, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{epoch_time:.2f},{audio_hours:.2f},{throughput_x_rt:.2f},{peak_vram:.2f},{torch.cuda.get_device_name(0)}\n")
            f.flush()

        # Validation at epoch end
        print(f"Running validation for Epoch {epoch}...")
        sys.stdout.flush()
        model.eval()
        val_loss_total = 0.0
        val_steps = 0
        total_ref_words_raw = 0
        total_ref_words_norm = 0
        total_word_dist_raw = 0
        total_word_dist_norm = 0
        total_ref_chars_raw = 0
        total_ref_chars_norm = 0
        total_char_dist_raw = 0
        total_char_dist_norm = 0
        total_frames = 0
        total_blank_frames = 0
        empty_predictions_count = 0
        val_predictions = []

        pad_id = processor.tokenizer.pad_token_id

        with torch.no_grad():
            for val_idx, batch_items in enumerate(val_dataset.batches[:32]):
                batch = load_batch_tensors(batch_items, processor)
                inputs = batch["input_values"].to(device)
                labels = batch["labels"].to(device)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(inputs, labels=labels)
                    v_loss = outputs.loss

                val_loss_total += v_loss.item()
                val_steps += 1

                logits = outputs.logits
                pred_ids = torch.argmax(logits, dim=-1)
                pred_texts = processor.batch_decode(pred_ids)

                for b_i in range(len(batch["sample_ids"])):
                    sid = batch["sample_ids"][b_i]
                    ref = batch["raw_texts"][b_i]
                    hyp = pred_texts[b_i]
                    p_seq = pred_ids[b_i]

                    n_f = len(p_seq)
                    n_b = (p_seq == pad_id).sum().item()
                    total_frames += n_f
                    total_blank_frames += n_b

                    if len(hyp.strip()) == 0:
                        empty_predictions_count += 1

                    norm_ref = normalize_perso_arabic_kashmiri(ref)
                    norm_hyp = normalize_perso_arabic_kashmiri(hyp)

                    m_raw = calculate_metrics(ref, hyp)
                    m_norm = calculate_metrics(norm_ref, norm_hyp)

                    total_ref_words_raw += m_raw["ref_words"]
                    total_word_dist_raw += m_raw["word_dist"]
                    total_ref_chars_raw += m_raw["ref_chars"]
                    total_char_dist_raw += m_raw["char_dist"]

                    total_ref_words_norm += m_norm["ref_words"]
                    total_word_dist_norm += m_norm["word_dist"]
                    total_ref_chars_norm += m_norm["ref_chars"]
                    total_char_dist_norm += m_norm["char_dist"]

                    val_predictions.append({
                        "sample_id": sid,
                        "ref_raw": ref,
                        "hyp_raw": hyp,
                        "ref_norm": norm_ref,
                        "hyp_norm": norm_hyp,
                        "blank_ratio": round(n_b / max(n_f, 1), 4)
                    })

        avg_val_loss = val_loss_total / max(val_steps, 1)
        raw_wer = total_word_dist_raw / max(total_ref_words_raw, 1)
        norm_wer = total_word_dist_norm / max(total_ref_words_norm, 1)
        raw_cer = total_char_dist_raw / max(total_ref_chars_raw, 1)
        norm_cer = total_char_dist_norm / max(total_ref_chars_norm, 1)
        blank_rate = total_blank_frames / max(total_frames, 1)
        empty_pct = empty_predictions_count / max(len(val_predictions), 1)

        print(f"Epoch {epoch} Validation: Loss: {avg_val_loss:.4f} | Raw WER: {raw_wer*100:.2f}% | "
              f"Norm WER: {norm_wer*100:.2f}% | Norm CER: {norm_cer*100:.2f}% | "
              f"Blank Rate: {blank_rate*100:.2f}% | Empty Predictions: {empty_pct*100:.2f}%")
        sys.stdout.flush()

        # Print 3 decoding examples
        print("  Sample Predictions:")
        for ex in val_predictions[:3]:
            print(f"    [{ex['sample_id']}] Ref:  {ex['ref_norm']}")
            print(f"                   Hyp:  {ex['hyp_norm']}")
            print(f"                   Blank: {ex['blank_ratio']*100:.1f}%")
        sys.stdout.flush()

        # Save metrics
        with open(epoch_val_file, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{avg_train_loss:.4f},{avg_val_loss:.4f},{raw_wer:.4f},{norm_wer:.4f},"
                    f"{raw_cer:.4f},{norm_cer:.4f},{blank_rate:.4f},{empty_pct:.4f},{epoch_time:.2f}\n")
            f.flush()

        # Save latest checkpoint for resumption
        print("Saving latest checkpoint state...")
        model.save_pretrained(str(LATEST_CKPT_DIR))
        processor.save_pretrained(str(LATEST_CKPT_DIR))
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_norm_wer": best_val_norm_wer,
            "patience_counter": patience_counter
        }, LATEST_CKPT_DIR / "training_state.pt")

        # Check for best model
        if norm_wer < best_val_norm_wer:
            best_val_norm_wer = norm_wer
            patience_counter = 0
            print(f"  >>> (***) NEW BEST VALIDATION WER: {best_val_norm_wer*100:.2f}% -> Saving to best_model/")
            model.save_pretrained(str(BEST_MODEL_DIR))
            processor.save_pretrained(str(BEST_MODEL_DIR))
            with open(EXP_DIR / "validation_predictions.jsonl", "w", encoding="utf-8") as f_vp:
                for vp in val_predictions:
                    f_vp.write(json.dumps(vp, ensure_ascii=False) + "\n")
        else:
            patience_counter += 1
            print(f"  No improvement in Norm WER ({best_val_norm_wer*100:.2f}%). Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at Epoch {epoch}!")
                break
        sys.stdout.flush()

    print(f"\n=== Training Complete. Total Time: {(time.time() - training_start_time)/3600:.2f} hours ===")
    print(f"Best Validation Normalized WER: {best_val_norm_wer*100:.2f}%")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
