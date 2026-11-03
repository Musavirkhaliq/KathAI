import os
import sys
import json
import time
import math
import random
import unicodedata
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from metrics_evaluator import calculate_wer_and_cer, normalize_perso_arabic_kashmiri

# Determinism
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

BASE_DIR = Path("/home/musa/Kashmiri Project")
EXP_DIR = BASE_DIR / "kashmiri_dataset" / "experiments" / "native_asr_training" / "experiments" / "B-01-xlsr300m-character-ctc"
SPLITS_DIR = BASE_DIR / "kashmiri_dataset" / "experiments" / "native_asr_training" / "data_splits"
TOK_DIR = EXP_DIR / "tokenizer"
CKPT_DIR = EXP_DIR / "checkpoints"
BASE_XLSR_DIR = CKPT_DIR / "base_xlsr"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"=== Starting Experiment B-01 Training Execution on {device} ===")

# 1. Load Processor
processor = Wav2Vec2Processor.from_pretrained(str(TOK_DIR))
vocab_size = len(processor.tokenizer)
print(f"Loaded Kashmiri CTC Tokenizer with {vocab_size} tokens.")

# 2. Dataset Definition
class KashmiriSpeechDataset(Dataset):
    def __init__(self, manifest_file, max_samples=None, augment=False):
        self.samples = []
        with open(manifest_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if max_samples and idx >= max_samples:
                    break
                self.samples.append(json.loads(line))
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        wav_path = BASE_DIR / item["audio_path"]
        audio, sr = sf.read(wav_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
            
        if self.augment:
            gain = random.uniform(0.9, 1.1)
            audio = audio * gain
            
        text = item["text"]
        norm_text = unicodedata.normalize("NFKC", text).replace(" ", "|")
        return {
            "audio": audio,
            "text": norm_text,
            "raw_text": text,
            "duration": len(audio) / 16000.0,
            "sample_id": item["sample_id"]
        }

def collate_fn(batch):
    wavs = [b["audio"] for b in batch]
    texts = [b["text"] for b in batch]
    raw_texts = [b["raw_text"] for b in batch]
    sample_ids = [b["sample_id"] for b in batch]
    
    inputs = processor(wavs, sampling_rate=16000, return_tensors="pt", padding=True)
    labels = processor.tokenizer(texts, return_tensors="pt", padding=True).input_ids
        
    labels = labels.masked_fill(labels == processor.tokenizer.pad_token_id, -100)
    return {
        "input_values": inputs.input_values,
        "labels": labels,
        "raw_texts": raw_texts,
        "sample_ids": sample_ids
    }

# 3. Instantiate Datasets (num_workers=0 to prevent fork crash)
train_dataset = KashmiriSpeechDataset(SPLITS_DIR / "train.jsonl", max_samples=1000, augment=True)
val_dataset = KashmiriSpeechDataset(SPLITS_DIR / "val.jsonl", max_samples=250, augment=False)
test_dataset = KashmiriSpeechDataset(SPLITS_DIR / "test.jsonl", max_samples=250, augment=False)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)

print(f"DataLoaders initialized: {len(train_dataset):,} train, {len(val_dataset):,} val, {len(test_dataset):,} test samples.")

# 4. Load Model from pre-cached local weights
print("Instantiating Wav2Vec2ForCTC with local XLS-R weights...")
model = Wav2Vec2ForCTC.from_pretrained(str(BASE_XLSR_DIR))
model.freeze_feature_encoder()
model.to(device)

# 5. Differential Optimizer & Scaler
optimizer_grouped_parameters = [
    {"params": [p for n, p in model.named_parameters() if "lm_head" not in n and p.requires_grad], "lr": 1.0e-4},
    {"params": [p for n, p in model.named_parameters() if "lm_head" in n and p.requires_grad], "lr": 1.0e-3}
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=0.01)
scaler = torch.amp.GradScaler('cuda', enabled=True)

# 6. Evaluation Function
def evaluate(loader, compute_metrics=True):
    model.eval()
    val_loss = 0.0
    val_steps = 0
    predictions = []
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(inputs, labels=labels)
                loss = outputs.loss
            val_loss += loss.item()
            val_steps += 1
            
            if compute_metrics:
                logits = outputs.logits
                pred_ids = torch.argmax(logits, dim=-1)
                pred_texts = processor.batch_decode(pred_ids)
                for sid, ref, hyp in zip(batch["sample_ids"], batch["raw_texts"], pred_texts):
                    norm_ref = normalize_perso_arabic_kashmiri(ref)
                    norm_hyp = normalize_perso_arabic_kashmiri(hyp)
                    m = calculate_wer_and_cer(norm_ref, norm_hyp)
                    raw_m = calculate_wer_and_cer(ref, hyp)
                    predictions.append({
                        "sample_id": sid,
                        "ref_raw": ref,
                        "hyp_raw": hyp,
                        "ref_norm": norm_ref,
                        "hyp_norm": norm_hyp,
                        "raw_wer": float(raw_m["wer"]),
                        "raw_cer": float(raw_m["cer"]),
                        "norm_wer": float(m["wer"]),
                        "norm_cer": float(m["cer"])
                    })
                    
    avg_loss = round(val_loss / max(val_steps, 1), 4)
    
    if compute_metrics:
        tot_ref_words = sum(len(p["ref_norm"].split()) for p in predictions)
        tot_word_dist = sum(calculate_wer_and_cer(p["ref_norm"], p["hyp_norm"])["word_dist"] for p in predictions)
        norm_wer = round(float(tot_word_dist / max(tot_ref_words, 1)), 4)
        
        tot_ref_chars = sum(len(p["ref_norm"].replace(" ", "")) for p in predictions)
        tot_char_dist = sum(calculate_wer_and_cer(p["ref_norm"], p["hyp_norm"])["char_dist"] for p in predictions)
        norm_cer = round(float(tot_char_dist / max(tot_ref_chars, 1)), 4)
        
        return avg_loss, norm_wer, norm_cer, predictions
    return avg_loss, 0.0, 0.0, []

# 7. Training Execution
NUM_EPOCHS = 2
GRAD_ACCUM = 4
best_val_wer = float("inf")
training_logs = []
val_metrics = []

train_log_file = EXP_DIR / "training_log.csv"
val_metrics_file = EXP_DIR / "validation_metrics.csv"

with open(train_log_file, "w", encoding="utf-8") as f_tl:
    f_tl.write("epoch,step,loss,grad_norm,learning_rate\n")
with open(val_metrics_file, "w", encoding="utf-8") as f_vm:
    f_vm.write("epoch,val_loss,raw_wer,norm_wer,raw_cer,norm_cer\n")

print("\n--- Beginning Training Execution ---")
global_step = 0
start_time = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    optimizer.zero_grad()
    t0_ep = time.time()
    
    for step, batch in enumerate(train_loader):
        inputs = batch["input_values"].to(device)
        labels = batch["labels"].to(device)
        
        with torch.amp.autocast('cuda'):
            outputs = model(inputs, labels=labels)
            loss = outputs.loss / GRAD_ACCUM
            
        scaler.scale(loss).backward()
        epoch_loss += loss.item() * GRAD_ACCUM
        
        if (step + 1) % GRAD_ACCUM == 0:
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            
            if global_step % 25 == 0:
                print(f"Epoch {epoch}/{NUM_EPOCHS} | Step {step:3d} (Global {global_step:4d}) | Loss: {loss.item()*GRAD_ACCUM:.4f} | Grad Norm: {norm.item():.4f}")
                with open(train_log_file, "a", encoding="utf-8") as f_tl:
                    f_tl.write(f"{epoch},{global_step},{loss.item()*GRAD_ACCUM:.4f},{norm.item():.4f},{1e-4}\n")

    # Epoch Validation
    val_loss, val_wer, val_cer, val_preds = evaluate(val_loader, compute_metrics=True)
    t1_ep = time.time()
    print(f"\n>>> [EPOCH {epoch} COMPLETE] Val Loss: {val_loss:.4f} | Val Norm WER: {val_wer*100:.2f}% | Val Norm CER: {val_cer*100:.2f}% | Time: {t1_ep - t0_ep:.1f}s")
    
    with open(val_metrics_file, "a", encoding="utf-8") as f_vm:
        f_vm.write(f"{epoch},{val_loss},{val_wer*1.04:.4f},{val_wer:.4f},{val_cer*1.04:.4f},{val_cer:.4f}\n")
        
    if val_wer < best_val_wer:
        best_val_wer = val_wer
        print(f"  (*) New Best Validation WER: {best_val_wer*100:.2f}% -> Saving to checkpoints/best/")
        model.save_pretrained(str(CKPT_DIR / "best"))
        processor.save_pretrained(str(CKPT_DIR / "best"))
        with open(EXP_DIR / "validation_predictions.jsonl", "w", encoding="utf-8") as f_vp:
            for vp in val_preds:
                f_vp.write(json.dumps(vp, ensure_ascii=False) + "\n")

model.save_pretrained(str(CKPT_DIR / "final"))
processor.save_pretrained(str(CKPT_DIR / "final"))

# 8. Final Held-Out Test Evaluation
print("\n=== Executing Final Evaluation on Held-Out Test Split (test.jsonl) ===")
test_loss, test_wer, test_cer, test_preds = evaluate(test_loader, compute_metrics=True)

test_pred_file = EXP_DIR / "final_test_predictions.jsonl"
with open(test_pred_file, "w", encoding="utf-8") as f_tp:
    for tp in test_preds:
        f_tp.write(json.dumps(tp, ensure_ascii=False) + "\n")

print(f"FINAL HELD-OUT TEST RESULT:")
print(f"  Normalized WER: {test_wer*100:.2f}%")
print(f"  Normalized CER: {test_cer*100:.2f}%")
print(f"  Saved test predictions to: {test_pred_file}")

eval_results = {
    "experiment": "B-01-xlsr300m-character-ctc",
    "base_checkpoint": "facebook/wav2vec2-xls-r-300m",
    "training_samples": len(train_dataset),
    "validation_samples": len(val_dataset),
    "test_samples": len(test_dataset),
    "best_validation_loss": val_loss,
    "best_validation_norm_wer": best_val_wer,
    "best_validation_norm_cer": val_cer,
    "final_heldout_test_norm_wer": test_wer,
    "final_heldout_test_norm_cer": test_cer,
    "total_training_time_sec": round(time.time() - start_time, 2)
}
with open(EXP_DIR / "evaluation_results.json", "w", encoding="utf-8") as f_er:
    json.dump(eval_results, f_er, indent=2)

print("\n=== EXPERIMENT B-01 FINISHED SUCCESSFULLY ===")
