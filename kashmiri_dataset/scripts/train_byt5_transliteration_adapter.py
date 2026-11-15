import os
import sys
import json
import time
import re
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Determinism
torch.manual_seed(42)

BASE_DIR = Path("/home/musa/Kashmiri Project")
DATA_DIR = BASE_DIR / "kashmiri_dataset/experiments/transliteration_evaluation/data"
OUT_DIR = BASE_DIR / "kashmiri_dataset/experiments/transliteration_evaluation/byt5_adapter"
CACHED_MODEL_DIR = Path("/home/musa/.cache/huggingface/hub/models--google--byt5-small/snapshots/68377bdc18a2ffec8a0533fef03b1c513a4dd49d")
OUT_DIR.mkdir(parents=True, exist_ok=True)

class TranslitDataset(Dataset):
    def __init__(self, jsonl_file, max_samples=None):
        self.samples = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if max_samples and idx >= max_samples:
                    break
                d = json.loads(line)
                src = d.get("src", "").strip()
                tgt = d.get("tgt", "").strip()
                if src and tgt and len(src) <= 256 and len(tgt) <= 256:
                    self.samples.append((src, tgt))
        print(f"Loaded {len(self.samples):,} pairs from {jsonl_file.name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn_factory(tokenizer):
    def collate_fn(batch):
        sources = [b[0] for b in batch]
        targets = [b[1] for b in batch]
        
        inputs = tokenizer(sources, padding=True, truncation=True, max_length=128, return_tensors="pt")
        labels = tokenizer(targets, padding=True, truncation=True, max_length=128, return_tensors="pt").input_ids
        labels[labels == tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "labels": labels,
            "sources": sources,
            "targets": targets
        }
    return collate_fn

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Training Lightweight ByT5 Transliteration Adapter on {device} ===")
    
    # Load directly from cached snapshot to avoid download checks
    tokenizer = AutoTokenizer.from_pretrained(str(CACHED_MODEL_DIR), local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(CACHED_MODEL_DIR), local_files_only=True)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"ByT5-small model loaded: {total_params/1e6:.1f}M parameters")
    
    # Datasets
    train_ds = TranslitDataset(DATA_DIR / "train.jsonl", max_samples=4000)
    val_ds = TranslitDataset(DATA_DIR / "val.jsonl", max_samples=500)
    test_ds = TranslitDataset(DATA_DIR / "test.jsonl", max_samples=2016)
    
    collate = collate_fn_factory(tokenizer)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate)
    
    # Optimizer (AdamW with 3e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    # Training for 2 epochs with BF16
    MAX_EPOCHS = 2
    print("\n--- Starting Adapter Training (2 Epochs, BF16) ---")
    sys.stdout.flush()
    
    best_val_loss = float("inf")
    t0_train = time.time()
    
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        steps = 0
        t0_ep = time.time()
        
        for step, batch in enumerate(train_loader):
            inputs = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=inputs, attention_mask=mask, labels=labels)
                loss = outputs.loss
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            steps += 1
            
            if (step + 1) % 50 == 0:
                print(f"Epoch {epoch}/{MAX_EPOCHS} | Step {step+1:3d}/{len(train_loader)} | Loss: {loss.item():.4f}")
                sys.stdout.flush()
                
        avg_train_loss = train_loss / steps
        
        # Validation
        model.eval()
        val_loss = 0.0
        v_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(input_ids=inputs, attention_mask=mask, labels=labels)
                    val_loss += outputs.loss.item()
                v_steps += 1
        avg_val_loss = val_loss / v_steps
        
        print(f"\n[Epoch {epoch} Complete] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {time.time()-t0_ep:.1f}s")
        sys.stdout.flush()
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print("  (*) Saving best adapter checkpoint...")
            model.save_pretrained(str(OUT_DIR / "best_checkpoint"))
            tokenizer.save_pretrained(str(OUT_DIR / "best_checkpoint"))
            
    print(f"\nTraining completed in {(time.time()-t0_train)/60:.1f} minutes!")
    
    # Final Rigorous Evaluation on the full 2,016 Held-Out Test Set
    print("\n=== Executing Rigorous Held-Out Test Evaluation (2,016 Sentences) ===")
    sys.stdout.flush()
    best_model = AutoModelForSeq2SeqLM.from_pretrained(str(OUT_DIR / "best_checkpoint")).to(device)
    best_model.eval()
    
    total_samples = 0
    total_char_dist = 0
    total_ref_chars = 0
    total_word_dist = 0
    total_ref_words = 0
    exact_matches = 0
    predictions = []
    
    t0_eval = time.time()
    with torch.no_grad():
        for b_idx, batch in enumerate(test_loader):
            inputs = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                gen_ids = best_model.generate(
                    input_ids=inputs,
                    attention_mask=mask,
                    max_length=128,
                    num_beams=1,
                    early_stopping=True
                )
            gen_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            
            for src, ref_raw, hyp_raw in zip(batch["sources"], batch["targets"], gen_texts):
                ref = normalize_text(ref_raw)
                hyp = normalize_text(hyp_raw)
                
                ref_chars = len(ref.replace(" ", ""))
                hyp_chars = len(hyp.replace(" ", ""))
                c_dist = levenshtein(ref.replace(" ", ""), hyp.replace(" ", ""))
                
                ref_words = ref.split()
                hyp_words = hyp.split()
                w_dist = levenshtein(ref_words, hyp_words)
                
                total_char_dist += c_dist
                total_ref_chars += max(ref_chars, 1)
                total_word_dist += w_dist
                total_ref_words += max(len(ref_words), 1)
                
                if ref == hyp:
                    exact_matches += 1
                total_samples += 1
                
                if total_samples <= 5:
                    predictions.append({"src": src, "ref": ref, "hyp": hyp})
                    
            if (b_idx + 1) % 20 == 0:
                print(f"  Evaluated {total_samples}/{len(test_ds)} test samples...")
                sys.stdout.flush()
                
    elapsed_eval = time.time() - t0_eval
    cer = (total_char_dist / total_ref_chars) * 100
    wer = (total_word_dist / total_ref_words) * 100
    acc = (exact_matches / total_samples) * 100
    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    
    print("\n================ FINAL BYT5 ADAPTER TEST RESULTS ================")
    print(f"Character Error Rate (CER): {cer:.2f}%")
    print(f"Word Error Rate (WER)     : {wer:.2f}%")
    print(f"Exact Match Accuracy      : {acc:.2f}%")
    print(f"Evaluation Throughput     : {total_samples/elapsed_eval:.1f} sentences/sec")
    print(f"Peak VRAM Consumption     : {peak_vram:.2f} GB")
    
    print("\n--- Qualitative ByT5 Predictions ---")
    for p in predictions:
        print("Src:", p["src"][:60])
        print("Ref:", p["ref"][:60])
        print("Hyp:", p["hyp"][:60])
        print("-" * 50)
        
    with open(OUT_DIR / "byt5_metrics.json", "w") as f:
        json.dump({
            "model": "ByT5-Small Transliteration Adapter",
            "test_samples": total_samples,
            "cer": round(cer, 2),
            "wer": round(wer, 2),
            "exact_match_acc": round(acc, 2),
            "latency_ms_per_sample": round((elapsed_eval / total_samples) * 1000, 2),
            "peak_vram_gb": round(peak_vram, 2)
        }, f, indent=2)

if __name__ == "__main__":
    main()
