import os
import sys
import json
import time
import random
import unicodedata
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from metrics_evaluator import levenshtein_distance
from char_seq2seq_adapter import (
    Seq2SeqAdapter, TranslitDataset, collate_fn,
    src_vocab, tgt_vocab, inv_tgt_vocab,
    PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX,
    train_path, val_path, test_path, device
)

print(f"=== Training Neural Transliteration Adapter on {device} ===")

train_dataset = TranslitDataset(train_path)
val_dataset = TranslitDataset(val_path, max_samples=1000)
test_dataset = TranslitDataset(test_path)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

model = Seq2SeqAdapter(len(src_vocab), len(tgt_vocab), emb_dim=128, hidden_dim=256).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

NUM_EPOCHS = 5
print(f"Training for {NUM_EPOCHS} epochs over {len(train_dataset):,} samples...")

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total_loss = 0.0
    t0 = time.time()
    
    for step, (src_pad, tgt_pad) in enumerate(train_loader):
        src_pad = src_pad.to(device)
        tgt_pad = tgt_pad.to(device)
        
        optimizer.zero_grad()
        logits = model(src_pad, tgt_pad, teacher_forcing_ratio=0.5)
        
        # Targets are tgt[:, 1:]
        targets = tgt_pad[:, 1:].contiguous()
        loss = criterion(logits.view(-1, len(tgt_vocab)), targets.view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}/{NUM_EPOCHS} | Train Loss: {avg_loss:.4f} | Time: {time.time()-t0:.1f}s")

# Save neural model
save_dir = "/home/musa/Kashmiri Project/kashmiri_dataset/experiments/romanization_benchmark/neural_model"
os.makedirs(save_dir, exist_ok=True)
torch.save(model.state_dict(), f"{save_dir}/adapter.pt")
with open(f"{save_dir}/vocabs.json", "w") as f:
    json.dump({"src_vocab": src_vocab, "tgt_vocab": tgt_vocab}, f)

print(f"Model saved to {save_dir}/adapter.pt")

# Rigorous Evaluation on Test Set
print("\n=== Running Rigorous Evaluation on 2,026 Held-Out Test Samples ===")
model.eval()

def ids_to_string(ids):
    chars = []
    for idx in ids:
        if idx == EOS_IDX:
            break
        if idx not in (PAD_IDX, SOS_IDX, UNK_IDX):
            chars.append(inv_tgt_vocab.get(idx, ""))
    return "".join(chars)

total_word_dist = 0
total_ref_words = 0
total_char_dist = 0
total_ref_chars = 0
exact_matches = 0
total_samples = 0
predictions = []

with torch.no_grad():
    for src_text, ref_text in test_dataset.samples:
        src_ids = torch.tensor([[src_vocab.get(c, UNK_IDX) for c in src_text]], dtype=torch.long, device=device)
        gen_ids = model.generate(src_ids, max_len=len(src_text)*2 + 20)[0].tolist()
        hyp_text = ids_to_string(gen_ids).strip()
        
        ref_words = ref_text.split()
        hyp_words = hyp_text.split()
        
        w_dist = levenshtein_distance(ref_words, hyp_words)
        total_word_dist += w_dist
        total_ref_words += len(ref_words)
        
        r_chars = list(ref_text.replace(" ", ""))
        h_chars = list(hyp_text.replace(" ", ""))
        
        c_dist = levenshtein_distance(r_chars, h_chars)
        total_char_dist += c_dist
        total_ref_chars += len(r_chars)
        
        if ref_text == hyp_text:
            exact_matches += 1
            
        total_samples += 1
        if len(predictions) < 5:
            predictions.append({"src": src_text, "ref": ref_text, "hyp": hyp_text})

wer = (total_word_dist / max(total_ref_words, 1)) * 100
cer = (total_char_dist / max(total_ref_chars, 1)) * 100
em = (exact_matches / max(total_samples, 1)) * 100

print(f"Neural Adapter Results:")
print(f"  Word Error Rate (WER)     : {wer:.2f}%")
print(f"  Character Error Rate (CER): {cer:.2f}%")
print(f"  Exact Match Accuracy      : {em:.2f}%")

print("\nSample Comparisons (Neural Adapter):")
for p in predictions:
    print(f"  Arabic: {p['src']}")
    print(f"  Ref   : {p['ref']}")
    print(f"  Hyp   : {p['hyp']}")
    print("  ---")

results = {
    "model": "Seq2SeqNeuralAdapter",
    "samples": total_samples,
    "wer": round(wer, 2),
    "cer": round(cer, 2),
    "exact_match_pct": round(em, 2)
}
with open("/home/musa/Kashmiri Project/kashmiri_dataset/experiments/romanization_benchmark/neural_eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

