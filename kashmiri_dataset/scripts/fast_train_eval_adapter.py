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

print(f"=== Fast Neural Transliteration Training on {device} ===")
sys.stdout.flush()

train_dataset = TranslitDataset(train_path)
val_dataset = TranslitDataset(val_path, max_samples=500)
test_dataset = TranslitDataset(test_path)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn, num_workers=0)

model = Seq2SeqAdapter(len(src_vocab), len(tgt_vocab), emb_dim=128, hidden_dim=256).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

NUM_EPOCHS = 3
print(f"Training for {NUM_EPOCHS} epochs over {len(train_dataset):,} samples with batch_size=128...")
sys.stdout.flush()

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total_loss = 0.0
    t0 = time.time()
    
    for step, (src_pad, tgt_pad) in enumerate(train_loader):
        src_pad = src_pad.to(device)
        tgt_pad = tgt_pad.to(device)
        
        optimizer.zero_grad()
        logits = model(src_pad, tgt_pad, teacher_forcing_ratio=0.6)
        
        targets = tgt_pad[:, 1:].contiguous()
        loss = criterion(logits.view(-1, len(tgt_vocab)), targets.view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | Time: {time.time()-t0:.1f}s")
    sys.stdout.flush()

# Save neural model
save_dir = "/home/musa/Kashmiri Project/kashmiri_dataset/experiments/romanization_benchmark/neural_model"
os.makedirs(save_dir, exist_ok=True)
torch.save(model.state_dict(), f"{save_dir}/adapter.pt")
print(f"Model saved to {save_dir}/adapter.pt")
sys.stdout.flush()

# Fast Batched Greedy Decoding on Held-Out Test Set
print("\n=== Evaluating on 2,026 Held-Out Test Samples ===")
sys.stdout.flush()
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

test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=0)

with torch.no_grad():
    sample_idx = 0
    for src_pad, tgt_pad in test_loader:
        src_pad = src_pad.to(device)
        gen_tensor = model.generate(src_pad, max_len=100)
        
        for b in range(src_pad.shape[0]):
            hyp_text = ids_to_string(gen_tensor[b].tolist()).strip()
            src_text, ref_text = test_dataset.samples[sample_idx]
            sample_idx += 1
            
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

print(f"\nNeural Adapter Results on Test Set ({total_samples} samples):")
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

print("Evaluation saved successfully!")
