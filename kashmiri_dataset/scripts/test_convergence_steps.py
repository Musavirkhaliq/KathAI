import sys
import json
import time
import random
import unicodedata
import numpy as np
import soundfile as sf
import torch
from pathlib import Path
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from metrics_evaluator import calculate_wer_and_cer, normalize_perso_arabic_kashmiri

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

BASE_DIR = Path("/home/musa/Kashmiri Project")
EXP_DIR = BASE_DIR / "kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc"
SPLITS_DIR = BASE_DIR / "kashmiri_dataset/experiments/native_asr_training/data_splits"

processor = Wav2Vec2Processor.from_pretrained(str(EXP_DIR / "tokenizer"))
# Start from scratch with base_xlsr weights
model = Wav2Vec2ForCTC.from_pretrained(str(EXP_DIR / "checkpoints/base_xlsr"))
model.freeze_feature_encoder()
model.to("cuda")

# Take 200 training samples
train_samples = []
with open(SPLITS_DIR / "train.jsonl") as f:
    for i, l in enumerate(f):
        if i >= 200: break
        train_samples.append(json.loads(l))

# Take 20 val samples
val_samples = []
with open(SPLITS_DIR / "val.jsonl") as f:
    for i, l in enumerate(f):
        if i >= 20: break
        val_samples.append(json.loads(l))

optimizer = torch.optim.AdamW([
    {"params": [p for n, p in model.named_parameters() if "lm_head" not in n and p.requires_grad], "lr": 1e-4},
    {"params": [p for n, p in model.named_parameters() if "lm_head" in n and p.requires_grad], "lr": 1e-3}
], weight_decay=0.01)

print("Running fast convergence test (100 steps)...")
model.train()
step = 0
for step in range(1, 101):
    batch = random.sample(train_samples, 4)
    wavs = []
    texts = []
    for s in batch:
        w, _ = sf.read(BASE_DIR / s["audio_path"])
        if w.ndim > 1: w = np.mean(w, axis=1)
        wavs.append(w)
        texts.append(unicodedata.normalize("NFKC", s["text"]).replace(" ", "|"))
    
    inputs = processor(wavs, sampling_rate=16000, return_tensors="pt", padding=True).input_values.to("cuda")
    labels = processor.tokenizer(texts, return_tensors="pt", padding=True).input_ids.to("cuda")
    labels = labels.masked_fill(labels == processor.tokenizer.pad_token_id, -100)
    
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = model(inputs, labels=labels).loss
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
    
    if step % 20 == 0:
        print(f"Step {step:3d} | Loss: {loss.item():.4f}")

# Quick decode test
model.eval()
print("\nDecoding test on 5 validation samples:")
with torch.no_grad():
    for s in val_samples[:5]:
        w, _ = sf.read(BASE_DIR / s["audio_path"])
        if w.ndim > 1: w = np.mean(w, axis=1)
        inputs = processor(w, sampling_rate=16000, return_tensors="pt").input_values.to("cuda")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(inputs).logits[0]
        preds = torch.argmax(logits, dim=-1)
        hyp = processor.decode(preds)
        pad_id = processor.tokenizer.pad_token_id
        blank_pct = (preds == pad_id).float().mean().item() * 100
        sid = s["sample_id"]
        ref = s["text"]
        print(f"[{sid}] Blank: {blank_pct:.1f}% | Decoded: {hyp!r} | Ref: {ref!r}")

