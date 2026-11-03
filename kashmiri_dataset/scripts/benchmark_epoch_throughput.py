import json
import time
import unicodedata
import soundfile as sf
import numpy as np
import torch
from pathlib import Path
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

BASE_DIR = Path("/home/musa/Kashmiri Project")
EXP_DIR = BASE_DIR / "kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc"
SPLITS_DIR = BASE_DIR / "kashmiri_dataset/experiments/native_asr_training/data_splits"

processor = Wav2Vec2Processor.from_pretrained(str(EXP_DIR / "tokenizer"))
model = Wav2Vec2ForCTC.from_pretrained(str(EXP_DIR / "checkpoints/base_xlsr")).to("cuda")
model.freeze_feature_encoder()
model.train()

# Load 160 samples
samples = []
with open(SPLITS_DIR / "train.jsonl") as f:
    for i, l in enumerate(f):
        if i >= 160: break
        samples.append(json.loads(l))

optimizer = torch.optim.AdamW([
    {"params": [p for n, p in model.named_parameters() if "lm_head" not in n and p.requires_grad], "lr": 1e-4},
    {"params": [p for n, p in model.named_parameters() if "lm_head" in n and p.requires_grad], "lr": 1e-3}
], weight_decay=0.01)

# Run 10 batches of size 16 (grad accum 2 -> 5 steps)
t0 = time.time()
total_audio_sec = 0
for b in range(10):
    batch_samples = samples[b*16:(b+1)*16]
    wavs = []
    texts = []
    for s in batch_samples:
        w, _ = sf.read(BASE_DIR / s["audio_path"])
        if w.ndim > 1: w = np.mean(w, axis=1)
        wavs.append(w)
        texts.append(unicodedata.normalize("NFKC", s["text"]).replace(" ", "|"))
        total_audio_sec += len(w) / 16000.0
    
    inputs = processor(wavs, sampling_rate=16000, return_tensors="pt", padding=True).input_values.to("cuda")
    labels = processor.tokenizer(texts, return_tensors="pt", padding=True).input_ids.to("cuda")
    labels = labels.masked_fill(labels == processor.tokenizer.pad_token_id, -100)
    
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = model(inputs, labels=labels).loss / 2
    loss.backward()
    if (b + 1) % 2 == 0:
        optimizer.step()
        optimizer.zero_grad()

torch.cuda.synchronize()
elapsed = time.time() - t0
print(f"160 samples (10 batches of 16, {total_audio_sec:.1f} audio sec) took {elapsed:.2f}s")
print(f"Speed: {160 / elapsed:.2f} samples/sec | Throughput: {total_audio_sec / elapsed:.2f}x real-time (RTF={elapsed/total_audio_sec:.4f})")
total_train_sec = 106.67 * 3600 # 384,000 audio seconds
est_epoch_time_sec = total_train_sec / (total_audio_sec / elapsed)
print(f"Estimated time per 1 full epoch (56,517 samples / 106.7 hrs): {est_epoch_time_sec / 60:.1f} minutes ({est_epoch_time_sec / 3600:.2f} hours)")
print(f"Estimated time for 20 epochs: {est_epoch_time_sec * 20 / 3600:.1f} hours")
