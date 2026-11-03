import time, json, soundfile as sf, unicodedata, numpy as np, torch
from pathlib import Path
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from torch.utils.data import Dataset, DataLoader, Sampler

BASE_DIR = Path("/home/musa/Kashmiri Project")
SPLITS_DIR = BASE_DIR / "kashmiri_dataset/experiments/native_asr_training/data_splits"
EXP_DIR = BASE_DIR / "kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc"
processor = Wav2Vec2Processor.from_pretrained(str(EXP_DIR / "tokenizer"))
model = Wav2Vec2ForCTC.from_pretrained(str(EXP_DIR / "checkpoints/base_xlsr")).to("cuda")
model.freeze_feature_encoder()
model.train()

# Load samples and filter out > 16.0s (captures 98% of all speech)
samples = []
with open(SPLITS_DIR / "train.jsonl") as f:
    for l in f:
        item = json.loads(l)
        if item.get("duration", 0) <= 16.0 and item.get("duration", 0) >= 0.5:
            samples.append(item)

# Sort by duration for length bucketing
samples.sort(key=lambda x: x["duration"])
print(f"Loaded {len(samples):,} length-bucketed samples (<= 16s)")

# Form batches of 16
batches = [samples[i:i+16] for i in range(0, len(samples), 16)]
# Shuffle batches so batches themselves are random, but within each batch lengths are homogeneous
import random
random.seed(42)
random.shuffle(batches)

print("Benchmarking 10 bucketed batches...")
t0 = time.time()
for b_idx in range(10):
    b = batches[b_idx]
    wavs = []
    texts = []
    for s in b:
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
    print(f"Batch {b_idx}: loss={loss.item():.4f}, input_shape={inputs.shape}, elapsed={time.time()-t0:.2f}s")

elapsed = time.time() - t0
print(f"10 bucketed batches (160 samples) took: {elapsed:.2f}s ({160/elapsed:.1f} samples/sec)!")
