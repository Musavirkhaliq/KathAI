import torch
import soundfile as sf
import numpy as np
import json
from pathlib import Path
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

BASE_DIR = Path("/home/musa/Kashmiri Project")
EXP_DIR = BASE_DIR / "kashmiri_dataset" / "experiments" / "native_asr_training" / "experiments" / "B-01-xlsr300m-character-ctc"
TOK_DIR = EXP_DIR / "tokenizer"
CKPT_DIR = EXP_DIR / "checkpoints" / "base_xlsr"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

proc = Wav2Vec2Processor.from_pretrained(str(TOK_DIR))
model = Wav2Vec2ForCTC.from_pretrained(str(CKPT_DIR)).to(device)
model.freeze_feature_encoder()

# Sample 4 audio files
manifest = BASE_DIR / "kashmiri_dataset" / "experiments" / "native_asr_training" / "data_splits" / "train.jsonl"
samples = []
with open(manifest) as f:
    for i, line in enumerate(f):
        if i >= 4: break
        samples.append(json.loads(line))

wavs = []
texts = []
for s in samples:
    a, _ = sf.read(BASE_DIR / s["audio_path"])
    if a.ndim > 1: a = np.mean(a, axis=1)
    wavs.append(a)
    texts.append(s["text"].replace(" ", "|"))

inputs = proc(wavs, sampling_rate=16000, return_tensors="pt", padding=True)
labels = proc.tokenizer(texts, return_tensors="pt", padding=True).input_ids
labels = labels.masked_fill(labels == proc.tokenizer.pad_token_id, -100)

print(f"Inputs shape: {inputs.input_values.shape}, Labels shape: {labels.shape}")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler('cuda', enabled=True)

model.train()
with torch.amp.autocast('cuda'):
    out = model(inputs.input_values.to(device), labels=labels.to(device))
    loss = out.loss

print(f"Initial Loss computed: {loss.item():.4f}")
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
print("Backward step and optimizer update succeeded without error!")
