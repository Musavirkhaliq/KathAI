import json
import unicodedata
import soundfile as sf
import numpy as np
import torch
from pathlib import Path
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

BASE_DIR = Path("/home/musa/Kashmiri Project")
EXP_DIR = BASE_DIR / "kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc"

processor = Wav2Vec2Processor.from_pretrained(str(EXP_DIR / "tokenizer"))
model = Wav2Vec2ForCTC.from_pretrained(str(EXP_DIR / "checkpoints/base_xlsr")).to("cuda")
# Unfreeze top 4 encoder layers as well
for param in model.wav2vec2.parameters():
    param.requires_grad = False
for layer in model.wav2vec2.encoder.layers[-4:]:
    for param in layer.parameters():
        param.requires_grad = True

with open("kashmiri_dataset/experiments/native_asr_training/data_splits/train.jsonl") as f:
    sample = json.loads(f.readline())

wav, _ = sf.read(BASE_DIR / sample["audio_path"])
if wav.ndim > 1: wav = np.mean(wav, axis=1)
text = unicodedata.normalize("NFKC", sample["text"]).replace(" ", "|")

inputs = processor(wav, sampling_rate=16000, return_tensors="pt").input_values.to("cuda")
labels = processor.tokenizer(text, return_tensors="pt").input_ids.to("cuda")
labels = labels.masked_fill(labels == processor.tokenizer.pad_token_id, -100)

optimizer = torch.optim.AdamW([
    {"params": [p for n, p in model.named_parameters() if "lm_head" not in n and p.requires_grad], "lr": 3e-4},
    {"params": [p for n, p in model.named_parameters() if "lm_head" in n and p.requires_grad], "lr": 3e-3}
], weight_decay=0.0)

print(f"Sample: {sample['sample_id']} | Text: {sample['text']} | Target IDs: {labels[0].tolist()}")
print("Running 300 iterations...")

for it in range(1, 301):
    model.train()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = model(inputs, labels=labels).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if it % 50 == 0:
        model.eval()
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(inputs).logits[0]
            preds = torch.argmax(logits, dim=-1)
            hyp = processor.decode(preds)
            pad_id = processor.tokenizer.pad_token_id
            blank_pct = (preds == pad_id).float().mean().item() * 100
            print(f"Iter {it:3d} | Loss: {loss.item():.4f} | Blank: {blank_pct:.1f}% | Decoded: {hyp!r}")

