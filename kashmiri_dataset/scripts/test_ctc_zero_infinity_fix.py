import torch
import unicodedata
from pathlib import Path
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

EXP_DIR = Path("kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc")
processor = Wav2Vec2Processor.from_pretrained(str(EXP_DIR / "tokenizer"))

# Load model and test ctc_zero_infinity = True
model = Wav2Vec2ForCTC.from_pretrained(str(EXP_DIR / "checkpoints/base_xlsr"), ctc_zero_infinity=True).to("cuda")

# Construct pathological batch where target length > acoustic frames
# 0.35s audio = 5600 audio samples -> 17 acoustic frames
# target text = 50 characters
dummy_audio = [torch.randn(5600).numpy()]
dummy_text = ["اکھ واریاہ لوٚکُٹ آڈیو چھُ مگر ٹیکسٹ چھُ واریاہ بوٚڈ"] # ~50 chars

inputs = processor(dummy_audio, sampling_rate=16000, return_tensors="pt").input_values.to("cuda")
labels = processor.tokenizer(dummy_text, return_tensors="pt").input_ids.to("cuda")
labels = labels.masked_fill(labels == processor.tokenizer.pad_token_id, -100)

with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    outputs = model(inputs, labels=labels)
    loss = outputs.loss

print(f"Loss with ctc_zero_infinity=True on pathological sample: {loss.item()}")
loss.backward()
grad_norm = model.lm_head.weight.grad.norm().item()
print(f"Gradient norm: {grad_norm} (is finite: {not torch.isnan(torch.tensor(grad_norm))})")
