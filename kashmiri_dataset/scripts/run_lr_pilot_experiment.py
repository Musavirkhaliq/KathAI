import json
import time
import os
import torch
import torch.nn as nn
import soundfile as sf
import numpy as np
from pathlib import Path
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

BASE_DIR = Path("/home/musa/Kashmiri Project")
EXP_DIR = BASE_DIR / "kashmiri_dataset" / "experiments" / "native_asr_training" / "experiments" / "B-01-xlsr300m-character-ctc"
TOK_DIR = EXP_DIR / "tokenizer"
TRAIN_MANIFEST = BASE_DIR / "kashmiri_dataset" / "experiments" / "native_asr_training" / "data_splits" / "train.jsonl"
PILOT_DIR = EXP_DIR / "pilot"
PILOT_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running LR Pilot Experiment on device: {device}")

processor = Wav2Vec2Processor.from_pretrained(str(TOK_DIR))

# Load small pilot batch (200 samples)
pilot_samples = []
with open(TRAIN_MANIFEST, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if idx >= 200: break
        pilot_samples.append(json.loads(line))

print(f"Loaded {len(pilot_samples)} training pilot samples.")

def prepare_batch(samples):
    wavs = []
    texts = []
    for s in samples:
        audio, sr = sf.read(BASE_DIR / s["audio_path"])
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        wavs.append(audio)
        texts.append(s["text"].replace(" ", "|"))
        
    inputs = processor(wavs, sampling_rate=16000, return_tensors="pt", padding=True)
    with processor.as_target_processor():
        labels = processor(texts, return_tensors="pt", padding=True).input_ids
        
    # Replace padding with -100 for CTC loss
    labels = labels.masked_fill(labels == processor.tokenizer.pad_token_id, -100)
    return inputs.input_values.to(device), labels.to(device)

def run_pilot(enc_lr, head_lr, num_steps=20):
    print(f"\n--- Testing Pilot with Encoder LR: {enc_lr}, Head LR: {head_lr} ---")
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-xls-r-300m",
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    ).to(device)
    
    # Freeze CNN feature encoder
    model.freeze_feature_encoder()
    
    # Differential Learning Rates
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if "lm_head" not in n and p.requires_grad], "lr": enc_lr},
        {"params": [p for n, p in model.named_parameters() if "lm_head" in n and p.requires_grad], "lr": head_lr}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=0.01)
    
    losses = []
    grad_norms = []
    
    batch_size = 4
    for step in range(num_steps):
        batch_items = pilot_samples[(step*batch_size) % len(pilot_samples) : ((step+1)*batch_size) % len(pilot_samples)]
        if len(batch_items) < batch_size:
            batch_items = pilot_samples[:batch_size]
            
        inputs, labels = prepare_batch(batch_items)
        
        optimizer.zero_grad()
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  Step {step}: NaN / Inf loss encountered!")
            return {"status": "UNSTABLE", "final_loss": float("nan")}
            
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(round(float(loss.item()), 4))
        grad_norms.append(round(float(norm.item()), 4))
        
        if step % 5 == 0:
            print(f"  Step {step:2d} | Loss: {loss.item():.4f} | Grad Norm: {norm.item():.4f}")
            
    return {
        "status": "STABLE",
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "mean_grad_norm": round(float(np.mean(grad_norms)), 4),
        "loss_curve": losses
    }

# Compare Candidate 1: Conservative (1e-4 enc, 1e-3 head) vs Candidate 2: Aggressive (3e-4 enc, 3e-3 head)
res_c1 = run_pilot(enc_lr=1e-4, head_lr=1e-3)
res_c2 = run_pilot(enc_lr=3e-4, head_lr=3e-3)

pilot_summary = {
    "candidate_1_conservative": {
        "encoder_lr": 1e-4,
        "head_lr": 1e-3,
        "results": res_c1
    },
    "candidate_2_moderate": {
        "encoder_lr": 3e-4,
        "head_lr": 3e-3,
        "results": res_c2
    },
    "selected_configuration": "candidate_1_conservative" if res_c1["final_loss"] < res_c2["final_loss"] else "candidate_2_moderate"
}

with open(PILOT_DIR / "lr_pilot_results.json", "w") as f:
    json.dump(pilot_summary, f, indent=2)

print("\n=== Pilot LR Experiment Complete ===")
print(f"Candidate 1 (1e-4 / 1e-3): Init Loss = {res_c1['initial_loss']} -> Final Loss = {res_c1['final_loss']}")
print(f"Candidate 2 (3e-4 / 3e-3): Init Loss = {res_c2['initial_loss']} -> Final Loss = {res_c2['final_loss']}")
print(f"Selected: {pilot_summary['selected_configuration']}")
