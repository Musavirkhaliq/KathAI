#!/usr/bin/env python3
"""
Create a stratified development pilot manifest of 150 samples across all 4 sources.
"""

import json
from pathlib import Path
import random

VAL_MANIFEST = Path("kashmiri_dataset/experiments/native_asr_training/data_splits/val.jsonl")
OUT_PILOT = Path("kashmiri_dataset/experiments/external_asr_benchmark/pilot_dev_manifest.jsonl")

random.seed(42)

samples_by_src = {
    "GAASH-Lab/Bolbosh": [],
    "humair025/KashmiriSpeech-IndicVoices": [],
    "openslr_slr122": [],
    "xkaab_asr_4hrs": []
}

for line in open(VAL_MANIFEST):
    d = json.loads(line)
    dur = float(d.get("duration", 0))
    if 0.5 <= dur <= 16.0:
        src = d.get("source") or d.get("dataset_source")
        if src in samples_by_src:
            samples_by_src[src].append(d)

target_counts = {
    "GAASH-Lab/Bolbosh": 50,
    "humair025/KashmiriSpeech-IndicVoices": 50,
    "openslr_slr122": 25,
    "xkaab_asr_4hrs": 25
}

pilot_samples = []
for src, count in target_counts.items():
    available = samples_by_src[src]
    sampled = random.sample(available, min(count, len(available)))
    pilot_samples.extend(sampled)

# Shuffle deterministically
random.shuffle(pilot_samples)

with open(OUT_PILOT, "w", encoding="utf-8") as f:
    for s in pilot_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"Created stratified pilot manifest with {len(pilot_samples)} samples at {OUT_PILOT}")
for src in target_counts:
    c = sum(1 for s in pilot_samples if (s.get("source") or s.get("dataset_source")) == src)
    print(f"  {src}: {c} samples")
