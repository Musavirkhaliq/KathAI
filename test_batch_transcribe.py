from transcribe_kashmiri import KashmiriASR
import json
from pathlib import Path

asr = KashmiriASR()

# Test 3 samples from val.jsonl
samples = []
with open("kashmiri_dataset/experiments/native_asr_training/data_splits/val.jsonl") as f:
    for i, l in enumerate(f):
        if i in [0, 1, 2]:
            samples.append(json.loads(l))

for s in samples:
    path = s["audio_path"]
    ref = s["text"]
    hyp = asr.transcribe(path)
    print("\n" + "-"*60)
    print(f"Sample: {s['sample_id']}")
    print(f"Ref: {ref}")
    print(f"Hyp: {hyp}")
