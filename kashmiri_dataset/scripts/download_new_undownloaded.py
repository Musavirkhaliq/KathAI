#!/usr/bin/env python3
"""
Downloads ONLY genuinely missing, un-downloaded Kashmiri datasets:
1. programindz/kashmiri-audio-corpus (~332.5 MB, Speech/Audio)
2. Aadilgani/kashmiri-text-dataset (~2.1 MB, Text/NLP)
3. Omarrran/Sample_kashmiri_sft_dataset (~0.03 MB, Text/SFT)
"""

import os
import sys
import time
from pathlib import Path
from huggingface_hub import snapshot_download

TOKEN = os.environ.get("HF_TOKEN", "")
BASE_DIR = Path("/home/musa/Kashmiri Project/kashmiri_dataset/external")
BASE_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [
    {
        "name": "programindz_kashmiri-audio-corpus",
        "repo_id": "programindz/kashmiri-audio-corpus",
        "modality": "Audio / Speech",
        "dest": BASE_DIR / "programindz_kashmiri-audio-corpus"
    },
    {
        "name": "aadilgani_kashmiri-text-dataset",
        "repo_id": "Aadilgani/kashmiri-text-dataset",
        "modality": "Text / Corpus",
        "dest": BASE_DIR / "aadilgani_kashmiri-text-dataset"
    },
    {
        "name": "omarrran_sample_kashmiri_sft_dataset",
        "repo_id": "Omarrran/Sample_kashmiri_sft_dataset",
        "modality": "Instruction / SFT",
        "dest": BASE_DIR / "omarrran_sample_kashmiri_sft_dataset"
    }
]

print("=" * 65)
print("DOWNLOADING TRULY NEW / MISSING KASHMIRI DATASETS")
print("=" * 65)

for item in TARGETS:
    repo = item["repo_id"]
    dest = item["dest"]
    name = item["name"]
    mod = item["modality"]
    
    if dest.exists() and any(dest.iterdir()):
        sz = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
        if sz > 10000:
            print(f"[SKIPPING - ALREADY EXISTS] {name} ({sz/1e6:.2f} MB)")
            continue
            
    print(f"\n>>> Downloading {name} ({repo}) [{mod}]...")
    t0 = time.time()
    try:
        snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            token=TOKEN,
            local_dir=str(dest)
        )
        elapsed = time.time() - t0
        sz = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
        print(f"--> SUCCESS: {name} downloaded in {elapsed:.2f}s ({sz / 1e6:.2f} MB)")
    except Exception as e:
        print(f"--> ERROR downloading {name}: {e}")

print("\n" + "=" * 65)
print("DOWNLOAD OF NEW MISSING DATASETS COMPLETED.")
print("=" * 65)
