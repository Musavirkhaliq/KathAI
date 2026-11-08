#!/usr/bin/env python3
"""
Downloads all newly unlocked / missing Kashmiri datasets across Modalities:
1. Speech / Audio:
   - Faizaniqbal/kashmiri-audio-corpus (~332 MB)
2. Text / LLM Pretraining:
   - Omarrran/3.1Million_KASHMIRI_text_Pre_training_Dataset_for_LLM_2026_by_HNM (~29.6 MB)
   - Omarrran/KS-PRET-5M_5_million_kashmiri_Pretrainning_LLM_dataset_12M_tokens_2026 (~50.2 MB)
   - Omarrran/kashmiri_parallel_Diacratic_to_Non_diacratic_Text_dataset (~23.0 MB)
   - Omarrran/Kashmiri_News_Articles_Text_Dataset (~9.4 MB)
   - Omarrran/Kashmiri_Poet_dataset_Shruk (~0.36 MB)
   - Omarrran/Kashmiri__Text_Corpus_Dataset (~10 KB)
3. Parallel / Translation:
   - SMUQamar/Kashmiri-English-Dataset-270K (~40 MB)
   - SMUQamar/Kashmiri-English-Parallel-Corpus (~15 MB)
4. OCR Image / Text Recognition:
   - Omarrran/600k_KS_OCR_Word_Segmented_Dataset (~10.37 GB)
   - Omarrran/31K_Kashmiri_text_and_image_dataset_for_text_Recognition (~1.2 GB)
   - Omarrran/40K_kashmiri_text_and_image_dataset (~1.5 GB)
   - Omarrran/Koshur_Pixel (Complete rest of 40.8 GB)
"""

import os
import sys
import time
from pathlib import Path
from huggingface_hub import snapshot_download

TOKEN = os.environ.get("HF_TOKEN", "")
BASE_DIR = Path("/home/musa/Kashmiri Project/kashmiri_dataset/raw/external")
BASE_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    # --- Priority 1: Audio / Speech ---
    {
        "name": "faizaniqbal_kashmiri_audio_corpus",
        "repo_id": "Faizaniqbal/kashmiri-audio-corpus",
        "category": "speech_audio",
        "target_dir": BASE_DIR / "faizaniqbal_audio_corpus"
    },
    # --- Priority 2: Text / Pretraining ---
    {
        "name": "omarrran_ks_lit_3m",
        "repo_id": "Omarrran/3.1Million_KASHMIRI_text_Pre_training_Dataset_for_LLM_2026_by_HNM",
        "category": "text_pretraining",
        "target_dir": BASE_DIR / "omarrran_ks_lit_3m"
    },
    {
        "name": "omarrran_ks_pret_5m",
        "repo_id": "Omarrran/KS-PRET-5M_5_million_kashmiri_Pretrainning_LLM_dataset_12M_tokens_2026",
        "category": "text_pretraining",
        "target_dir": BASE_DIR / "omarrran_ks_pret_5m"
    },
    {
        "name": "omarrran_diacritic_parallel",
        "repo_id": "Omarrran/kashmiri_parallel_Diacratic_to_Non_diacratic_Text_dataset",
        "category": "text_normalization",
        "target_dir": BASE_DIR / "omarrran_diacritic_parallel"
    },
    {
        "name": "omarrran_news_articles",
        "repo_id": "Omarrran/Kashmiri_News_Articles_Text_Dataset",
        "category": "text_news",
        "target_dir": BASE_DIR / "omarrran_news_articles"
    },
    {
        "name": "omarrran_poet_shruk",
        "repo_id": "Omarrran/Kashmiri_Poet_dataset_Shruk",
        "category": "text_poetry",
        "target_dir": BASE_DIR / "omarrran_poet_shruk"
    },
    {
        "name": "omarrran_text_corpus",
        "repo_id": "Omarrran/Kashmiri__Text_Corpus_Dataset",
        "category": "text_general",
        "target_dir": BASE_DIR / "omarrran_text_corpus"
    },
    # --- Priority 3: Parallel / Translation ---
    {
        "name": "smuqamar_270k",
        "repo_id": "SMUQamar/Kashmiri-English-Dataset-270K",
        "category": "parallel_translation",
        "target_dir": BASE_DIR / "smuqamar_270k"
    },
    {
        "name": "smuqamar_parallel_corpus",
        "repo_id": "SMUQamar/Kashmiri-English-Parallel-Corpus",
        "category": "parallel_translation",
        "target_dir": BASE_DIR / "smuqamar_parallel_corpus"
    },
    # --- Priority 4: OCR Image Datasets ---
    {
        "name": "omarrran_31k_text_recognition",
        "repo_id": "Omarrran/31K_Kashmiri_text_and_image_dataset_for_text_Recognition",
        "category": "ocr_text_recognition",
        "target_dir": BASE_DIR / "omarrran_31k_recognition"
    },
    {
        "name": "omarrran_40k_text_image",
        "repo_id": "Omarrran/40K_kashmiri_text_and_image_dataset",
        "category": "ocr_text_image",
        "target_dir": BASE_DIR / "omarrran_40k_text_image"
    },
    {
        "name": "omarrran_600k_ks_ocr",
        "repo_id": "Omarrran/600k_KS_OCR_Word_Segmented_Dataset",
        "category": "ocr_word_segmented",
        "target_dir": BASE_DIR / "omarrran_600k_ks_ocr"
    },
    {
        "name": "omarrran_koshur_pixel",
        "repo_id": "Omarrran/Koshur_Pixel",
        "category": "ocr_synthetic",
        "target_dir": BASE_DIR / "omarrran_koshur_pixel"
    }
]

print("=" * 70)
print("COMMENCING DOWNLOAD OF ALL UNLOCKED KASHMIRI RESEARCH DATASETS")
print("=" * 70)
total_start = time.time()

results = []

for idx, item in enumerate(DATASETS, 1):
    repo = item["repo_id"]
    target = item["target_dir"]
    name = item["name"]
    cat = item["category"]
    
    print(f"\n[{idx}/{len(DATASETS)}] Downloading {name} ({repo}) -> {target.name}...")
    target.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    
    try:
        snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            token=TOKEN,
            local_dir=str(target),
            max_workers=8
        )
        elapsed = time.time() - t0
        total_sz = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
        print(f"SUCCESS: {name} downloaded in {elapsed:.1f}s ({total_sz / (1024**2):.2f} MB)")
        results.append({
            "name": name,
            "repo_id": repo,
            "status": "SUCCESS",
            "size_mb": round(total_sz / (1024**2), 2),
            "time_sec": round(elapsed, 1),
            "path": str(target)
        })
    except Exception as e:
        elapsed = time.time() - t0
        print(f"ERROR downloading {name}: {e}")
        results.append({
            "name": name,
            "repo_id": repo,
            "status": f"FAILED: {e}",
            "size_mb": 0,
            "time_sec": round(elapsed, 1),
            "path": str(target)
        })

print("\n" + "=" * 70)
print(f"ALL DOWNLOADS COMPLETED IN {(time.time() - total_start)/60:.2f} MINUTES")
print("=" * 70)

for r in results:
    print(f"  [{r['status'][:7]}] {r['name']:<35s} | {r['size_mb']:>8.2f} MB | {r['time_sec']:>6.1f}s")

