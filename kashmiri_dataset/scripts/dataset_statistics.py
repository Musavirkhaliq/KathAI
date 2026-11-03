#!/usr/bin/env python3
"""
Comprehensive Dataset Quantification & Metric Reporting
========================================================
Calculates exact statistics across both speech and text modalities:
  - Speech hours, clip count, durations (mean, median, min, max)
  - Train / Val / Test speech hours
  - Speaker count, dialect distributions, SNR distribution
  - Code-switching prevalence
  - Text corpus tokens (raw words, unique vocabulary, characters)
  - Comparison of original raw vs external expanded dataset
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List

MANIFEST_DIR = "/home/musa/Kashmiri Project/kashmiri_dataset/manifests"
TEXT_DIR = "/home/musa/Kashmiri Project/kashmiri_dataset/raw/external"
REPORTS_DIR = "/home/musa/Kashmiri Project/kashmiri_dataset/quality_reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

def analyze_speech_dataset() -> Dict[str, any]:
    master_path = os.path.join(MANIFEST_DIR, "manifest_master.jsonl")
    if not os.path.exists(master_path):
        return {"error": "manifest_master.jsonl not found"}

    df = pd.read_json(master_path, lines=True)

    durations = df['duration'].values
    snrs = df['snr'].values
    speakers = df['speaker_id'].unique()
    sources = df['source'].value_counts().to_dict()

    # Split hours
    train_path = os.path.join(MANIFEST_DIR, "train.csv")
    val_path = os.path.join(MANIFEST_DIR, "val.csv")
    test_path = os.path.join(MANIFEST_DIR, "test.csv")

    train_hours = pd.read_csv(train_path)['duration'].sum() / 3600 if os.path.exists(train_path) else 0.0
    val_hours = pd.read_csv(val_path)['duration'].sum() / 3600 if os.path.exists(val_path) else 0.0
    test_hours = pd.read_csv(test_path)['duration'].sum() / 3600 if os.path.exists(test_path) else 0.0

    # Code-switching
    cs_count = 0
    for txt in df['text']:
        if any(ord('a') <= ord(c.lower()) <= ord('z') for c in str(txt)):
            cs_count += 1
    cs_percent = round((cs_count / len(df)) * 100, 2) if len(df) > 0 else 0.0

    return {
        'total_audio_files': len(df),
        'total_speech_hours': round(float(np.sum(durations)) / 3600.0, 2),
        'total_speakers': len(speakers),
        'average_clip_duration_sec': round(float(np.mean(durations)), 2),
        'median_clip_duration_sec': round(float(np.median(durations)), 2),
        'min_clip_duration_sec': round(float(np.min(durations)), 2),
        'max_clip_duration_sec': round(float(np.max(durations)), 2),
        'train_hours': round(train_hours, 2),
        'validation_hours': round(val_hours, 2),
        'test_hours': round(test_hours, 2),
        'average_snr_db': round(float(np.mean(snrs)), 2),
        'estimated_code_switching_percentage': cs_percent,
        'sources_breakdown': sources
    }

def analyze_text_corpora() -> Dict[str, any]:
    """Analyzes text corpora in original and external directories."""
    total_docs = 0
    total_tokens = 0
    vocab = Counter()

    # 1. Kashmiri Language Corpus CSV
    ext_csv = os.path.join(TEXT_DIR, "nawabhussain_corpus/Kashmiri-Language-Corpus.csv")
    if os.path.exists(ext_csv):
        df_c = pd.read_csv(ext_csv)
        col = 'sentence' if 'sentence' in df_c else df_c.columns[0]
        for s in df_c[col].dropna():
            words = str(s).split()
            total_docs += 1
            total_tokens += len(words)
            vocab.update(words)

    # 2. Parallel Corpus
    par_files = glob.glob("/home/musa/Kashmiri Project/Data/*/Kashmiri Project/06_Directly_Parsable_Sources/Kashmiri-English-Parallel-Corpus/Kashmiri.txt")
    if par_files:
        with open(par_files[0], 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line:
                    words = line.split()
                    total_docs += 1
                    total_tokens += len(words)
                    vocab.update(words)

    # 3. Excel Raw Corpus
    excel_files = glob.glob("/home/musa/Kashmiri Project/Data/*/Kashmiri Project/06_Directly_Parsable_Sources/Kashmiri_Language_Corpus_Raw.xlsx")
    if excel_files:
        try:
            df_x = pd.read_excel(excel_files[0])
            for s in df_x['sentence'].dropna():
                words = str(s).split()
                total_docs += 1
                total_tokens += len(words)
                vocab.update(words)
        except Exception:
            pass

    # 4. Sangraha verified split
    sangraha_file = os.path.join(TEXT_DIR, "sangraha/verified/kas/data-0.parquet")
    if os.path.exists(sangraha_file):
        import pyarrow.parquet as pq
        tbl = pq.read_table(sangraha_file)
        df_s = tbl.to_pandas()
        for t in df_s['text'].dropna():
            words = str(t).split()
            total_docs += 1
            total_tokens += len(words)
            vocab.update(words)

    return {
        'total_text_documents': total_docs,
        'total_text_tokens': total_tokens,
        'total_unique_tokens': len(vocab),
        'top_20_vocabulary': [w for w, _ in vocab.most_common(20)]
    }

def generate_report():
    speech_stats = analyze_speech_dataset()
    text_stats = analyze_text_corpora()

    full_stats = {
        'speech_statistics': speech_stats,
        'text_statistics': text_stats
    }

    out_json = os.path.join(REPORTS_DIR, "dataset_summary_statistics.json")
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(full_stats, f, indent=2, ensure_ascii=False)

    print("=== DATASET QUANTIFICATION SUMMARY ===")
    print(f"Total Text Documents:       {text_stats.get('total_text_documents', 0):,}")
    print(f"Total Text Tokens:          {text_stats.get('total_text_tokens', 0):,}")
    print(f"Total Unique Tokens (Vocab):{text_stats.get('total_unique_tokens', 0):,}")
    print(f"Total Audio Files:          {speech_stats.get('total_audio_files', 0):,}")
    print(f"Total Speech Hours:         {speech_stats.get('total_speech_hours', 0.0)} hrs")
    print(f"Total Speakers:             {speech_stats.get('total_speakers', 0)}")
    print(f"Average Clip Duration:      {speech_stats.get('average_clip_duration_sec', 0.0)}s")
    print(f"Median Clip Duration:       {speech_stats.get('median_clip_duration_sec', 0.0)}s")
    print(f"Train Hours:                {speech_stats.get('train_hours', 0.0)} hrs")
    print(f"Validation Hours:           {speech_stats.get('validation_hours', 0.0)} hrs")
    print(f"Test Hours:                 {speech_stats.get('test_hours', 0.0)} hrs")
    print(f"Code-Switching %:           {speech_stats.get('estimated_code_switching_percentage', 0.0)}%")
    print(f"\nSaved detailed JSON report to {out_json}")

if __name__ == '__main__':
    generate_report()
