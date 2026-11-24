#!/usr/bin/env python3
"""
Canonical ASR Dataset Partitioning System (Release v3.0 Unified Master Splits)
Recovers all usable Kashmiri speech data without waste while maintaining
rigorous, speaker-disjoint benchmark test and validation suites.

Output Splits in kashmiri_dataset/releases/asr_v3_unified_splits/:
  1. test_standard_v3.jsonl         (~1,200 - 1,500 utts, ~3.0 hrs, speaker-disjoint multi-source)
  2. val_standard_v3.jsonl          (~800 - 1,000 utts, ~2.0 hrs, speaker-disjoint multi-source)
  3. test_community_waaz_v3.jsonl   (~150 utts, ~0.35 hrs, held-out real-world community speakers)
  4. val_community_waaz_v3.jsonl    (~100 utts, ~0.25 hrs, held-out real-world community speakers)
  5. train_unified_v3.jsonl         (104,000+ utts, ~210 hrs, ALL remaining clean speech data)
  6. held_out_benchmark_speakers.json (Frozen speaker IDs reserved for evaluation)
"""

import os
import sys
import json
import random
from pathlib import Path
import pandas as pd

WORKSPACE = Path("/home/musa/Kashmiri Project")
sys.path.insert(0, str(WORKSPACE))

from kashmiri_dataset.scripts.metrics_evaluator import normalize_perso_arabic_kashmiri as norm_strict

PARQUET_INVENTORY = WORKSPACE / "docs/data/full_audio_text_inventory/audio_text_pairs.parquet"
OUT_DIR = WORKSPACE / "kashmiri_dataset/releases/asr_v3_unified_splits"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 80)
    print("BUILDING CANONICAL V3.0 UNIFIED ASR DATASET PARTITIONS")
    print("=" * 80)

    # 1. Load full forensic inventory
    print("\n[1/5] Loading master forensic audio-text inventory...")
    df = pd.read_parquet(PARQUET_INVENTORY)

    # 2. Filter genuine, verified, playable Kashmiri Perso-Arabic speech
    print("[2/5] Filtering genuine, verified, playable Kashmiri speech records...")
    pool = df[
        df['has_transcript'] &
        df['final_alignment_category'].isin(['A', 'B']) &
        (df['language'] == 'kas') &
        (df['script'].isin(['kas_Arab', 'Arab']))
    ].copy()

    # Check physical existence
    pool['exists'] = pool['real_path'].apply(os.path.exists)
    pool = pool[pool['exists']].copy()

    # Normalize text and filter empty
    pool['normalized_text'] = pool['text'].apply(lambda t: norm_strict(str(t)) if pd.notna(t) else '')
    pool = pool[pool['normalized_text'].str.strip().str.len() > 0].copy()

    # Duration filtering for acoustic viability [0.5, 25.0]s
    pool = pool[(pool['duration_seconds'] >= 0.5) & (pool['duration_seconds'] <= 25.0)].copy()

    print(f"Total eligible clean speech pool: {len(pool)} utterances ({pool['duration_seconds'].sum()/3600:.2f} hours)")
    print(f"Total distinct speakers: {pool['speaker_id'].nunique()}")

    # 3. Speaker Partitioning Strategy
    print("\n[3/5] Performing speaker-disjoint benchmark allocation...")
    random.seed(42)

    # A. WAAZ Community Speakers
    waaz_records = pool[pool['source_dataset'] == 'WAAZ_Community'].copy()
    waaz_speakers = sorted([s for s in waaz_records['speaker_id'].dropna().unique() if str(s).strip() and str(s) != 'unknown'])
    random.shuffle(waaz_speakers)

    # Reserve 5 speakers for WAAZ test (~150 utts), 3 for WAAZ val (~100 utts)
    waaz_test_speakers = set(waaz_speakers[:5])
    waaz_val_speakers = set(waaz_speakers[5:8])
    waaz_train_speakers = set(waaz_speakers[8:])

    print(f"WAAZ Speakers: {len(waaz_test_speakers)} test, {len(waaz_val_speakers)} val, {len(waaz_train_speakers)} train")

    # B. Standard Benchmark Speakers (from IndicVoices and Gaash)
    # In IndicVoices, speakers have ~40-150 utts. Select balanced speaker groups.
    iv_records = pool[pool['source_dataset'].isin(['IndicVoices_Kashmiri', 'indicvoices_r'])].copy()
    iv_speakers = sorted([s for s in iv_records['speaker_id'].dropna().unique() if str(s).strip() and str(s) != 'unknown'])
    random.shuffle(iv_speakers)

    # 15 speakers for standard test (~1,200 utts), 10 speakers for standard val (~800 utts)
    std_test_speakers = set(iv_speakers[:15])
    std_val_speakers = set(iv_speakers[15:25])
    
    # Check Gaash speakers (each has ~1,000 utts). Put 1 speaker into test and 1 into val for studio-quality representation
    gaash_records = pool[pool['source_dataset'] == 'Gaash_Bolbosh'].copy()
    gaash_speakers = sorted([s for s in gaash_records['speaker_id'].dropna().unique() if str(s).strip() and str(s) != 'unknown'])
    random.shuffle(gaash_speakers)

    std_test_speakers.add(gaash_speakers[0]) # bolbosh test speaker
    std_val_speakers.add(gaash_speakers[1])  # bolbosh val speaker

    # Combine all held-out benchmark speakers
    all_test_speakers = std_test_speakers | waaz_test_speakers
    all_val_speakers = std_val_speakers | waaz_val_speakers
    all_benchmark_speakers = all_test_speakers | all_val_speakers

    print(f"Total benchmark held-out speakers: {len(all_benchmark_speakers)}")
    print(f"  - Test speakers: {len(all_test_speakers)}")
    print(f"  - Val speakers: {len(all_val_speakers)}")

    # 4. Partition the data into the 5 splits
    print("\n[4/5] Partitioning pool into standardized manifests...")

    # WAAZ Community Benchmark Test & Val
    df_waaz_test = pool[pool['source_dataset'] == 'WAAZ_Community'][pool['speaker_id'].isin(waaz_test_speakers)].copy()
    df_waaz_val = pool[pool['source_dataset'] == 'WAAZ_Community'][pool['speaker_id'].isin(waaz_val_speakers)].copy()

    # Standard Benchmark Test & Val
    df_std_test = pool[pool['source_dataset'].isin(['IndicVoices_Kashmiri', 'indicvoices_r', 'Gaash_Bolbosh'])][pool['speaker_id'].isin(std_test_speakers)].copy()
    df_std_val = pool[pool['source_dataset'].isin(['IndicVoices_Kashmiri', 'indicvoices_r', 'Gaash_Bolbosh'])][pool['speaker_id'].isin(std_val_speakers)].copy()

    # Cap test and val to standard sizes if needed
    if len(df_std_test) > 1500:
        df_std_test = df_std_test.sample(n=1500, random_state=42)
    if len(df_std_val) > 1000:
        df_std_val = df_std_val.sample(n=1000, random_state=42)

    eval_audio_ids = set(df_waaz_test['audio_id']) | set(df_waaz_val['audio_id']) | set(df_std_test['audio_id']) | set(df_std_val['audio_id'])

    # Training Pool: Every single other record whose speaker does NOT collide with benchmark!
    df_train = pool[
        (~pool['audio_id'].isin(eval_audio_ids)) &
        (~pool['speaker_id'].isin(all_benchmark_speakers))
    ].copy()

    # 5. Export JSONL manifests and registry
    print("\n[5/5] Writing canonical manifests...")
    splits = {
        "test_standard_v3.jsonl": df_std_test,
        "val_standard_v3.jsonl": df_std_val,
        "test_community_waaz_v3.jsonl": df_waaz_test,
        "val_community_waaz_v3.jsonl": df_waaz_val,
        "train_unified_v3.jsonl": df_train
    }

    summary = {}
    for filename, split_df in splits.items():
        out_path = OUT_DIR / filename
        records = []
        for _, row in split_df.iterrows():
            records.append({
                "sample_id": row['audio_id'],
                "audio_path": row['real_path'],
                "text": row['normalized_text'],
                "source": row['source_dataset'],
                "duration": round(float(row['duration_seconds']), 2),
                "speaker_id": str(row['speaker_id']) if pd.notna(row['speaker_id']) else "unknown"
            })
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        hrs = split_df['duration_seconds'].sum() / 3600.0
        spks = split_df['speaker_id'].nunique()
        summary[filename] = {
            "utterances": len(split_df),
            "hours": round(hrs, 2),
            "speakers": spks,
            "path": str(out_path)
        }
        print(f"  -> {filename:30s}: {len(split_df):6d} utts | {hrs:6.2f} hrs | {spks:3d} speakers")

    # Export Benchmark Speakers Registry
    bench_speakers_data = {
        "description": "Frozen speaker IDs reserved strictly for test and validation benchmarking",
        "standard_test_speakers": sorted(list(std_test_speakers)),
        "standard_val_speakers": sorted(list(std_val_speakers)),
        "waaz_test_speakers": sorted(list(waaz_test_speakers)),
        "waaz_val_speakers": sorted(list(waaz_val_speakers)),
        "all_held_out_speakers": sorted(list(all_benchmark_speakers))
    }
    with open(OUT_DIR / "held_out_benchmark_speakers.json", "w", encoding="utf-8") as f:
        json.dump(bench_speakers_data, f, indent=2)

    with open(OUT_DIR / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"CANONICAL V3.0 DATASETS CREATED SUCCESSFULLY IN: {OUT_DIR}")
    print(f"Training Pool: {summary['train_unified_v3.jsonl']['utterances']} utterances ({summary['train_unified_v3.jsonl']['hours']} hours)")
    print(f"Total Held-out Evaluation: {sum(v['utterances'] for k, v in summary.items() if 'train' not in k)} utterances ({sum(v['hours'] for k, v in summary.items() if 'train' not in k):.2f} hours)")
    print("=" * 80)

if __name__ == "__main__":
    main()
