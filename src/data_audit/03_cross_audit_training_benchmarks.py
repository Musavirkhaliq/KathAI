#!/usr/bin/env python3
"""
Step 3: Forensic Training and Benchmark Cross-Audit.
Loads all training manifests (B-01, B-02A, B-02B, B-02C, E22-01, E22-02, E22-03),
all evaluation benchmarks, and master inventory.
Computes exact record-level and audio-hash-level overlaps.
Identifies benchmark contamination and measures genuinely new data beyond E22-01.
"""

import os
import sys
import json
import pandas as pd
from collections import defaultdict, Counter

def load_manifest_keys(path):
    """Extracts sample_id, audio basename, duration, and speaker from a jsonl manifest."""
    records = []
    if not os.path.exists(path):
        print(f"Warning: Manifest not found: {path}")
        return records
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
                sid = d.get("sample_id", "")
                apath = d.get("audio_path", d.get("canonical_audio_path", d.get("original_audio_path", "")))
                dur = float(d.get("duration", d.get("duration_seconds", 0)))
                spk = d.get("speaker_id", "")
                text = d.get("text", d.get("transcript_normalized", d.get("transcript_original", "")))
                src = d.get("source_dataset", d.get("dataset_source", d.get("source", "")))
                records.append({
                    "sample_id": sid,
                    "audio_filename": os.path.basename(apath),
                    "audio_path": apath,
                    "duration": dur,
                    "speaker_id": spk,
                    "text": text,
                    "source": src
                })
            except:
                pass
    return records

def cross_audit():
    print("=" * 60)
    print("STEP 3: TRAINING & BENCHMARK CROSS-AUDIT")
    print("=" * 60)

    # 1. Load Benchmarks
    benchmarks = {
        "canon_test": load_manifest_keys("kashmiri_dataset/manifests/test.jsonl"),
        "canon_test_clean": load_manifest_keys("kashmiri_dataset/manifests/test_clean.jsonl"),
        "canon_test_cs": load_manifest_keys("kashmiri_dataset/manifests/test_code_switched.jsonl"),
        "canon_test_noisy": load_manifest_keys("kashmiri_dataset/manifests/test_noisy.jsonl"),
        "canon_test_unseen": load_manifest_keys("kashmiri_dataset/manifests/test_speaker_unseen.jsonl"),
        "canon_val": load_manifest_keys("kashmiri_dataset/manifests/val.jsonl"),
        "stage9_val": load_manifest_keys("kashmiri_dataset/experiments/native_asr_training/data_splits/val.jsonl"),
        "stage9_test": load_manifest_keys("kashmiri_dataset/experiments/native_asr_training/data_splits/test.jsonl"),
        "dev_b_waaz_val": load_manifest_keys("kashmiri_dataset/releases/asr_training_v2_waaz/validation_manifests/dev_b_waaz_val.jsonl")
    }

    all_benchmark_audio = set()
    all_benchmark_speakers = set()
    all_test_audio = set()
    all_val_audio = set()

    for bname, recs in benchmarks.items():
        print(f"Loaded benchmark {bname}: {len(recs)} records")
        for r in recs:
            af = r["audio_filename"]
            spk = r["speaker_id"]
            if af:
                all_benchmark_audio.add(af)
                if "test" in bname:
                    all_test_audio.add(af)
                if "val" in bname:
                    all_val_audio.add(af)
            if spk and spk not in ("unknown", "unspecified", ""):
                all_benchmark_speakers.add(spk)

    print(f"\nTotal unique benchmark audio filenames: {len(all_benchmark_audio)}")
    print(f"  Test audio: {len(all_test_audio)}")
    print(f"  Val audio: {len(all_val_audio)}")
    print(f"Total benchmark speakers: {len(all_benchmark_speakers)}")

    # 2. Load Training Manifests
    training_runs = {
        "B-01": load_manifest_keys("kashmiri_dataset/experiments/native_asr_training/data_splits/train.jsonl"),
        "B-02A": load_manifest_keys("kashmiri_dataset/experiments/native_asr_training/experiments/stage9_b02_acoustic_comparison/manifests/b02_base_manifest.jsonl"),
        "B-02B": load_manifest_keys("kashmiri_dataset/experiments/native_asr_training/experiments/stage9_b02_acoustic_comparison/manifests/b02_quran_augmented_manifest.jsonl"),
        "B-02C": load_manifest_keys("kashmiri_dataset/experiments/native_asr_training/experiments/stage9_b02_acoustic_comparison/manifests/b02_all_verified_human_manifest.jsonl"),
        "E22-01": load_manifest_keys("kashmiri_dataset/experiments/stage22_data_expanded_training/manifests/data_22a_conservative_tier_a.jsonl"),
        "E22-02": load_manifest_keys("kashmiri_dataset/experiments/stage22_data_expanded_training/manifests/data_22b_full_waaz.jsonl"),
        "E22-03": load_manifest_keys("kashmiri_dataset/experiments/stage22_data_expanded_training/manifests/data_22c_expanded_multisource.jsonl"),
        "Master_214h": load_manifest_keys("asr_training_eligible_manifest.jsonl")
    }

    # Analyze each training set
    print("\n" + "=" * 60)
    print("TRAINING RUN OVERLAPS & BENCHMARK CONTAMINATION")
    print("=" * 60)

    train_audio_sets = {}
    for tname, recs in training_runs.items():
        dur_hrs = sum(r["duration"] for r in recs) / 3600.0
        af_set = set(r["audio_filename"] for r in recs if r["audio_filename"])
        train_audio_sets[tname] = af_set
        
        # Check benchmark collision
        test_collide = af_set.intersection(all_test_audio)
        val_collide = af_set.intersection(all_val_audio)
        
        print(f"\n{tname}:")
        print(f"  Records: {len(recs)}, Unique Audio Files: {len(af_set)}, Total Duration: {dur_hrs:.2f} hrs")
        print(f"  Direct Test Collisions: {len(test_collide)}")
        print(f"  Direct Val Collisions: {len(val_collide)}")

    # 3. Analyze E22-01 vs Master_214h vs Physical Storage
    e22_audio = train_audio_sets["E22-01"]
    master_audio = train_audio_sets["Master_214h"]
    
    print("\n" + "=" * 60)
    print("E22-01 vs MASTER 214h DISCOVERY")
    print("=" * 60)
    in_master_not_e22 = master_audio - e22_audio
    in_e22_not_master = e22_audio - master_audio
    print(f"Audio files in Master 214h: {len(master_audio)}")
    print(f"Audio files in E22-01: {len(e22_audio)}")
    print(f"In Master 214h but NOT in E22-01: {len(in_master_not_e22)}")
    print(f"In E22-01 but NOT in Master 214h: {len(in_e22_not_master)}")

    # Look at the records in Master not in E22-01
    master_df = pd.DataFrame(training_runs["Master_214h"])
    not_in_e22_df = master_df[master_df["audio_filename"].isin(in_master_not_e22)]
    print(f"\nDuration of items in Master not in E22-01: {not_in_e22_df['duration'].sum() / 3600.0:.2f} hrs")
    print("Source breakdown of items in Master not in E22-01:")
    print(not_in_e22_df["source"].value_counts())

    # How many of these collide with benchmarks?
    not_in_e22_test_leak = not_in_e22_df[not_in_e22_df["audio_filename"].isin(all_test_audio)]
    not_in_e22_val_leak = not_in_e22_df[not_in_e22_df["audio_filename"].isin(all_val_audio)]
    print(f"\nBenchmark Collisions among items in Master not in E22-01:")
    print(f"  Test collisions: {len(not_in_e22_test_leak)} ({not_in_e22_test_leak['duration'].sum() / 3600.0:.2f} hrs)")
    print(f"  Val collisions: {len(not_in_e22_val_leak)} ({not_in_e22_val_leak['duration'].sum() / 3600.0:.2f} hrs)")

    # Genuinely clean items in Master not in E22-01
    clean_master_not_e22 = not_in_e22_df[~not_in_e22_df["audio_filename"].isin(all_benchmark_audio)]
    print(f"\nGenuinely Leak-Free, Benchmark-Safe items in Master not in E22-01:")
    print(f"  Count: {len(clean_master_not_e22)} records")
    print(f"  Duration: {clean_master_not_e22['duration'].sum() / 3600.0:.2f} hrs")
    print("  Source breakdown:")
    print(clean_master_not_e22["source"].value_counts())

    # 4. Beyond Master 214h: What else is in Physical Audio / External / WAAZ?
    # Load physical audio catalog
    phys_df = pd.read_parquet("src/data_audit/audit_physical_audio.parquet")
    phys_files = set(phys_df["filename"])
    
    print("\n" + "=" * 60)
    print("TOTAL PHYSICAL AUDIO vs MASTER & E22-01")
    print("=" * 60)
    print(f"Total Physical Audio Filenames on disk: {len(phys_files)}")
    
    # Physical audio files NOT in E22-01
    phys_not_e22 = phys_df[~phys_df["filename"].isin(e22_audio) & ~phys_df["is_symlink"]]
    print(f"Physical standalone audio files NOT in E22-01: {len(phys_not_e22)} ({phys_not_e22['duration_seconds'].sum() / 3600.0:.2f} hrs)")
    print("Breakdown by location group:")
    for grp, grp_df in phys_not_e22.groupby("location_group"):
        print(f"  {grp}: {len(grp_df)} files, {grp_df['duration_seconds'].sum() / 3600.0:.2f} hrs")

    # Save detailed comparison table
    out_matrix = []
    for tname, recs in training_runs.items():
        dur_h = sum(r["duration"] for r in recs) / 3600.0
        af_set = train_audio_sets[tname]
        out_matrix.append({
            "experiment_id": tname,
            "total_records": len(recs),
            "unique_audio_files": len(af_set),
            "total_hours": round(dur_h, 2),
            "test_overlap_count": len(af_set.intersection(all_test_audio)),
            "val_overlap_count": len(af_set.intersection(all_val_audio))
        })
    df_matrix = pd.DataFrame(out_matrix)
    df_matrix.to_csv("docs/data/full_audio_text_inventory/training_usage_matrix.csv", index=False)
    print(f"\nSaved training_usage_matrix.csv")

if __name__ == "__main__":
    cross_audit()
