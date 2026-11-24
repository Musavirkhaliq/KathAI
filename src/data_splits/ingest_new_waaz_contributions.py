#!/usr/bin/env python3
"""
Dynamic Ingestion Pipeline for WAAZ Community Speech & New Collections

Connects directly to the DataCollection platform database (datacollect.db),
identifies newly approved community speech contributions, checks against
frozen benchmark speakers, standardizes paths and text, and seamlessly
appends them to the canonical training manifest without benchmark contamination.

Usage:
  python src/data_splits/ingest_new_waaz_contributions.py [--dry-run]
"""

import os
import sys
import json
import sqlite3
import argparse
from pathlib import Path
import soundfile as sf

WORKSPACE = Path("/home/musa/Kashmiri Project")
sys.path.insert(0, str(WORKSPACE))

from kashmiri_dataset.scripts.metrics_evaluator import normalize_perso_arabic_kashmiri as norm_strict

DB_PATH = Path("/home/musa/Desktop/Projects/DataCollection/data/datacollect.db")
AUDIO_DIR_16K = Path("/home/musa/Desktop/Projects/DataCollection/data/processed/audio/16k_mono")
SPLITS_DIR = WORKSPACE / "kashmiri_dataset/releases/asr_v3_unified_splits"
BENCHMARK_SPEAKERS_FILE = SPLITS_DIR / "held_out_benchmark_speakers.json"
TRAIN_MANIFEST = SPLITS_DIR / "train_unified_v3.jsonl"
WAAZ_TEST_MANIFEST = SPLITS_DIR / "test_community_waaz_v3.jsonl"
WAAZ_VAL_MANIFEST = SPLITS_DIR / "val_community_waaz_v3.jsonl"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Inspect and report without modifying manifests")
    args = parser.parse_args()

    print("=" * 80)
    print("WAAZ DYNAMIC INGESTION PIPELINE: RECLAIMING & EXPANDING COMMUNITY SPEECH")
    print(f"Database: {DB_PATH}")
    print(f"Target Training Manifest: {TRAIN_MANIFEST}")
    print("=" * 80)

    if not DB_PATH.exists():
        print(f"[Error] DataCollection database not found at {DB_PATH}")
        sys.exit(1)

    # 1. Load benchmark speakers registry
    with open(BENCHMARK_SPEAKERS_FILE, "r", encoding="utf-8") as f:
        bench_data = json.load(f)
    
    test_speakers = set(bench_data.get("waaz_test_speakers", [])) | set(bench_data.get("standard_test_speakers", []))
    val_speakers = set(bench_data.get("waaz_val_speakers", [])) | set(bench_data.get("standard_val_speakers", []))
    all_bench_speakers = set(bench_data.get("all_held_out_speakers", []))

    print(f"[Benchmark] Loaded {len(all_bench_speakers)} frozen benchmark speakers.")

    # 2. Load existing sample IDs already in train/test/val
    existing_ids = set()
    for manifest_path in [TRAIN_MANIFEST, WAAZ_TEST_MANIFEST, WAAZ_VAL_MANIFEST]:
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        existing_ids.add(json.loads(line.strip())["sample_id"])
    print(f"[Manifests] Existing tracked records across manifests: {len(existing_ids)}")

    # 3. Query DataCollection database for native community contributions
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    query = """
    SELECT 
        r.id,
        COALESCE(r.processed_file_path, r.raw_file_path),
        r.duration_seconds,
        p.prompt_text,
        c.speaker_id,
        s.speaker_code,
        s.district,
        r.status
    FROM recordings r
    JOIN contributions c ON r.contribution_id = c.id
    LEFT JOIN prompts p ON r.prompt_id = p.id
    LEFT JOIN speakers s ON c.speaker_id = s.id
    WHERE (r.status = 'approved' OR r.status = 'completed')
    """
    cur.execute(query)
    rows = cur.fetchall()
    conn.close()

    print(f"[Database] Found {len(rows)} approved/completed community recordings.")

    new_train_records = []
    new_test_records = []
    new_val_records = []
    skipped_existing = 0
    skipped_missing_audio = 0
    skipped_empty_text = 0

    for r_id, rel_path, dur, raw_text, spk_id, spk_code, district, status in rows:
        sample_id = f"waaz_db_{r_id}"
        if sample_id in existing_ids:
            skipped_existing += 1
            continue

        # Check audio file
        audio_name = Path(rel_path).name
        audio_16k = AUDIO_DIR_16K / audio_name
        audio_orig = Path("/home/musa/Desktop/Projects/DataCollection") / rel_path

        real_audio_path = None
        if audio_16k.exists():
            real_audio_path = str(audio_16k)
        elif audio_orig.exists():
            real_audio_path = str(audio_orig)
        else:
            skipped_missing_audio += 1
            continue

        # Normalize text
        norm_t = norm_strict(str(raw_text or ""))
        if not norm_t.strip():
            skipped_empty_text += 1
            continue

        # Get duration
        duration = float(dur or 0.0)
        if duration <= 0.0:
            try:
                info = sf.info(real_audio_path)
                duration = round(info.duration, 2)
            except Exception:
                duration = 5.0

        if duration < 0.5 or duration > 25.0:
            continue

        speaker_str = str(spk_code or spk_id or "unknown")

        record = {
            "sample_id": sample_id,
            "audio_path": real_audio_path,
            "text": norm_t,
            "source": "WAAZ_Community_Live",
            "duration": round(duration, 2),
            "speaker_id": speaker_str,
            "district": str(district or "unknown")
        }

        # Check speaker holdout
        if speaker_str in test_speakers:
            new_test_records.append(record)
        elif speaker_str in val_speakers:
            new_val_records.append(record)
        else:
            new_train_records.append(record)

    print("\n[Ingestion Summary]")
    print(f"  - Already Tracked: {skipped_existing}")
    print(f"  - Missing Audio:   {skipped_missing_audio}")
    print(f"  - Empty Text:      {skipped_empty_text}")
    print(f"  - New Train Clips: {len(new_train_records)} ({sum(r['duration'] for r in new_train_records)/3600:.2f} hrs)")
    print(f"  - New Test Clips:  {len(new_test_records)} ({sum(r['duration'] for r in new_test_records)/3600:.2f} hrs)")
    print(f"  - New Val Clips:   {len(new_val_records)} ({sum(r['duration'] for r in new_val_records)/3600:.2f} hrs)")

    if args.dry_run:
        print("\n[Dry Run] No files modified.")
        return

    # Append to manifests
    if new_train_records:
        with open(TRAIN_MANIFEST, "a", encoding="utf-8") as f:
            for r in new_train_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[Success] Appended {len(new_train_records)} records to {TRAIN_MANIFEST}")

    if new_test_records:
        with open(WAAZ_TEST_MANIFEST, "a", encoding="utf-8") as f:
            for r in new_test_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[Success] Appended {len(new_test_records)} records to {WAAZ_TEST_MANIFEST}")

    if new_val_records:
        with open(WAAZ_VAL_MANIFEST, "a", encoding="utf-8") as f:
            for r in new_val_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[Success] Appended {len(new_val_records)} records to {WAAZ_VAL_MANIFEST}")

    print("=" * 80)
    print("INGESTION PIPELINE COMPLETED SUCCESSFULLY.")
    print("=" * 80)

if __name__ == "__main__":
    main()
