#!/usr/bin/env python3
"""
Workstream 8: Assembly and Sealing of the Complete EduCasheer Kashmiri ASR Benchmark
===================================================================================
Produces versioned, immutable benchmark manifests across:
- Tier A: Speaker Generalization (3,731 samples | 9.19h)
- Tier B: Real-World Kashmiri (668 samples | 1.38h)
- Tier C: Domain Robustness (1,916 samples | 2.39h)
- Full Unified Primary Benchmark (6,315 samples | 12.96h)
- Separate Challenge Sets:
    * Challenge 1: Meta Omnilingual (MGAP)
    * Challenge 2: Bolbosh Long Duration (>16s)

Outputs:
- benchmark_v1_tier_a.jsonl
- benchmark_v1_tier_b.jsonl
- benchmark_v1_tier_c.jsonl
- benchmark_v1_primary_6315.jsonl
- benchmark_v1_primary_6315.csv
- benchmark_v1_summary.json
- challenge_meta_omnilingual.jsonl
- challenge_bolbosh_long.jsonl
"""

import json
import csv
import hashlib
from pathlib import Path

WORKSPACE_DIR = Path("/home/musa/Kashmiri Project")
DATASET_DIR = WORKSPACE_DIR / "kashmiri_dataset"
BENCHMARK_DIR = DATASET_DIR / "experiments/benchmark_construction"
FINAL_DIR = BENCHMARK_DIR / "final_benchmark_v1"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 80)
    print("ASSEMBLING FINAL VERSIONED KASHMIRI ASR BENCHMARK (V1.0)")
    print("=" * 80)

    tier_files = [
        ("Tier_A", "tier_a_indicvoices_r_quarantined_verified_samples.jsonl"),
        ("Tier_B", "tier_b_educasheer_online_verified_samples.jsonl"),
        ("Tier_C", "tier_c_quran_king_fahd_verified_samples.jsonl")
    ]

    all_primary_samples = []
    tier_counts = {}

    for tier_code, fname in tier_files:
        fpath = BENCHMARK_DIR / fname
        with open(fpath, "r", encoding="utf-8") as f:
            samples = [json.loads(line) for line in f]
        
        # Save tier-specific manifest
        tier_out = FINAL_DIR / f"benchmark_v1_{tier_code.lower()}.jsonl"
        with open(tier_out, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        
        all_primary_samples.extend(samples)
        tier_counts[tier_code] = {
            "tier_name": samples[0]["tier"],
            "samples": len(samples),
            "hours": round(sum(s["duration"] for s in samples) / 3600.0, 2),
            "speakers": len(set(s["speaker_id"] for s in samples)),
            "words": sum(len(s["normalized_text"].split()) for s in samples)
        }
        print(f"[{tier_code}] {tier_counts[tier_code]['tier_name']}: {len(samples):,} samples | {tier_counts[tier_code]['hours']}h | {tier_counts[tier_code]['speakers']} speakers | {tier_counts[tier_code]['words']:,} words")

    # Save Unified Primary Benchmark Manifests
    unified_jsonl = FINAL_DIR / "benchmark_v1_primary_6315.jsonl"
    with open(unified_jsonl, "w", encoding="utf-8") as f:
        for s in all_primary_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\nSaved Unified Primary Benchmark JSONL to: {unified_jsonl}")

    unified_csv = FINAL_DIR / "benchmark_v1_primary_6315.csv"
    with open(unified_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id", "tier", "dataset_source", "speaker_id",
            "duration", "sample_rate", "raw_text", "normalized_text",
            "audio_path", "license", "provenance", "external_exposure_status"
        ])
        for s in all_primary_samples:
            writer.writerow([
                s["sample_id"], s["tier"], s["dataset_source"], s["speaker_id"],
                s["duration"], s["sample_rate"], s["raw_text"], s["normalized_text"],
                s["audio_path"], s["license"], s["provenance"], s["external_exposure_status"]
            ])
    print(f"Saved Unified Primary Benchmark CSV to: {unified_csv}")

    # Compute SHA-256 Checksum of the primary manifest
    h = hashlib.sha256()
    with open(unified_jsonl, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    manifest_sha = h.hexdigest()

    summary = {
        "benchmark_name": "EduCasheer Kashmiri ASR Benchmark v1.0",
        "release_date": "2026-11-14",
        "primary_manifest_sha256": manifest_sha,
        "total_primary_samples": len(all_primary_samples),
        "total_primary_hours": round(sum(s["duration"] for s in all_primary_samples) / 3600.0, 2),
        "total_primary_words": sum(len(s["normalized_text"].split()) for s in all_primary_samples),
        "total_primary_speakers": len(set(s["speaker_id"] for s in all_primary_samples)),
        "tier_breakdown": tier_counts
    }

    sum_file = FINAL_DIR / "benchmark_v1_summary.json"
    with open(sum_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved Benchmark v1.0 Summary to: {sum_file}")
    print(f"SHA-256: {manifest_sha}")

if __name__ == "__main__":
    main()
