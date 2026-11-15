#!/usr/bin/env python3
"""
Workstream 8: Stratified Selection and Freezing of the 150-Sample Pilot Benchmark
================================================================================
Selects exactly 50 representative samples from each of the three primary tiers:
- Tier A: Speaker Generalization (IndicVoices-R Quarantined Test)
- Tier B: Real-World Kashmiri (EduCasheer Private Online Audio)
- Tier C: Domain Robustness (King Fahd Kashmiri Translations)

Sampling Methodology:
- Deterministic Seed: 42
- Stratified across duration bins:
    * Short (0.5s - 3.5s): 15 samples
    * Medium (3.5s - 8.0s): 20 samples
    * Long (8.0s - 16.0s): 15 samples
- Speaker-balanced (maximizing unique speaker representation)
- Text length and phonetic variety preserved

Outputs:
- pilot_150_manifest.jsonl
- pilot_150_manifest.csv
- pilot_150_selection_report.json
"""

import json
import csv
import hashlib
from pathlib import Path
import numpy as np

WORKSPACE_DIR = Path("/home/musa/Kashmiri Project")
DATASET_DIR = WORKSPACE_DIR / "kashmiri_dataset"
BENCHMARK_DIR = DATASET_DIR / "experiments/benchmark_construction"
PILOT_DIR = BENCHMARK_DIR / "pilot_150"
PILOT_DIR.mkdir(parents=True, exist_ok=True)

def stratified_sample_tier(tier_samples, tier_name, seed=42, target_count=50):
    rng = np.random.default_rng(seed)
    
    # Partition into duration bins
    short_bin = [s for s in tier_samples if 0.5 <= s["duration"] < 3.5]
    med_bin = [s for s in tier_samples if 3.5 <= s["duration"] < 8.0]
    long_bin = [s for s in tier_samples if 8.0 <= s["duration"] <= 20.0]
    
    print(f"[{tier_name}] Total available: {len(tier_samples)} | Short: {len(short_bin)} | Med: {len(med_bin)} | Long: {len(long_bin)}")
    
    # Target allocations
    n_short = 15
    n_med = 20
    n_long = 15
    
    # If a bin doesn't have enough (e.g. King Fahd may have fewer very long), reallocate gracefully
    if len(long_bin) < n_long:
        shortfall = n_long - len(long_bin)
        n_long = len(long_bin)
        n_med += shortfall
    if len(short_bin) < n_short:
        shortfall = n_short - len(short_bin)
        n_short = len(short_bin)
        n_med += shortfall
        
    def sample_from_bin(items, k):
        # Sort deterministically by hash of sample_id to avoid arbitrary ordering before shuffle
        sorted_items = sorted(items, key=lambda x: hashlib.sha256(x["sample_id"].encode()).hexdigest())
        # Group by speaker to maximize speaker diversity
        spk_groups = {}
        for it in sorted_items:
            spk = it.get("speaker_id", "unknown")
            spk_groups.setdefault(spk, []).append(it)
            
        selected = []
        spk_list = sorted(spk_groups.keys())
        rng.shuffle(spk_list)
        
        # Round-robin selection across speakers
        idx = 0
        while len(selected) < k and any(len(v) > 0 for v in spk_groups.values()):
            curr_spk = spk_list[idx % len(spk_list)]
            if spk_groups[curr_spk]:
                selected.append(spk_groups[curr_spk].pop(0))
            idx += 1
        return selected

    sampled_short = sample_from_bin(short_bin, n_short)
    sampled_med = sample_from_bin(med_bin, n_med)
    sampled_long = sample_from_bin(long_bin, n_long)
    
    selected_tier = sampled_short + sampled_med + sampled_long
    print(f"[{tier_name}] Selected exactly: {len(selected_tier)} samples (Short: {len(sampled_short)}, Med: {len(sampled_med)}, Long: {len(sampled_long)})")
    return selected_tier

def main():
    print("=" * 80)
    print("BUILDING STRATIFIED 150-SAMPLE PILOT BENCHMARK")
    print("=" * 80)
    
    tier_files = {
        "Tier_A": BENCHMARK_DIR / "tier_a_indicvoices_r_quarantined_verified_samples.jsonl",
        "Tier_B": BENCHMARK_DIR / "tier_b_educasheer_online_verified_samples.jsonl",
        "Tier_C": BENCHMARK_DIR / "tier_c_quran_king_fahd_verified_samples.jsonl"
    }
    
    pilot_samples = []
    pilot_stats = {}
    
    for tier_code, fpath in tier_files.items():
        with open(fpath, "r", encoding="utf-8") as f:
            samples = [json.loads(line) for line in f]
            
        sampled = stratified_sample_tier(samples, tier_code, seed=42, target_count=50)
        pilot_samples.extend(sampled)
        
        pilot_stats[tier_code] = {
            "tier_name": sampled[0]["tier"],
            "sample_count": len(sampled),
            "total_duration_sec": round(sum(s["duration"] for s in sampled), 2),
            "mean_duration_sec": round(float(np.mean([s["duration"] for s in sampled])), 2),
            "unique_speakers": len(set(s["speaker_id"] for s in sampled)),
            "mean_word_count": round(float(np.mean([len(s["normalized_text"].split()) for s in sampled])), 2)
        }

    assert len(pilot_samples) == 150, f"Expected 150 pilot samples, got {len(pilot_samples)}"
    
    # Save frozen JSONL
    out_jsonl = PILOT_DIR / "pilot_150_manifest.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for s in pilot_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\nSaved frozen pilot manifest to: {out_jsonl}")
    
    # Save CSV
    out_csv = PILOT_DIR / "pilot_150_manifest.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id", "tier", "dataset_source", "speaker_id", 
            "duration", "sample_rate", "raw_text", "normalized_text", 
            "audio_path", "license", "provenance"
        ])
        for s in pilot_samples:
            writer.writerow([
                s["sample_id"], s["tier"], s["dataset_source"], s["speaker_id"],
                s["duration"], s["sample_rate"], s["raw_text"], s["normalized_text"],
                s["audio_path"], s["license"], s["provenance"]
            ])
    print(f"Saved frozen pilot CSV to: {out_csv}")
    
    # Save selection metadata report
    report_file = PILOT_DIR / "pilot_150_selection_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(pilot_stats, f, indent=2, ensure_ascii=False)
    print(f"Saved pilot selection report to: {report_file}")
    
    print("\n--- Pilot 150 Summary ---")
    for t, st in pilot_stats.items():
        print(f"  {t:8s} ({st['tier_name']}): {st['sample_count']} samples | {st['total_duration_sec']:.1f}s ({st['total_duration_sec']/60:.2f}m) | {st['unique_speakers']} speakers | avg dur: {st['mean_duration_sec']}s | avg words: {st['mean_word_count']}")

if __name__ == "__main__":
    main()
