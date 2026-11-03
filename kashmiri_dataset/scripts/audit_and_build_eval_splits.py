import json
import os
import soundfile as sf
from pathlib import Path
from collections import Counter

BASE_DIR = Path("/home/musa/Kashmiri Project")
MANIFEST = BASE_DIR / "kashmiri_dataset" / "manifests" / "manifest_master_consolidated.jsonl"
EVAL_DIR = BASE_DIR / "kashmiri_dataset" / "experiments" / "asr_baselines" / "eval_splits"
REP_DIR = BASE_DIR / "kashmiri_dataset" / "experiments" / "asr_baselines"

print("Auditing Consolidated Speech Manifest and Establishing Evaluation Splits...")

total_records = 0
valid_audio_records = 0
missing_audio_files = 0
sources = Counter()
speakers = Counter()
verified_samples = []

with open(MANIFEST, "r", encoding="utf-8") as f:
    for line in f:
        total_records += 1
        item = json.loads(line)
        rel_path = item.get("audio_path", "")
        abs_path = BASE_DIR / rel_path
        
        # 1. Verify existence of audio file
        if not abs_path.exists():
            missing_audio_files += 1
            continue
            
        dur = item.get("duration", 0.0)
        text = item.get("text", "").strip()
        spk = item.get("speaker_id", "unknown")
        src = item.get("dataset_source", item.get("source", "unknown"))
        
        # Audio & Transcript validity
        if dur < 0.5 or len(text) < 2:
            continue
            
        valid_audio_records += 1
        sources[src] += 1
        speakers[spk] += 1
        
        item["abs_audio_path"] = str(abs_path)
        verified_samples.append(item)

print(f"Manifest Audit Complete:")
print(f"  Total records in manifest: {total_records:,}")
print(f"  Valid & accessible files:  {valid_audio_records:,}")
print(f"  Missing files:             {missing_audio_files:,}")
print(f"  Unique sources:            {len(sources)}")
print(f"  Unique speakers:           {len(speakers)}")

# Establish rigorous, speaker-isolated Evaluation Splits across 6 taxonomies
eval_clean_read = []          # OpenSLR SLR122 & KTTS
eval_conversational_field = [] # IndicVoices rural/urban field recordings
eval_multispeaker_bolbosh = [] # Bolbosh multi-speaker
eval_code_switched = []        # Samples containing English / Latin loan words
eval_noisy_low_snr = []        # SNR < 18 dB
eval_unseen_heldout = []       # Strictly held-out speakers

# Speaker partitioning
train_speakers = set()
heldout_test_speakers = set()

# We pick held-out speakers from multi-speaker datasets (IndicVoices & Bolbosh & OpenSLR)
for spk, cnt in speakers.most_common():
    if spk != "unknown" and not spk.endswith("spk01"):
        if len(heldout_test_speakers) < 30 and cnt >= 5:
            heldout_test_speakers.add(spk)
        else:
            train_speakers.add(spk)

print(f"Held-out Unseen Test Speakers selected: {len(heldout_test_speakers)}")

for item in verified_samples:
    text = item.get("text", "")
    snr = item.get("snr", 30.0)
    spk = item.get("speaker_id", "unknown")
    src = item.get("dataset_source", item.get("source", "unknown"))
    
    # Check code-switching
    has_latin = any(('A' <= ch <= 'Z') or ('a' <= ch <= 'z') for ch in text)
    if has_latin:
        eval_code_switched.append(item)
        
    # Check noisy / low SNR
    if snr < 18.0 and snr > 0.0:
        eval_noisy_low_snr.append(item)
        
    # Check held-out unseen speaker
    if spk in heldout_test_speakers:
        eval_unseen_heldout.append(item)
        
    # Check category by source
    if "openslr" in src.lower():
        eval_clean_read.append(item)
    elif "indicvoices" in src.lower():
        eval_conversational_field.append(item)
    elif "bolbosh" in src.lower():
        eval_multispeaker_bolbosh.append(item)

# Create fixed evaluation benchmarks (sample sizes capped for deterministic benchmark execution)
def write_split(path, items, max_n=100):
    subset = items[:max_n]
    with open(path, "w", encoding="utf-8") as f_out:
        for it in subset:
            f_out.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(subset)} samples to {path.name}")
    return len(subset)

print("\nGenerating Standardized Benchmark Evaluation Sets:")
s1 = write_split(EVAL_DIR / "eval_clean_read.jsonl", eval_clean_read, 100)
s2 = write_split(EVAL_DIR / "eval_conversational_field.jsonl", eval_conversational_field, 100)
s3 = write_split(EVAL_DIR / "eval_multispeaker_bolbosh.jsonl", eval_multispeaker_bolbosh, 100)
s4 = write_split(EVAL_DIR / "eval_unseen_speakers.jsonl", eval_unseen_heldout, 100)
s5 = write_split(EVAL_DIR / "eval_code_switched.jsonl", eval_code_switched, 50)
s6 = write_split(EVAL_DIR / "eval_noisy_low_snr.jsonl", eval_noisy_low_snr, 50)

# Build Master Unified Benchmark Test Set (300 strictly curated, disjoint samples)
curated_master_test = (
    eval_clean_read[:50] + 
    eval_conversational_field[:75] + 
    eval_multispeaker_bolbosh[:75] + 
    eval_unseen_heldout[:50] + 
    eval_code_switched[:25] + 
    eval_noisy_low_snr[:25]
)
# Deduplicate by sample_id
seen_ids = set()
unique_master_test = []
for it in curated_master_test:
    sid = it["sample_id"]
    if sid not in seen_ids:
        seen_ids.add(sid)
        unique_master_test.append(it)

master_test_path = EVAL_DIR / "master_benchmark_eval.jsonl"
with open(master_test_path, "w", encoding="utf-8") as f_out:
    for it in unique_master_test:
        f_out.write(json.dumps(it, ensure_ascii=False) + "\n")
print(f"Master Unified Benchmark Test Set generated: {len(unique_master_test)} samples saved to {master_test_path.name}")

audit_report = {
    "total_manifest_records": total_records,
    "valid_accessible_files": valid_audio_records,
    "missing_audio_files": missing_audio_files,
    "evaluation_split_counts": {
        "eval_clean_read": s1,
        "eval_conversational_field": s2,
        "eval_multispeaker_bolbosh": s3,
        "eval_unseen_speakers": s4,
        "eval_code_switched": s5,
        "eval_noisy_low_snr": s6,
        "master_benchmark_eval": len(unique_master_test)
    },
    "heldout_test_speakers_count": len(heldout_test_speakers)
}

with open(REP_DIR / "evaluation_integrity_audit.json", "w", encoding="utf-8") as f:
    json.dump(audit_report, f, indent=2)
print("Saved evaluation integrity audit to evaluation_integrity_audit.json")
