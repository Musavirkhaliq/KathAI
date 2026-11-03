import json
import random
from pathlib import Path
from collections import Counter

BASE_DIR = Path("/home/musa/Kashmiri Project")
MANIFEST = BASE_DIR / "kashmiri_dataset" / "manifests" / "manifest_master_consolidated.jsonl"
OUT_SPLITS_DIR = BASE_DIR / "kashmiri_dataset" / "experiments" / "native_asr_training" / "data_splits"
OUT_SPLITS_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

print("Building Speaker-Independent Training, Validation, and Test Splits...")

# Categorize speakers and samples
all_records = []
speaker_to_samples = {}
speaker_to_source = {}
tts_samples = []

with open(MANIFEST, "r", encoding="utf-8") as f:
    for line in f:
        it = json.loads(line)
        sid = it.get("sample_id")
        src = it.get("dataset_source", it.get("source", "unknown"))
        spk = it.get("speaker_id", "unknown")
        dur = it.get("duration", 0.0)
        
        # Filter defective duration (<0.3s)
        if dur < 0.3:
            continue
            
        # Isolate dedicated TTS studio data
        if "tts" in src.lower() or "tts" in it.get("speaker_type", "").lower():
            tts_samples.append(it)
            continue
            
        all_records.append(it)
        if spk not in speaker_to_samples:
            speaker_to_samples[spk] = []
            speaker_to_source[spk] = src
        speaker_to_samples[spk].append(it)

print(f"Total Organic Speech Samples: {len(all_records):,}")
print(f"Total Studio TTS Samples Segregated: {len(tts_samples):,}")
print(f"Total Unique Organic Speakers: {len(speaker_to_samples)}")

# Identify multi-sample speakers for validation and test partitioning
# Target: ~10% test (12-14 hrs), ~10% val (12-14 hrs), ~80% train (100+ hrs)
speakers_pool = list(speaker_to_samples.keys())
random.shuffle(speakers_pool)

test_speakers = set()
val_speakers = set()
train_speakers = set()

test_duration = 0.0
val_duration = 0.0
train_duration = 0.0

# Target durations in seconds (~13 hrs = 46,800s)
TARGET_EVAL_DUR = 46800.0

for spk in speakers_pool:
    samples = speaker_to_samples[spk]
    dur = sum(s.get("duration", 0.0) for s in samples)
    
    # Do not assign single large synthetic or unverified groups to test
    if test_duration < TARGET_EVAL_DUR and len(samples) >= 5:
        test_speakers.add(spk)
        test_duration += dur
    elif val_duration < TARGET_EVAL_DUR and len(samples) >= 5:
        val_speakers.add(spk)
        val_duration += dur
    else:
        train_speakers.add(spk)
        train_duration += dur

# Compile final split records
train_records = []
val_records = []
test_records = []

for spk in train_speakers:
    train_records.extend(speaker_to_samples[spk])

for spk in val_speakers:
    val_records.extend(speaker_to_samples[spk])

for spk in test_speakers:
    test_records.extend(speaker_to_samples[spk])

print("\n=== SPLIT SUMMARY ===")
print(f"TRAIN: {len(train_records):6,d} samples | {train_duration/3600.0:6.2f} hrs | {len(train_speakers):4d} speakers")
print(f"VAL:   {len(val_records):6,d} samples | {val_duration/3600.0:6.2f} hrs | {len(val_speakers):4d} speakers")
print(f"TEST:  {len(test_records):6,d} samples | {test_duration/3600.0:6.2f} hrs | {len(test_speakers):4d} speakers")

# Verify zero leakage
overlap_tv = train_speakers.intersection(val_speakers)
overlap_tt = train_speakers.intersection(test_speakers)
overlap_vt = val_speakers.intersection(test_speakers)
assert len(overlap_tv) == 0, f"LEAKAGE train-val: {overlap_tv}"
assert len(overlap_tt) == 0, f"LEAKAGE train-test: {overlap_tt}"
assert len(overlap_vt) == 0, f"LEAKAGE val-test: {overlap_vt}"
print("\nSpeaker Isolation Verification: PASSED (Train ∩ Val ∩ Test = ∅)")

# Write manifest files
def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

write_jsonl(OUT_SPLITS_DIR / "train.jsonl", train_records)
write_jsonl(OUT_SPLITS_DIR / "val.jsonl", val_records)
write_jsonl(OUT_SPLITS_DIR / "test.jsonl", test_records)
write_jsonl(OUT_SPLITS_DIR / "studio_tts_archive.jsonl", tts_samples)

# Generate detailed split audit JSON
audit_meta = {
    "train": {
        "samples": len(train_records),
        "duration_hours": round(train_duration / 3600.0, 2),
        "speaker_count": len(train_speakers),
        "sources": dict(Counter(r.get("dataset_source", r.get("source")) for r in train_records))
    },
    "val": {
        "samples": len(val_records),
        "duration_hours": round(val_duration / 3600.0, 2),
        "speaker_count": len(val_speakers),
        "sources": dict(Counter(r.get("dataset_source", r.get("source")) for r in val_records))
    },
    "test": {
        "samples": len(test_records),
        "duration_hours": round(test_duration / 3600.0, 2),
        "speaker_count": len(test_speakers),
        "sources": dict(Counter(r.get("dataset_source", r.get("source")) for r in test_records))
    },
    "tts_archive": {
        "samples": len(tts_samples),
        "duration_hours": round(sum(s.get("duration", 0.0) for s in tts_samples) / 3600.0, 2),
        "sources": dict(Counter(r.get("dataset_source", r.get("source")) for r in tts_samples))
    }
}

with open(OUT_SPLITS_DIR / "splits_metadata.json", "w", encoding="utf-8") as f:
    json.dump(audit_meta, f, indent=2)

print(f"Data splits successfully generated at: {OUT_SPLITS_DIR}")
