import json
import os
import glob
from pathlib import Path
import numpy as np

BASE_DIR = Path("/home/musa/Kashmiri Project")
MANIFEST_DIR = BASE_DIR / "kashmiri_dataset" / "manifests"
SOURCE_MANIFEST_DIR = MANIFEST_DIR / "source_manifests"
REP_DIR = BASE_DIR / "kashmiri_dataset" / "quality_reports"

# 1. Existing baseline manifest
master_baseline = MANIFEST_DIR / "manifest_master.jsonl"
baseline_records = [json.loads(line) for line in open(master_baseline, "r", encoding="utf-8")]
print(f"Existing baseline speech samples: {len(baseline_records):,}")

# 2. Add newly extracted sources
new_source_manifests = [
    SOURCE_MANIFEST_DIR / "manifest_programindz_tts.jsonl",
    SOURCE_MANIFEST_DIR / "manifest_omarrran_tts.jsonl",
    SOURCE_MANIFEST_DIR / "manifest_whisper_kashmiri.jsonl",
    SOURCE_MANIFEST_DIR / "manifest_gaash_bolbosh.jsonl",
    SOURCE_MANIFEST_DIR / "manifest_indicvoices_kashmiri.jsonl"
]

all_records = list(baseline_records)
source_stats = {}

for sm in new_source_manifests:
    if sm.exists():
        src_recs = [json.loads(line) for line in open(sm, "r", encoding="utf-8")]
        src_name = sm.stem.replace("manifest_", "")
        dur_hrs = sum(r.get("duration", 0.0) for r in src_recs) / 3600.0
        spks = len(set(r.get("speaker_id", "unknown") for r in src_recs))
        source_stats[src_name] = {
            "samples": len(src_recs),
            "hours": round(dur_hrs, 2),
            "speakers": spks
        }
        all_records.extend(src_recs)
        print(f"  + {src_name:25s}: {len(src_recs):6,d} samples ({dur_hrs:5.2f} hrs, {spks} speakers)")

# Write updated master manifest
updated_manifest_path = MANIFEST_DIR / "manifest_master_consolidated.jsonl"
with open(updated_manifest_path, "w", encoding="utf-8") as f_out:
    for r in all_records:
        f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

# Copy to KASHMIRI-SPEECH merged directory
merged_speech_manifest = BASE_DIR / "kashmiri_dataset" / "merged" / "KASHMIRI-SPEECH" / "speech_master_manifest.jsonl"
import shutil
shutil.copyfile(updated_manifest_path, merged_speech_manifest)

total_dur_hrs = sum(r.get("duration", 0.0) for r in all_records) / 3600.0
unique_spks = len(set(r.get("speaker_id", "unknown") for r in all_records))

stats = {
    "total_speech_samples": len(all_records),
    "total_speech_hours": round(total_dur_hrs, 2),
    "unique_speakers": unique_spks,
    "sources": source_stats
}

with open(REP_DIR / "consolidated_speech_statistics.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)

print("\n=== Master Speech Manifest Updated ===")
print(f"Total Speech Samples: {len(all_records):,}")
print(f"Total Speech Hours:   {total_dur_hrs:.2f} hours")
print(f"Unique Speakers:      {unique_spks:,}")
print(f"Saved to: {updated_manifest_path} and {merged_speech_manifest}")
