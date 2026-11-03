import os
import glob
import json
import unicodedata
from collections import Counter
import soundfile as sf
import numpy as np
from pathlib import Path

BASE_DIR = Path("/home/musa/Kashmiri Project")
PROCESSED_AUDIO_DIR = BASE_DIR / "kashmiri_dataset" / "processed" / "audio"
AUDIT_DIR = BASE_DIR / "kashmiri_dataset" / "quality_reports"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

print("Starting Deep Data Audit of Processed Kashmiri Speech & Master Manifest...")

manifest_path = BASE_DIR / "kashmiri_dataset" / "manifests" / "manifest_master.jsonl"

total_records = 0
durations = []
sample_rates = []
snr_estimates = []
char_counter = Counter()
script_counter = Counter()
speaker_counter = Counter()
dataset_counter = Counter()
gender_counter = Counter()
district_counter = Counter()
empty_transcripts = 0
zero_duration = 0

with open(manifest_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        total_records += 1
        dur = item.get("duration", 0.0)
        durations.append(dur)
        if dur <= 0:
            zero_duration += 1
            
        spk = item.get("speaker_id", "UNKNOWN")
        speaker_counter[spk] += 1
        
        src = item.get("source", item.get("dataset_source", "UNKNOWN"))
        dataset_counter[src] += 1
        
        gender = item.get("speaker_gender", "unknown")
        gender_counter[gender] += 1
        
        region = item.get("speaker_region", "unknown")
        district_counter[region] += 1
        
        snr = item.get("snr", 0.0)
        snr_estimates.append(snr)
        
        raw_text = item.get("text", "")
        if not raw_text or len(raw_text.strip()) == 0:
            empty_transcripts += 1
            continue
            
        for ch in raw_text:
            char_counter[ch] += 1
            name = unicodedata.name(ch, "UNKNOWN")
            if "ARABIC" in name:
                script_counter["Perso-Arabic"] += 1
            elif "DEVANAGARI" in name:
                script_counter["Devanagari"] += 1
            elif "LATIN" in name:
                script_counter["Latin/Romanized"] += 1
            elif unicodedata.category(ch).startswith("P"):
                script_counter["Punctuation"] += 1
            elif unicodedata.category(ch).startswith("N"):
                script_counter["Digits"] += 1
            elif unicodedata.category(ch).startswith("Z"):
                script_counter["Whitespace"] += 1
            else:
                script_counter["Other"] += 1

total_duration_sec = sum(durations)
total_duration_hrs = total_duration_sec / 3600.0

audit_summary = {
    "audit_timestamp": "2026-11-02T11:30:00+05:30",
    "speech_metrics": {
        "total_speech_samples": total_records,
        "total_speech_hours": round(total_duration_hrs, 2),
        "mean_duration_sec": round(float(np.mean(durations)), 2) if durations else 0,
        "median_duration_sec": round(float(np.median(durations)), 2) if durations else 0,
        "std_duration_sec": round(float(np.std(durations)), 2) if durations else 0,
        "min_duration_sec": round(float(np.min(durations)), 2) if durations else 0,
        "max_duration_sec": round(float(np.max(durations)), 2) if durations else 0,
        "zero_duration_records": zero_duration,
        "empty_transcripts": empty_transcripts,
        "mean_snr_db": round(float(np.mean(snr_estimates)), 2) if snr_estimates else 0,
        "dataset_distribution": dict(dataset_counter),
        "gender_distribution": dict(gender_counter),
        "region_distribution": dict(district_counter),
        "total_unique_speakers": len(speaker_counter),
        "top_speakers": speaker_counter.most_common(10)
    },
    "text_and_script_metrics": {
        "total_characters_analyzed": sum(char_counter.values()),
        "unique_characters": len(char_counter),
        "script_distribution": dict(script_counter),
        "top_characters": [{"char": ch, "hex": f"U+{ord(ch):04X}", "count": count, "name": unicodedata.name(ch, 'UNKNOWN')} for ch, count in char_counter.most_common(25)]
    }
}

out_file = AUDIT_DIR / "deep_data_audit_speech_and_text.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(audit_summary, f, indent=2, ensure_ascii=False)

print(f"Deep Audit complete!")
print(f"Total Speech Samples: {total_records}")
print(f"Total Audio Duration: {total_duration_hrs:.2f} hours")
print(f"Mean Clip Duration: {np.mean(durations):.2f}s (Median: {np.median(durations):.2f}s)")
print(f"Mean SNR: {np.mean(snr_estimates):.2f} dB")
print(f"Unique Speakers: {len(speaker_counter)}")
print(f"Script breakdown: {dict(script_counter)}")
print(f"Gender breakdown: {dict(gender_counter)}")
print(f"Source breakdown: {dict(dataset_counter)}")
print(f"Report saved to: {out_file}")
