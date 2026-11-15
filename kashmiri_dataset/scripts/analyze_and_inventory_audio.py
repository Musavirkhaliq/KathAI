#!/usr/bin/env python3
"""
Audio Inventory and Metadata Extraction for Kashmiri Quran Archive
==================================================================
Processes all 114 original MP3 recordings, verifies integrity,
extracts technical audio specifications (codec, sample rate, channels,
duration, bitrate), computes SHA-256, and links with Surah metadata.
"""

import os
import sys
import json
import csv
import glob
import hashlib
import subprocess
import soundfile as sf

RAW_AUDIO_DIR = "kashmiri_dataset/raw/external/quran_kashmiri/raw_audio"
METADATA_DIR = "kashmiri_dataset/raw/external/quran_kashmiri/kaeshur-quran-data/kaeshur_quran/metadata"
OUTPUT_JSON = "kashmiri_dataset/raw/external/quran_kashmiri/audio_inventory.json"
OUTPUT_CSV = "kashmiri_dataset/raw/external/quran_kashmiri/audio_inventory.csv"

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def probe_audio(filepath):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration,size,bit_rate:format_tags=artist,album,title,album_artist,date:stream=codec_name,sample_rate,channels",
        "-of", "json",
        filepath
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        return None
    try:
        return json.loads(res.stdout)
    except:
        return None

def parse_surah_from_filename(fname):
    # e.g. "001 - Sūra Al-Fātihah - ( The Opener ) - سورة الفاتحة.mp3"
    base = os.path.basename(fname)
    parts = base.split(" - ")
    try:
        surah_num = int(parts[0])
    except:
        surah_num = None
    return surah_num

def main():
    with open(os.path.join(METADATA_DIR, "surahs.json"), "r", encoding="utf-8") as f:
        surahs_meta = json.load(f)
    surah_lookup = {s["surah_id"]: s for s in surahs_meta}

    mp3_files = sorted(glob.glob(os.path.join(RAW_AUDIO_DIR, "*.mp3")))
    print(f"Found {len(mp3_files)} MP3 files to inventory.")

    records = []
    seen_hashes = {}
    corrupted_files = []

    for fpath in mp3_files:
        fname = os.path.basename(fpath)
        surah_num = parse_surah_from_filename(fname)
        parts = fname.split(" - ")
        smeta = surah_lookup.get(surah_num, {})

        sha256 = compute_sha256(fpath)
        file_size = os.path.getsize(fpath)

        if sha256 in seen_hashes:
            is_dup = True
            dup_of = seen_hashes[sha256]
        else:
            is_dup = False
            dup_of = None
            seen_hashes[sha256] = fname

        pdata = probe_audio(fpath)
        if not pdata or "format" not in pdata:
            corrupted_files.append(fname)
            records.append({
                "filename": fname,
                "surah_number": surah_num,
                "is_corrupt": True,
                "file_size_bytes": file_size,
                "sha256": sha256
            })
            continue

        fmt = pdata.get("format", {})
        streams = pdata.get("streams", [])
        audio_stream = next((s for s in streams if s.get("codec_name") == "mp3"), streams[0] if streams else {})

        duration = float(fmt.get("duration", 0.0))
        bit_rate = int(fmt.get("bit_rate", 0))
        tags = fmt.get("tags", {})
        codec = audio_stream.get("codec_name", "mp3")
        sample_rate = int(audio_stream.get("sample_rate", 44100))
        channels = int(audio_stream.get("channels", 2))

        rec = {
            "surah_number": surah_num,
            "surah_name_en": smeta.get("name_en", parts[1] if len(parts) > 1 else ""),
            "surah_name_ar": smeta.get("name_ar", parts[-1].replace(".mp3", "") if len(parts) > 2 else ""),
            "surah_ayah_count": smeta.get("verses_count", 0),
            "surah_revelation_place": smeta.get("revelation_place", "unknown"),
            "filename": fname,
            "filepath": fpath,
            "source_url": f"https://archive.org/download/al-quran-with-kashmiri-koshur-translation-audio-mp3-hq/{fname}",
            "source_collection": "al-quran-with-kashmiri-koshur-translation-audio-mp3-hq",
            "file_size_bytes": file_size,
            "sha256": sha256,
            "duration_seconds": round(duration, 3),
            "duration_minutes": round(duration / 60.0, 2),
            "duration_hours": round(duration / 3600.0, 4),
            "codec": codec,
            "sample_rate": sample_rate,
            "channels": channels,
            "bit_rate_bps": bit_rate,
            "id3_artist": tags.get("artist", ""),
            "id3_album": tags.get("album", ""),
            "id3_title": tags.get("title", ""),
            "is_corrupt": False,
            "is_duplicate": is_dup,
            "duplicate_of": dup_of
        }
        records.append(rec)

    records.sort(key=lambda x: x.get("surah_number", 0))

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    if records:
        keys = list(records[0].keys())
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)

    total_duration_hours = sum(r.get("duration_hours", 0) for r in records if not r.get("is_corrupt"))
    total_size_gb = sum(r.get("file_size_bytes", 0) for r in records) / (1024**3)

    print("\nInventory Summary:")
    print(f"  Total Files Inventory: {len(records)}")
    print(f"  Valid Files: {len(records) - len(corrupted_files)}")
    print(f"  Corrupt Files: {len(corrupted_files)}")
    print(f"  Total Duration: {total_duration_hours:.2f} hours")
    print(f"  Total Size: {total_size_gb:.2f} GB")
    print(f"  Saved JSON to: {OUTPUT_JSON}")
    print(f"  Saved CSV to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
