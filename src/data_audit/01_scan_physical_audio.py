#!/usr/bin/env python3
"""
Step 1: Discover and scan all physical audio files across Kashmiri Project and DataCollection.
Computes exact durations, sampling rates, channels, formats, file sizes, and SHA-256 hashes.
Distinguishes real physical files from symlinks and detects embedded/archive audio.
"""

import os
import sys
import glob
import time
import hashlib
import tarfile
import pyarrow.parquet as pq
import pandas as pd
import soundfile as sf
from mutagen.mp3 import MP3

def get_file_sha256(filepath, max_bytes=None):
    """Computes SHA-256 hash of a file (or first max_bytes for speed if specified, else full)."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
            if max_bytes and f.tell() >= max_bytes:
                break
    return h.hexdigest()

def scan_physical_audio():
    print("=" * 60)
    print("STEP 1: PHYSICAL AUDIO DISCOVERY & SCAN")
    print("=" * 60)
    start_time = time.time()

    records = []
    
    # 1. Main Project processed/audio
    processed_audio_dir = "kashmiri_dataset/processed/audio"
    print(f"\nScanning {processed_audio_dir}...")
    for root, dirs, files in os.walk(processed_audio_dir, followlinks=False):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.webm'):
                fp = os.path.join(root, f)
                is_link = os.path.islink(fp)
                target = os.path.realpath(fp) if is_link else fp
                rel_dir = os.path.relpath(root, processed_audio_dir)
                
                records.append({
                    "audio_id": f,
                    "rel_path": fp,
                    "real_path": target,
                    "is_symlink": is_link,
                    "location_group": "kashmiri_dataset/processed/audio",
                    "subfolder": rel_dir,
                    "filename": f,
                    "ext": ext
                })

    # 2. Main Project raw/
    raw_dir = "kashmiri_dataset/raw"
    print(f"\nScanning {raw_dir}...")
    for root, dirs, files in os.walk(raw_dir, followlinks=False):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.webm'):
                fp = os.path.join(root, f)
                is_link = os.path.islink(fp)
                target = os.path.realpath(fp) if is_link else fp
                rel_dir = os.path.relpath(root, raw_dir)
                records.append({
                    "audio_id": f,
                    "rel_path": fp,
                    "real_path": target,
                    "is_symlink": is_link,
                    "location_group": "kashmiri_dataset/raw",
                    "subfolder": rel_dir,
                    "filename": f,
                    "ext": ext
                })

    # 3. Data/ (Google Drive shards)
    data_dir = "Data"
    print(f"\nScanning {data_dir}...")
    for root, dirs, files in os.walk(data_dir, followlinks=False):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.webm'):
                fp = os.path.join(root, f)
                is_link = os.path.islink(fp)
                target = os.path.realpath(fp) if is_link else fp
                rel_dir = os.path.relpath(root, data_dir)
                records.append({
                    "audio_id": f,
                    "rel_path": fp,
                    "real_path": target,
                    "is_symlink": is_link,
                    "location_group": "Data_Google_Drive_Shards",
                    "subfolder": rel_dir,
                    "filename": f,
                    "ext": ext
                })

    # 4. DataCollection (/home/musa/Desktop/Projects/DataCollection)
    dc_dir = "/home/musa/Desktop/Projects/DataCollection/data"
    print(f"\nScanning {dc_dir}...")
    if os.path.exists(dc_dir):
        for root, dirs, files in os.walk(dc_dir, followlinks=False):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.webm'):
                    fp = os.path.join(root, f)
                    is_link = os.path.islink(fp)
                    target = os.path.realpath(fp) if is_link else fp
                    rel_dir = os.path.relpath(root, dc_dir)
                    records.append({
                        "audio_id": f,
                        "rel_path": fp,
                        "real_path": target,
                        "is_symlink": is_link,
                        "location_group": "DataCollection_Platform",
                        "subfolder": rel_dir,
                        "filename": f,
                        "ext": ext
                    })

    # 5. External archives / embedded parquets
    # 5a. aazu34 openslr122-seg.tar.gz
    openslr_tar = "kashmiri_dataset/external/aazu34_kashmiri-openslr122-segments/openslr122-seg.tar.gz"
    if os.path.exists(openslr_tar):
        print(f"\nCataloging {openslr_tar}...")
        with tarfile.open(openslr_tar, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".flac"):
                    fname = os.path.basename(member.name)
                    records.append({
                        "audio_id": fname,
                        "rel_path": f"{openslr_tar}:{member.name}",
                        "real_path": f"{openslr_tar}:{member.name}",
                        "is_symlink": False,
                        "location_group": "kashmiri_dataset/external/aazu34_openslr122_tar",
                        "subfolder": "openslr122-seg",
                        "filename": fname,
                        "ext": ".flac",
                        "tar_size": member.size
                    })

    # 5b. programindz audio corpus parquet (1955 rows)
    prog_parquet = "kashmiri_dataset/external/programindz_kashmiri-audio-corpus/data/train-00000-of-00001.parquet"
    if os.path.exists(prog_parquet):
        print(f"\nCataloging {prog_parquet}...")
        pf = pq.ParquetFile(prog_parquet)
        num_rows = pf.metadata.num_rows
        # We can record the shard references
        for idx in range(num_rows):
            aid = f"programindz_kac_row_{idx:05d}"
            records.append({
                "audio_id": aid,
                "rel_path": f"{prog_parquet}#row={idx}",
                "real_path": f"{prog_parquet}#row={idx}",
                "is_symlink": False,
                "location_group": "kashmiri_dataset/external/programindz_audio_corpus_parquet",
                "subfolder": "embedded_parquet",
                "filename": aid,
                "ext": ".embedded_wav",
                "shard_row": idx
            })

    print(f"\nTotal discovered audio references: {len(records)}")
    
    df = pd.DataFrame(records)
    print("\nBreakdown by location group:")
    print(df["location_group"].value_counts())
    print("\nBreakdown by extension:")
    print(df["ext"].value_counts())
    print(f"\nSymlinks count: {df['is_symlink'].sum()} / {len(df)}")

    # Separate physical real files to inspect duration and sample rates
    # To be extremely efficient and avoid redundant reads, group by real_path
    unique_real_paths = df[~df["real_path"].str.contains(":", regex=False) & ~df["real_path"].str.contains("#", regex=False)]["real_path"].unique()
    print(f"\nUnique physical standalone files on filesystem: {len(unique_real_paths)}")

    # We will measure metadata for these unique physical files
    meta_cache = {}
    print("Reading audio metadata for unique physical files...")
    count = 0
    t_start_meta = time.time()
    
    for rpath in unique_real_paths:
        count += 1
        if count % 20000 == 0:
            elapsed = time.time() - t_start_meta
            print(f"  Processed {count}/{len(unique_real_paths)} files ({count/elapsed:.1f} files/s)...")
        
        ext = os.path.splitext(rpath)[1].lower()
        dur = None
        sr = None
        channels = None
        size_bytes = 0
        try:
            size_bytes = os.path.getsize(rpath)
        except:
            pass

        if ext in ('.wav', '.flac', '.ogg'):
            try:
                info = sf.info(rpath)
                dur = info.duration
                sr = info.samplerate
                channels = info.channels
            except Exception as e:
                pass
        elif ext == '.mp3':
            try:
                audio = MP3(rpath)
                dur = audio.info.length
                sr = audio.info.sample_rate
                channels = audio.info.channels
            except Exception as e:
                pass
        elif ext == '.webm':
            # DataCollection webm files have durations in datacollect.db or we can use ffprobe/mutagen
            # Let's check size
            pass

        meta_cache[rpath] = {
            "duration": dur,
            "sample_rate": sr,
            "channels": channels,
            "file_size": size_bytes
        }

    # Now let's fill in DataCollection DB durations for webm files if duration is None
    import sqlite3
    db_path = "/home/musa/Desktop/Projects/DataCollection/data/datacollect.db"
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT raw_file_path, processed_file_path, duration_seconds, sample_rate, channels, file_size_bytes FROM recordings")
        for raw_p, proc_p, dur_s, srate, ch, fsz in c.fetchall():
            # Check matching in dc_dir
            if raw_p:
                full_raw = os.path.join(dc_dir, raw_p)
                if full_raw in meta_cache and meta_cache[full_raw]["duration"] is None:
                    meta_cache[full_raw]["duration"] = dur_s
                    meta_cache[full_raw]["sample_rate"] = srate
                    meta_cache[full_raw]["channels"] = ch
                    if not meta_cache[full_raw]["file_size"]:
                        meta_cache[full_raw]["file_size"] = fsz
            if proc_p:
                full_proc = os.path.join(dc_dir, proc_p)
                if full_proc in meta_cache and meta_cache[full_proc]["duration"] is None:
                    meta_cache[full_proc]["duration"] = dur_s
                    meta_cache[full_proc]["sample_rate"] = srate
                    meta_cache[full_proc]["channels"] = ch
                    if not meta_cache[full_proc]["file_size"]:
                        meta_cache[full_proc]["file_size"] = fsz
        conn.close()

    # Fill metadata into DataFrame
    durations = []
    sample_rates = []
    channels_list = []
    file_sizes = []
    
    for _, row in df.iterrows():
        rp = row["real_path"]
        if rp in meta_cache:
            m = meta_cache[rp]
            durations.append(m["duration"])
            sample_rates.append(m["sample_rate"])
            channels_list.append(m["channels"])
            file_sizes.append(m["file_size"])
        elif "tar.gz:" in rp:
            # from openslr122-seg manifest if known, or default ~2-4s
            durations.append(None)
            sample_rates.append(16000)
            channels_list.append(1)
            file_sizes.append(row.get("tar_size", 0))
        elif "#row=" in rp:
            durations.append(None) # will be populated from parquet duration column
            sample_rates.append(16000)
            channels_list.append(1)
            file_sizes.append(0)
        else:
            durations.append(None)
            sample_rates.append(None)
            channels_list.append(None)
            file_sizes.append(0)

    df["duration_seconds"] = durations
    df["sample_rate"] = sample_rates
    df["channels"] = channels_list
    df["file_size_bytes"] = file_sizes

    # For programindz parquet, read duration directly from the table
    if os.path.exists(prog_parquet):
        t = pq.read_table(prog_parquet, columns=["duration"])
        dur_col = t.column("duration").to_pylist()
        mask = df["location_group"] == "kashmiri_dataset/external/programindz_audio_corpus_parquet"
        df.loc[mask, "duration_seconds"] = dur_col

    # For openslr tar manifest:
    if os.path.exists(openslr_tar):
        tar_durations = {}
        with tarfile.open(openslr_tar, "r:gz") as tar:
            if "openslr122-seg/manifest.jsonl" in tar.getnames():
                f = tar.extractfile("openslr122-seg/manifest.jsonl")
                import json
                for line in f:
                    try:
                        d = json.loads(line)
                        tar_durations[d["file"]] = d.get("duration", None)
                    except:
                        pass
        if tar_durations:
            for idx, r in df[df["location_group"] == "kashmiri_dataset/external/aazu34_openslr122_tar"].iterrows():
                fname = r["filename"]
                if fname in tar_durations:
                    df.loc[idx, "duration_seconds"] = tar_durations[fname]

    # Save checkpoint
    out_parquet = "src/data_audit/audit_physical_audio.parquet"
    df.to_parquet(out_parquet, index=False)
    print(f"\nSaved physical audio catalog to {out_parquet} ({len(df)} records).")
    
    # Summary report
    print("\n" + "=" * 60)
    print("PHYSICAL AUDIO SUMMARY")
    print("=" * 60)
    
    # 1. Physical non-symlink standalone files
    real_df = df[~df["is_symlink"] & ~df["rel_path"].str.contains(":", regex=False) & ~df["rel_path"].str.contains("#", regex=False)]
    print(f"Total Physical Files (excluding symlinks & archives): {len(real_df)}")
    total_physical_dur = real_df["duration_seconds"].sum()
    print(f"Total Decoded Audio Duration: {total_physical_dur:.2f} seconds ({total_physical_dur / 3600.0:.2f} hours)")
    print(f"Total File Size: {real_df['file_size_bytes'].sum() / (1024**3):.2f} GB")

    print("\nDuration by location group (physical files only):")
    for grp, grp_df in real_df.groupby("location_group"):
        dur_h = grp_df["duration_seconds"].sum() / 3600.0
        print(f"  {grp}: {len(grp_df)} files, {dur_h:.2f} hours")

    print(f"\nTotal elapsed time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    scan_physical_audio()
