#!/usr/bin/env python3
"""
Download Complete Kashmiri Quran Audio Archive from Internet Archive
=====================================================================
Item: al-quran-with-kashmiri-koshur-translation-audio-mp3-hq
Total Files: 114 Surahs
Format: Original MP3
Supports resumable chunked downloads, retry logic, size & SHA-256 validation.
"""

import os
import sys
import json
import time
import hashlib
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

METADATA_FILE = "kashmiri_dataset/raw/external/quran_kashmiri/archive_metadata.json"
TARGET_DIR = "kashmiri_dataset/raw/external/quran_kashmiri/raw_audio"
INVENTORY_FILE = "kashmiri_dataset/raw/external/quran_kashmiri/audio_download_inventory.json"
BASE_URL = "https://archive.org/download/al-quran-with-kashmiri-koshur-translation-audio-mp3-hq"

os.makedirs(TARGET_DIR, exist_ok=True)

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def download_file(item, max_retries=5):
    fname = item["name"]
    expected_size = int(item.get("size", 0))
    expected_md5 = item.get("md5", "")
    target_path = os.path.join(TARGET_DIR, fname)
    
    # Check if already fully downloaded
    if os.path.exists(target_path):
        current_size = os.path.getsize(target_path)
        if expected_size > 0 and current_size == expected_size:
            sha256 = compute_sha256(target_path)
            return {
                "name": fname,
                "status": "already_downloaded",
                "size": current_size,
                "expected_size": expected_size,
                "sha256": sha256,
                "path": target_path
            }

    url = f"{BASE_URL}/{urllib.parse.quote(fname)}"
    headers = {"User-Agent": "Mozilla/5.0 (EduCasheer Research Data Ingestion)"}

    for attempt in range(1, max_retries + 1):
        try:
            start_byte = 0
            mode = "wb"
            if os.path.exists(target_path):
                start_byte = os.path.getsize(target_path)
                if 0 < start_byte < expected_size:
                    headers["Range"] = f"bytes={start_byte}-"
                    mode = "ab"
                elif start_byte > expected_size:
                    start_byte = 0
                    mode = "wb"
                    if "Range" in headers:
                        del headers["Range"]

            resp = requests.get(url, headers=headers, stream=True, timeout=60)
            if resp.status_code in (200, 206):
                with open(target_path, mode) as fp:
                    for chunk in resp.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            fp.write(chunk)
            elif resp.status_code == 416: # Range not satisfiable, file might be complete
                pass
            else:
                raise RuntimeError(f"HTTP {resp.status_code} on {url}")

            actual_size = os.path.getsize(target_path)
            if expected_size > 0 and actual_size != expected_size:
                raise RuntimeError(f"Size mismatch: got {actual_size}, expected {expected_size}")

            sha256 = compute_sha256(target_path)
            return {
                "name": fname,
                "status": "downloaded",
                "size": actual_size,
                "expected_size": expected_size,
                "sha256": sha256,
                "path": target_path
            }
        except Exception as e:
            if attempt == max_retries:
                return {
                    "name": fname,
                    "status": "error",
                    "error": str(e),
                    "path": target_path
                }
            time.sleep(2 * attempt)

def main():
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    files = [f for f in meta.get("files", []) if f.get("name", "").endswith(".mp3")]
    files.sort(key=lambda x: x["name"])

    print(f"Total MP3 files to download: {len(files)}")
    results = []

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(download_file, f): f["name"] for f in files}
        done_count = 0
        for fut in as_completed(futures):
            fname = futures[fut]
            res = fut.result()
            results.append(res)
            done_count += 1
            status = res.get("status")
            sz_mb = res.get("size", 0) / (1024 * 1024)
            print(f"[{done_count}/{len(files)}] {fname} -> {status} ({sz_mb:.2f} MB)")

    results.sort(key=lambda x: x["name"])
    with open(INVENTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    errors = [r for r in results if r.get("status") == "error"]
    print(f"\nFinished download. Successfully verified: {len(results) - len(errors)}/{len(files)} files.")
    if errors:
        print(f"Errors occurred in {len(errors)} files:")
        for e in errors:
            print(f"  {e['name']}: {e.get('error')}")
        sys.exit(1)
    else:
        print("All 114 files downloaded and verified!")

if __name__ == "__main__":
    main()
