#!/usr/bin/env python3
"""
DATA Audit Step 1: Exhaustive File Discovery
EduCasheer Kashmiri AI Research Project
"""

import os
import sys
import time
import json
import pandas as pd
from datetime import datetime

def classify_extension(ext):
    ext = ext.lower().strip(".")
    if ext in ["wav", "mp3", "flac", "ogg", "m4a", "opus", "aac", "wma"]:
        return "audio"
    elif ext in ["txt", "tsv", "json", "jsonl", "csv", "xml", "yaml", "yml", "md", "rst"]:
        return "text_manifest"
    elif ext in ["parquet", "arrow", "feather"]:
        return "columnar_data"
    elif ext in ["pt", "pth", "bin", "safetensors", "onnx", "ckpt", "joblib"]:
        return "model_checkpoint"
    elif ext in ["db", "sqlite", "sqlite3"]:
        return "database"
    elif ext in ["tar", "gz", "zip", "rar", "7z", "bz2", "xz"]:
        return "archive"
    elif ext in ["pdf", "epub", "docx", "html"]:
        return "document"
    elif ext in ["py", "sh", "c", "cpp", "h", "rs"]:
        return "code"
    elif ext in ["png", "jpg", "jpeg", "svg", "webp"]:
        return "image"
    else:
        return "other"

def associate_dataset(rel_path):
    lower = rel_path.lower()
    if "bolbosh" in lower or "gaash" in lower:
        return "Gaash_Bolbosh"
    elif "indicvoices_r" in lower:
        return "IndicVoices_R"
    elif "indicvoices" in lower:
        return "IndicVoices_Baseline"
    elif "alfajr" in lower:
        return "Quran_AlFajr"
    elif "salafi" in lower:
        return "Quran_Salafi"
    elif "hakeem" in lower:
        return "Quran_Hakeem_Shafiq"
    elif "mirwaiz" in lower or "bayan" in lower:
        return "Quran_Mirwaiz"
    elif "dr_war" in lower or "dr war" in lower:
        return "Quran_Dr_War"
    elif "king_fahd" in lower:
        return "Quran_King_Fahd"
    elif "omarrran_conversational" in lower or "omarrran_conv" in lower:
        return "Omarrran_Conversational"
    elif "omarrran" in lower and "single" in lower:
        return "Omarrran_Single_Speaker"
    elif "omarrran" in lower:
        return "Omarrran_Archive"
    elif "koshur_org" in lower or "ciil" in lower or "kachru" in lower:
        return "Koshur_Org_Language_Courses"
    elif "wakhlu" in lower:
        return "Wakhlu_Learn_Kashmiri"
    elif "akashvani" in lower or "radio" in lower:
        return "Akashvani_Radio_Srinagar"
    elif "datacollect" in lower or "waaz" in lower:
        return "WAAZ_Community_Collection"
    elif "xkaab" in lower:
        return "Xkaab_ASR"
    elif "openslr" in lower or "slr122" in lower:
        return "OpenSLR122"
    elif "programindz" in lower:
        return "Programindz_Kashmiri"
    elif "whisper" in lower:
        return "Whisper_Kashmiri"
    elif "meta_omnilingual" in lower:
        return "Meta_Omnilingual"
    elif "koshur_pixel" in lower:
        return "Koshur_Pixel"
    elif "parallel" in lower or "translation" in lower:
        return "Parallel_Translation"
    elif "lexicon" in lower or "dict" in lower:
        return "Lexicon_Dictionary"
    elif "wikipedia" in lower:
        return "Wikipedia_Kashmiri"
    elif "codemix" in lower or "rasa" in lower:
        return "CodeMixed_RASA"
    else:
        return "General_Project"

def scan_directory(base_dir, label):
    records = []
    print(f"Scanning {label} at {base_dir}...")
    for root, dirs, files in os.walk(base_dir, followlinks=False):
        # Exclude git directories and virtualenvs to keep inventory clean
        if "/.git" in root or "/venv" in root or "/__pycache__" in root or "/.cache" in root:
            continue
        for f in files:
            abs_p = os.path.join(root, f)
            try:
                st = os.lstat(abs_p)
                is_sym = os.path.islink(abs_p)
                size_bytes = st.st_size if not is_sym else 0
                mtime_iso = datetime.fromtimestamp(st.st_mtime).isoformat()
                _, ext = os.path.splitext(f)
                rel_p = os.path.relpath(abs_p, base_dir)
                
                cat = classify_extension(ext)
                ds = associate_dataset(rel_p)

                records.append({
                    "scan_root": label,
                    "rel_path": rel_p,
                    "abs_path": abs_p,
                    "filename": f,
                    "extension": ext.lower(),
                    "category": cat,
                    "dataset_association": ds,
                    "size_bytes": size_bytes,
                    "is_symlink": is_sym,
                    "mtime": mtime_iso
                })
            except Exception as e:
                pass
    return records

def main():
    start_t = time.time()
    all_records = []
    
    # 1. Main Project Directory
    all_records.extend(scan_directory("/home/musa/Kashmiri Project", "MAIN_PROJECT"))
    
    # 2. WAAZ Platform Directory
    waaz_dir = "/home/musa/Desktop/Projects/DataCollection"
    if os.path.exists(waaz_dir):
        all_records.extend(scan_directory(waaz_dir, "WAAZ_COLLECTION_PLATFORM"))
        
    df = pd.DataFrame(all_records)
    print(f"Total physical files discovered: {len(df):,}")
    
    # Calculate physical vs logical disk usage
    total_physical_bytes = df[~df["is_symlink"]]["size_bytes"].sum()
    print(f"Total Physical Disk Consumption: {total_physical_bytes / (1024**3):.2f} GB")
    
    out_parquet = "DATA/01_Master_Inventory/physical_file_inventory.parquet"
    df.to_parquet(out_parquet, index=False)
    print(f"Wrote physical inventory parquet to {out_parquet}")

    # Aggregated summary
    cat_summary = df.groupby(["category", "extension"]).agg(
        file_count=("filename", "count"),
        total_size_mb=("size_bytes", lambda s: round(s.sum() / (1024*1024), 2))
    ).reset_index()
    
    cat_summary.to_csv("DATA/01_Master_Inventory/complete_data_inventory.csv", index=False)
    
    # Source / Dataset Summary
    ds_summary = df.groupby("dataset_association").agg(
        file_count=("filename", "count"),
        total_size_mb=("size_bytes", lambda s: round(s.sum() / (1024*1024), 2))
    ).reset_index().to_dict(orient="records")
    
    summary_obj = {
        "audit_timestamp": datetime.now().isoformat(),
        "total_files": len(df),
        "total_physical_size_bytes": int(total_physical_bytes),
        "total_physical_size_gb": round(total_physical_bytes / (1024**3), 2),
        "categories": df.groupby("category")["size_bytes"].agg(["count", lambda s: round(s.sum()/(1024**3), 2)]).rename(columns={"<lambda_0>": "size_gb"}).to_dict(orient="index"),
        "dataset_summaries": ds_summary,
        "elapsed_seconds": round(time.time() - start_t, 2)
    }
    
    with open("DATA/01_Master_Inventory/complete_data_inventory.json", "w", encoding="utf-8") as f:
        json.dump(summary_obj, f, indent=2)
    print("Wrote complete data inventory JSON summary.")

if __name__ == "__main__":
    main()
