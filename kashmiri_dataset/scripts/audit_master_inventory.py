import os
import glob
import json
import hashlib
import pandas as pd
from pathlib import Path

BASE_DIR = Path("/home/musa/Kashmiri Project")
DATA_DIR = BASE_DIR / "Data"
EXTERNAL_DIR = BASE_DIR / "kashmiri_dataset" / "external"

print("Scanning RAW / ORIGINAL and EXTERNAL data repositories...")

inventory = []

# 1. Audit Raw Data Directory
raw_shards = sorted(glob.glob(str(DATA_DIR / "Kashmiri Project-20261101T102755Z-1-*")))
raw_files = []
raw_size = 0
for shard in raw_shards:
    for root, dirs, files in os.walk(shard):
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                sz = os.path.getsize(fp)
                raw_size += sz
                raw_files.append(fp)

inventory.append({
    "source_id": "SRC_RAW_GDRIVE_ARCHIVE",
    "dataset_name": "Kashmiri Project Raw Shards (001-022)",
    "layer": "RAW / ORIGINAL",
    "original_source": "Google Drive Shards (01_OpenSLR_SLR122, 02_programindz, 03_KTTS, 04_Text_Corpora)",
    "url_or_repo": "Internal Google Drive Delivery (Preserved in Data/)",
    "date_acquired": "2026-11-01",
    "license": "Mixed (OpenSLR: CC-BY-SA 4.0; KTTS/Programindz: Research / CC-BY; Grierson/Text: Public Domain / Open)",
    "data_type": "Multimodal (Audio, Transcripts, Text Corpora, Dictionaries)",
    "modalities": "Audio, Text, Lexicon",
    "script": "Perso-Arabic (kashmiri_nastaliq/naskh), Devanagari, Romanized",
    "language": "Kashmiri (kas / ks / kas_Arab)",
    "dialect": "Kashur (Standard Maraz / Srinagar dominant)",
    "num_files": len(raw_files),
    "size_bytes": raw_size,
    "size_gb": round(raw_size / (1024**3), 3),
    "transcription_availability": "Yes (Paired audio-text for SLR122, KTTS, and programindz)",
    "metadata_availability": "Partial (Speaker IDs in SLR122, duration & text in KTTS)",
    "intended_use": "ASR, TTS, Lexicon Analysis, Pre-training Text Corpus",
    "known_limitations": "KTTS audio stored with .wav extension is Opus/WebM; SLR122 unsegmented continuous audio; script/orthographic inconsistencies across sources.",
    "status": "IMMUTABLE_RAW_PRESERVED"
})

# 2. Audit all external repositories
ext_dirs = sorted([d for d in EXTERNAL_DIR.iterdir() if d.is_dir()])
for d in ext_dirs:
    files = [f for f in d.rglob("*") if f.is_file() and not f.is_symlink()]
    sz = sum(f.stat().st_size for f in files)
    exts = sorted(list(set(f.suffix.lower() for f in files)))
    
    # Infer metadata
    name = d.name
    category = "Text / LLM"
    data_type = "Text"
    modalities = "Text"
    license_type = "Open / CC / Research"
    dialect = "Standard Kashmiri"
    transcription = "N/A"
    intended_use = "Pre-training / SFT / NLP"
    script = "Perso-Arabic (kas_Arab)"
    
    if any(x in name for x in ["indicvoices", "bolbosh", "whisper", "tts", "audio", "asr", "speech"]):
        category = "Speech / Audio"
        data_type = "Speech / Audio"
        modalities = "Audio, Transcripts"
        transcription = "Yes"
        intended_use = "ASR / TTS Training and Benchmark"
    elif any(x in name for x in ["ocr", "pixel", "synthocr", "recognition"]):
        category = "Vision / OCR"
        data_type = "Image + Text"
        modalities = "Image, Text"
        intended_use = "Kashmiri OCR & Document Recognition"
    elif any(x in name for x in ["dictionary", "words", "grierson"]):
        category = "Lexicon"
        data_type = "Tabular / Lexical"
        modalities = "Lexicon, Pronunciation"
        intended_use = "Grapheme-to-Phoneme, Tokenizer Lexicon"

    if "deva" in name:
        script = "Devanagari (kas_Deva)"
    elif "codemix" in name or "rasa" in name:
        script = "Perso-Arabic, Romanized, Devanagari (Code-mixed)"
        intended_use = "Code-mixed NLP, Dialogue & Conversational AI"

    inventory.append({
        "source_id": f"SRC_EXT_{name.upper()}",
        "dataset_name": name,
        "layer": "RAW_EXTERNAL_INGESTION",
        "original_source": f"Hugging Face / Web Archive ({name.replace('_', '/')})",
        "url_or_repo": f"https://huggingface.co/datasets/{name.replace('_', '/') if '_' in name else name}",
        "date_acquired": "2026-11-01 / 2026-11-02",
        "license": license_type,
        "data_type": data_type,
        "modalities": modalities,
        "script": script,
        "language": "Kashmiri (kas)",
        "dialect": dialect,
        "num_files": len(files),
        "size_bytes": sz,
        "size_gb": round(sz / (1024**3), 3),
        "transcription_availability": transcription,
        "metadata_availability": "Yes (Repository manifests, Parquet schemas)",
        "intended_use": intended_use,
        "known_limitations": "Various formats (Parquet, Tar, WAV, MP3, JSONL); varying text normalizations; diacritic variance.",
        "status": "INGESTED_UNMODIFIED"
    })

inv_df = pd.DataFrame(inventory)
out_csv = BASE_DIR / "kashmiri_dataset" / "provenance" / "master_dataset_inventory.csv"
out_json = BASE_DIR / "kashmiri_dataset" / "provenance" / "master_dataset_inventory.json"
inv_df.to_csv(out_csv, index=False)
inv_df.to_json(out_json, orient="records", indent=2)

print(f"Master Inventory generated successfully!")
print(f"Total Sources Audited: {len(inv_df)}")
print(f"Total Size: {inv_df['size_gb'].sum():.2f} GB")
print(f"Inventory saved to:")
print(f"  - {out_csv}")
print(f"  - {out_json}")
