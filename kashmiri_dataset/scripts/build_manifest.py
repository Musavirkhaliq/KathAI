#!/usr/bin/env python3
"""
Master Manifest Builder for Kashmiri Speech & ASR Dataset
=========================================================
Generates standardized JSONL and CSV manifests containing:
  - sample_id
  - audio_path
  - text (raw)
  - normalized_text (Kashmiri script standardized)
  - duration (seconds)
  - sample_rate (16000)
  - speaker_id
  - speaker_gender
  - speaker_region
  - dialect
  - script
  - language
  - quality_score
  - snr
  - source
  - license
  - split (placeholder, assigned by create_splits.py)
"""

import os
import glob
import json
import csv
import io
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
from typing import Dict, List, Optional

from normalize_transcripts import normalize_kashmiri_text, detect_code_switching
from validate_audio import convert_to_wav16k, slice_audio, get_audio_info
from quality_filter import estimate_snr_and_quality

DATASET_ROOT = "/home/musa/Kashmiri Project"
PROCESSED_AUDIO = os.path.join(DATASET_ROOT, "kashmiri_dataset/processed/audio")
MANIFEST_DIR = os.path.join(DATASET_ROOT, "kashmiri_dataset/manifests")

os.makedirs(PROCESSED_AUDIO, exist_ok=True)
os.makedirs(MANIFEST_DIR, exist_ok=True)

def process_openslr122() -> List[Dict]:
    """Processes OpenSLR SLR122: slices continuous WAVs into 1,983 sentence segments."""
    records = []
    out_dir = os.path.join(PROCESSED_AUDIO, "openslr122")
    os.makedirs(out_dir, exist_ok=True)

    txt_files = sorted(glob.glob(f"{DATASET_ROOT}/Data/*/Kashmiri Project/01_Speech_Audio/01_OpenSLR_SLR122/kashmiri/*.txt"))
    wav_files = sorted(glob.glob(f"{DATASET_ROOT}/Data/*/Kashmiri Project/01_Speech_Audio/01_OpenSLR_SLR122/kashmiri/*.wav"))
    
    # Map base name without extension
    wav_map = {os.path.splitext(os.path.basename(w))[0].strip(): w for w in wav_files}

    for tf in txt_files:
        tbase = os.path.splitext(os.path.basename(tf))[0].strip()
        if 'txt_links' in tbase:
            continue
        # Find matching wav
        matching_wav = None
        for wbase, wpath in wav_map.items():
            if wbase.lower() == tbase.lower() or wbase.split(' - ')[0] == tbase.split(' - ')[0]:
                matching_wav = wpath
                break
        if not matching_wav:
            continue

        # Extract speaker id from filename (e.g. "bilal_192303_18_01_2022 - Danish Ahmad")
        parts = tbase.split(' - ')
        speaker_id = parts[0].strip()
        region = "Kashmir"
        if "shopian" in tbase.lower():
            region = "Shopian"
        elif "srinagar" in tbase.lower():
            region = "Srinagar"
        elif "anantnag" in tbase.lower():
            region = "Anantnag"

        with open(tf, 'r', encoding='utf-8', errors='ignore') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                cols = line.split('\t')
                if len(cols) < 3:
                    continue
                try:
                    start_sec = float(cols[0])
                    end_sec = float(cols[1])
                    raw_text = cols[2].strip()
                except ValueError:
                    continue

                if end_sec <= start_sec or (end_sec - start_sec) < 0.2:
                    continue

                sample_id = f"slr122_{speaker_id[:15]}_{idx:04d}".replace(" ", "_").replace(".", "_")
                clip_filename = f"{sample_id}.wav"
                clip_path = os.path.join(out_dir, clip_filename)

                # Slice audio
                ok = slice_audio(matching_wav, clip_path, start_sec, end_sec)
                if not ok or not os.path.exists(clip_path):
                    continue

                # Normalization & Quality
                norm_text = normalize_kashmiri_text(raw_text)
                q_info = estimate_snr_and_quality(clip_path)

                rel_audio_path = os.path.relpath(clip_path, DATASET_ROOT)
                records.append({
                    'sample_id': sample_id,
                    'audio_path': rel_audio_path,
                    'text': raw_text,
                    'normalized_text': norm_text,
                    'duration': q_info['duration'],
                    'sample_rate': 16000,
                    'speaker_id': speaker_id,
                    'speaker_gender': 'unknown',
                    'speaker_region': region,
                    'dialect': 'standard_central',
                    'script': 'perso_arabic_kashmiri',
                    'language': 'kas',
                    'quality_score': q_info['quality_score'],
                    'snr': q_info['snr_db'],
                    'flags': ";".join(q_info['flags']),
                    'source': 'openslr_slr122',
                    'license': 'OpenSLR_Public',
                    'split': 'train'
                })

    print(f"OpenSLR SLR122: Processed {len(records)} segments.")
    return records

def process_ktts() -> List[Dict]:
    """Processes KTTS single-speaker dataset: converts WebM/Opus to 16kHz PCM WAV."""
    records = []
    out_dir = os.path.join(PROCESSED_AUDIO, "ktts")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = glob.glob(f"{DATASET_ROOT}/Data/*/Kashmiri Project/01_Speech_Audio/03_KTTS/**/textcorpus-fRSMsx.csv", recursive=True)[0]
    df_text = pd.read_csv(csv_path)
    text_dict = dict(zip(df_text['id'].astype(str), df_text['sentence']))

    wav_files = glob.glob(f"{DATASET_ROOT}/Data/*/Kashmiri Project/01_Speech_Audio/03_KTTS/**/*.wav", recursive=True)

    for wf in wav_files:
        base = os.path.splitext(os.path.basename(wf))[0]
        if base not in text_dict:
            continue
        raw_text = str(text_dict[base]).strip()
        sample_id = f"ktts_{int(base):06d}"
        out_wav = os.path.join(out_dir, f"{sample_id}.wav")

        ok = convert_to_wav16k(wf, out_wav)
        if not ok or not os.path.exists(out_wav):
            continue

        norm_text = normalize_kashmiri_text(raw_text)
        q_info = estimate_snr_and_quality(out_wav)
        rel_path = os.path.relpath(out_wav, DATASET_ROOT)

        records.append({
            'sample_id': sample_id,
            'audio_path': rel_path,
            'text': raw_text,
            'normalized_text': norm_text,
            'duration': q_info['duration'],
            'sample_rate': 16000,
            'speaker_id': 'ktts_male_01',
            'speaker_gender': 'male',
            'speaker_region': 'Srinagar',
            'dialect': 'standard_central',
            'script': 'perso_arabic_kashmiri',
            'language': 'kas',
            'quality_score': q_info['quality_score'],
            'snr': q_info['snr_db'],
            'flags': ";".join(q_info['flags']),
            'source': 'mendeley_ktts',
            'license': 'CC-BY-NC-3.0',
            'split': 'train'
        })

    print(f"KTTS: Processed {len(records)} utterances.")
    return records

def process_kashmiri_audio_corpus() -> List[Dict]:
    """Processes programindz/kashmiri-audio-corpus parquet with embedded audio."""
    records = []
    out_dir = os.path.join(PROCESSED_AUDIO, "audio_corpus")
    os.makedirs(out_dir, exist_ok=True)

    pq_files = glob.glob(f"{DATASET_ROOT}/Data/*/Kashmiri Project/01_Speech_Audio/02_Kashmiri_Audio_Segmented/*.parquet")
    if not pq_files:
        return records

    table = pq.read_table(pq_files[0])
    df = table.to_pandas()

    for idx, row in df.iterrows():
        sample_id = f"kac_{idx:06d}"
        out_wav = os.path.join(out_dir, f"{sample_id}.wav")
        raw_text = str(row['text']).strip()

        # Audio binary bytes
        audio_dict = row['audio']
        if isinstance(audio_dict, dict) and 'bytes' in audio_dict and audio_dict['bytes']:
            raw_bytes = audio_dict['bytes']
            if not os.path.exists(out_wav) or os.path.getsize(out_wav) < 1024:
                try:
                    data, sr = sf.read(io.BytesIO(raw_bytes))
                    # Save as 16k mono
                    sf.write(out_wav, data, sr, subtype='PCM_16')
                    # Resample if needed
                    if sr != 16000:
                        convert_to_wav16k(out_wav, out_wav)
                except Exception:
                    continue

        if not os.path.exists(out_wav) or os.path.getsize(out_wav) == 0:
            continue

        norm_text = normalize_kashmiri_text(raw_text)
        q_info = estimate_snr_and_quality(out_wav)
        rel_path = os.path.relpath(out_wav, DATASET_ROOT)

        records.append({
            'sample_id': sample_id,
            'audio_path': rel_path,
            'text': raw_text,
            'normalized_text': norm_text,
            'duration': q_info['duration'],
            'sample_rate': 16000,
            'speaker_id': 'kac_spk_unknown',
            'speaker_gender': 'unknown',
            'speaker_region': 'Kashmir',
            'dialect': 'standard_central',
            'script': 'perso_arabic_kashmiri',
            'language': 'kas',
            'quality_score': q_info['quality_score'],
            'snr': q_info['snr_db'],
            'flags': ";".join(q_info['flags']),
            'source': 'programindz_audio_corpus',
            'license': 'GPL-3.0',
            'split': 'train'
        })

    print(f"Kashmiri Audio Corpus: Processed {len(records)} records.")
    return records

def process_xkaab_asr_4hrs() -> List[Dict]:
    """Processes downloaded XKaab/ASR-Kashmiri_4hrs parquet."""
    records = []
    out_dir = os.path.join(PROCESSED_AUDIO, "xkaab_4hrs")
    os.makedirs(out_dir, exist_ok=True)

    pq_file = os.path.join(DATASET_ROOT, "kashmiri_dataset/raw/external/xkaab_asr_4hrs/data/valid-00000-of-00001.parquet")
    if not os.path.exists(pq_file):
        return records

    table = pq.read_table(pq_file)
    df = table.to_pandas()

    for idx, row in df.iterrows():
        sample_id = f"xkaab_{idx:06d}"
        out_wav = os.path.join(out_dir, f"{sample_id}.wav")
        raw_text = str(row['text'] if 'text' in row and row['text'] else row.get('verbatim', '')).strip()

        audio_dict = row.get('audio_filepath')
        if isinstance(audio_dict, dict) and 'bytes' in audio_dict and audio_dict['bytes']:
            if not os.path.exists(out_wav) or os.path.getsize(out_wav) < 1024:
                try:
                    data, sr = sf.read(io.BytesIO(audio_dict['bytes']))
                    sf.write(out_wav, data, sr, subtype='PCM_16')
                    if sr != 16000:
                        convert_to_wav16k(out_wav, out_wav)
                except Exception:
                    continue

        if not os.path.exists(out_wav):
            continue

        norm_text = normalize_kashmiri_text(raw_text)
        q_info = estimate_snr_and_quality(out_wav)
        rel_path = os.path.relpath(out_wav, DATASET_ROOT)

        spk_id = str(row.get('speaker_id', 'xkaab_speaker')).strip()
        gender = str(row.get('gender', 'unknown')).strip().lower()
        district = str(row.get('district', 'Kashmir')).strip()

        records.append({
            'sample_id': sample_id,
            'audio_path': rel_path,
            'text': raw_text,
            'normalized_text': norm_text,
            'duration': q_info['duration'],
            'sample_rate': 16000,
            'speaker_id': spk_id,
            'speaker_gender': gender,
            'speaker_region': district,
            'dialect': 'standard_central',
            'script': 'perso_arabic_kashmiri',
            'language': 'kas',
            'quality_score': q_info['quality_score'],
            'snr': q_info['snr_db'],
            'flags': ";".join(q_info['flags']),
            'source': 'xkaab_asr_4hrs',
            'license': 'Open_Research',
            'split': 'train'
        })

    print(f"XKaab ASR 4hrs: Processed {len(records)} samples.")
    return records

def write_manifests(records: List[Dict]):
    """Saves master manifest in JSONL and CSV."""
    jsonl_path = os.path.join(MANIFEST_DIR, "manifest_master.jsonl")
    csv_path = os.path.join(MANIFEST_DIR, "manifest_master.csv")

    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False, encoding='utf-8')

    print(f"Master manifests written successfully!")
    print(f"  JSONL: {jsonl_path} ({len(records)} entries)")
    print(f"  CSV:   {csv_path} ({len(records)} entries)")

if __name__ == '__main__':
    all_records = []
    all_records.extend(process_openslr122())
    all_records.extend(process_ktts())
    all_records.extend(process_kashmiri_audio_corpus())
    all_records.extend(process_xkaab_asr_4hrs())
    write_manifests(all_records)
