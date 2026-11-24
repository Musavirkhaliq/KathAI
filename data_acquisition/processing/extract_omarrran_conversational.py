#!/usr/bin/env python3
import os
import io
import json
import hashlib
import pandas as pd
import soundfile as sf
import numpy as np

def compute_sha256(data_bytes):
    return hashlib.sha256(data_bytes).hexdigest()

out_raw_dir = "data_acquisition/raw/community/omarrran_conversational/audio_full"
out_clips_dir = "data_acquisition/raw/community/omarrran_conversational/audio_clips"
os.makedirs(out_raw_dir, exist_ok=True)
os.makedirs(out_clips_dir, exist_ok=True)

p_train = "/home/musa/.cache/huggingface/hub/datasets--Omarrran--kashmiri_tts_hf_dataset/snapshots/5249e3bfd5c311e695264026e6adf3231a10eb08/data/train-00000-of-00001.parquet"
p_test = "/home/musa/.cache/huggingface/hub/datasets--Omarrran--kashmiri_tts_hf_dataset/snapshots/5249e3bfd5c311e695264026e6adf3231a10eb08/data/test-00000-of-00001.parquet"

df_all = pd.concat([pd.read_parquet(p_train), pd.read_parquet(p_test)]).reset_index(drop=True)

manifest_records = []
total_clips = 0
total_dur = 0.0

for idx, row in df_all.iterrows():
    fn = row["file_name"].strip()
    clean_fn = "".join(c if c.isalnum() or c in "-_" else "_" for c in fn)
    aud_dict = row["audio"]
    raw_bytes = aud_dict["bytes"]
    
    # Save full recording
    full_path = os.path.join(out_raw_dir, f"{clean_fn}.wav")
    data, sr = sf.read(io.BytesIO(raw_bytes))
    if data.ndim > 1:
        data = data.mean(axis=1)
    sf.write(full_path, data, sr, subtype="PCM_16")
    
    # Infer speaker and gender
    spk_raw = fn.split("-")[0].strip()
    gender = "female" if any(g in spk_raw.lower() for g in ["sehrish", "fatima", "iqra", "asma", "shazia", "female", "she", "misbah", "sabiya"]) else "male"
    
    # Parse utterances
    text_field = str(row["text"])
    lines = [l.strip() for l in text_field.split("\n") if l.strip()]
    
    for utt_idx, line in enumerate(lines):
        parts = line.split("\t")
        if len(parts) >= 3:
            try:
                start_sec = float(parts[0])
                end_sec = float(parts[1])
                utt_text = "\t".join(parts[2:]).strip()
            except ValueError:
                continue
        elif len(parts) == 2:
            try:
                start_sec = float(parts[0])
                utt_text = parts[1].strip()
                end_sec = start_sec + 3.0
            except ValueError:
                continue
        else:
            continue
            
        if end_sec <= start_sec or start_sec >= len(data)/sr:
            continue
            
        start_sample = int(start_sec * sr)
        end_sample = min(int(end_sec * sr), len(data))
        seg_data = data[start_sample:end_sample]
        
        seg_dur = len(seg_data) / sr
        if seg_dur < 0.3:
            continue
            
        clip_name = f"{clean_fn}_seg{utt_idx:04d}.wav"
        clip_path = os.path.join(out_clips_dir, clip_name)
        sf.write(clip_path, seg_data, sr, subtype="PCM_16")
        
        with open(clip_path, "rb") as cf:
            c_sha = hashlib.sha256(cf.read()).hexdigest()
            
        rec = {
            "sample_id": f"omarrran_conv_{idx}_{utt_idx:04d}",
            "source_dataset": "SRC_OMARRRAN_CONVERSATIONAL",
            "audio_path": clip_path,
            "full_audio_path": full_path,
            "duration_seconds": round(seg_dur, 3),
            "sample_rate": sr,
            "channels": 1,
            "speaker_id": f"omarrran_{spk_raw.split('_')[0].lower()}",
            "speaker_gender": gender,
            "language": "kas_Arab",
            "transcript_original": utt_text,
            "transcript_normalized": utt_text,
            "transcript_verified": True,
            "audio_sha256": c_sha,
            "quality_tier": "Tier A",
            "corpus_tier": "GOLD",
            "domain": "Conversational & Dialogue"
        }
        manifest_records.append(rec)
        total_clips += 1
        total_dur += seg_dur

manifest_out = "data_acquisition/raw/community/omarrran_conversational/manifest.jsonl"
with open(manifest_out, "w", encoding="utf-8") as f:
    for r in manifest_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Extracted {len(df_all)} full recordings and {total_clips} utterance clips.")
print(f"Total sliced utterance speech: {total_dur/3600.0:.2f} hours.")
print(f"Manifest written to {manifest_out}")
