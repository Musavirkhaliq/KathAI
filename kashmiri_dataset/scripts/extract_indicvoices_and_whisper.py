import os
import glob
import json
import io
import soundfile as sf
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path("/home/musa/Kashmiri Project")
RAW_IV = BASE_DIR / "kashmiri_dataset" / "raw" / "external" / "indicvoices_kashmiri" / "Kashmiri_data" / "kashmiri"
RAW_WHISPER = BASE_DIR / "kashmiri_dataset" / "raw" / "external" / "whisper_kashmiri"
RAW_OMAR_TTS = BASE_DIR / "kashmiri_dataset" / "external" / "omarrran_kashmiri_single_speaker_ttd_hf_dataset_2024" / "data"

OUT_AUDIO_BASE = BASE_DIR / "kashmiri_dataset" / "processed" / "audio"
MANIFEST_DIR = BASE_DIR / "kashmiri_dataset" / "manifests" / "source_manifests"

def compute_snr_and_stats(audio_data, sr):
    if len(audio_data) == 0:
        return 0.0, 0.0, False
    peak = np.max(np.abs(audio_data))
    is_clipped = peak >= 0.999
    energy = audio_data ** 2
    signal_energy = np.mean(energy[energy > np.mean(energy)]) if np.any(energy > np.mean(energy)) else 1e-9
    noise_energy = np.mean(energy[energy <= np.mean(energy)]) if np.any(energy <= np.mean(energy)) else 1e-9
    snr = 10 * np.log10(max(signal_energy / max(noise_energy, 1e-9), 1e-9))
    duration = len(audio_data) / sr
    return round(float(snr), 2), round(float(duration), 3), bool(is_clipped)

# 1. Omar Single Speaker TTS Extraction
print("--- Extracting Omar Single Speaker TTS ---")
omar_out_audio = OUT_AUDIO_BASE / "omarrran_tts"
omar_out_audio.mkdir(parents=True, exist_ok=True)
omar_manifest = MANIFEST_DIR / "manifest_omarrran_tts.jsonl"
omar_parquets = sorted(glob.glob(str(RAW_OMAR_TTS / "*.parquet")))
sample_idx = 0
omar_records = []

with open(omar_manifest, "w", encoding="utf-8") as f_out:
    for pq_file in omar_parquets:
        df = pd.read_parquet(pq_file)
        for _, row in df.iterrows():
            sample_idx += 1
            audio_field = row.get("audio")
            text = str(row.get("sentence", row.get("text", ""))).strip()
            if not audio_field or not text:
                continue
            audio_bytes = audio_field.get("bytes") if isinstance(audio_field, dict) else audio_field
            if not audio_bytes:
                continue
            try:
                data, sr = sf.read(io.BytesIO(audio_bytes))
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                snr, dur, clipped = compute_snr_and_stats(data, sr)
                fname = f"omar_tts_{sample_idx:06d}.wav"
                fpath = omar_out_audio / fname
                sf.write(str(fpath), data, sr, subtype="PCM_16")
                rec = {
                    "sample_id": f"omar_tts_{sample_idx:06d}",
                    "audio_path": str(fpath.relative_to(BASE_DIR)),
                    "text": text,
                    "duration": dur,
                    "sample_rate": sr,
                    "speaker_id": "omarrran_tts_spk01",
                    "speaker_gender": "male",
                    "speaker_type": "studio_native_single_speaker",
                    "snr": snr,
                    "is_clipped": clipped,
                    "dataset_source": "omarrran/kashmiri_single_speaker_ttd_hf_dataset_2024",
                    "license": "Open / Research"
                }
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                omar_records.append(rec)
            except Exception as e:
                pass
print(f"Omar TTS Complete: {len(omar_records):,} standardized WAV clips saved.")

# 2. Whisper Kashmiri Extraction
print("\n--- Extracting Whisper Kashmiri Dataset ---")
whisper_out_audio = OUT_AUDIO_BASE / "whisper_kashmiri"
whisper_out_audio.mkdir(parents=True, exist_ok=True)
whisper_manifest = MANIFEST_DIR / "manifest_whisper_kashmiri.jsonl"
whisper_parquets = sorted(glob.glob(str(RAW_WHISPER / "**" / "*.parquet"), recursive=True))
sample_idx = 0
whisper_records = []

with open(whisper_manifest, "w", encoding="utf-8") as f_out:
    for pq_file in whisper_parquets:
        df = pd.read_parquet(pq_file)
        for _, row in df.iterrows():
            sample_idx += 1
            audio_field = row.get("audio")
            text = str(row.get("text", "")).strip()
            if not audio_field or not text:
                continue
            audio_bytes = audio_field.get("bytes") if isinstance(audio_field, dict) else audio_field
            if not audio_bytes:
                continue
            try:
                data, sr = sf.read(io.BytesIO(audio_bytes))
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                snr, dur, clipped = compute_snr_and_stats(data, sr)
                fname = f"whisper_ks_{sample_idx:06d}.wav"
                fpath = whisper_out_audio / fname
                sf.write(str(fpath), data, sr, subtype="PCM_16")
                rec = {
                    "sample_id": f"whisper_ks_{sample_idx:06d}",
                    "audio_path": str(fpath.relative_to(BASE_DIR)),
                    "text": text,
                    "duration": dur,
                    "sample_rate": sr,
                    "speaker_id": "whisper_ks_eval_speaker",
                    "speaker_gender": "unknown",
                    "speaker_type": "eval_paired_speech",
                    "snr": snr,
                    "is_clipped": clipped,
                    "dataset_source": "muneebharoon/whisper-kashmiri",
                    "license": "Open / Research"
                }
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                whisper_records.append(rec)
            except Exception as e:
                pass
print(f"Whisper Kashmiri Complete: {len(whisper_records):,} standardized WAV clips saved.")

# 3. IndicVoices Kashmiri Extraction (First 5 Shards ~20,000 Clips to preserve disk I/O & memory)
print("\n--- Extracting IndicVoices Kashmiri Subset ---")
iv_out_audio = OUT_AUDIO_BASE / "indicvoices_kashmiri"
iv_out_audio.mkdir(parents=True, exist_ok=True)
iv_manifest = MANIFEST_DIR / "manifest_indicvoices_kashmiri.jsonl"
iv_parquets = sorted(glob.glob(str(RAW_IV / "*.parquet")))[:5] # Process 5 shards
sample_idx = 0
iv_records = []

with open(iv_manifest, "w", encoding="utf-8") as f_out:
    for pq_file in iv_parquets:
        print(f"  Reading shard: {os.path.basename(pq_file)}...")
        df = pd.read_parquet(pq_file)
        for _, row in df.iterrows():
            sample_idx += 1
            af = row.get("audio_filepath")
            text = str(row.get("normalized", row.get("text", ""))).strip()
            if not af or not text:
                continue
            audio_bytes = af.get("bytes") if isinstance(af, dict) else af
            if not audio_bytes:
                continue
            try:
                data, sr = sf.read(io.BytesIO(audio_bytes))
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                snr, dur, clipped = compute_snr_and_stats(data, sr)
                fname = f"iv_ks_{sample_idx:06d}.wav"
                fpath = iv_out_audio / fname
                sf.write(str(fpath), data, sr, subtype="PCM_16")
                rec = {
                    "sample_id": f"iv_ks_{sample_idx:06d}",
                    "audio_path": str(fpath.relative_to(BASE_DIR)),
                    "text": text,
                    "duration": dur,
                    "sample_rate": sr,
                    "speaker_id": str(row.get("speaker_id", "unknown")),
                    "speaker_gender": str(row.get("gender", "unknown")),
                    "speaker_region": str(row.get("district", "Kashmir")),
                    "speaker_type": "indicvoices_field_speaker",
                    "snr": snr,
                    "is_clipped": clipped,
                    "dataset_source": "humair025/KashmiriSpeech-IndicVoices",
                    "license": "CC-BY 4.0"
                }
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                iv_records.append(rec)
            except Exception as e:
                pass

print(f"IndicVoices Kashmiri extraction complete: {len(iv_records):,} standardized WAV clips saved.")
