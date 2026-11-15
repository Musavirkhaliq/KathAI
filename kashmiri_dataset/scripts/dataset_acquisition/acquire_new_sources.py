#!/usr/bin/env python3
"""
Workstream 9: Audio Acquisition & Standardization Pipeline
Extracts embedded audio from Meta Omnilingual ASR, Whisper Kashmiri 2,
and Sarvam Indic DiarBench, standardizes to 16kHz 16-bit mono PCM WAV,
and generates download_manifest.jsonl.
"""

import os
import io
import json
import glob
import hashlib
import numpy as np
import soundfile as sf
import pyarrow.parquet as pq

PROCESSED_AUDIO_ROOT = "kashmiri_dataset/processed/audio"
DOWNLOAD_MANIFEST_PATH = "download_manifest.jsonl"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def write_wav_16k_mono(audio_bytes, out_path, orig_sr_expected=None):
    """Decodes audio bytes and saves as 16kHz mono 16-bit PCM WAV."""
    ensure_dir(os.path.dirname(out_path))
    bio = io.BytesIO(audio_bytes)
    data, sr = sf.read(bio, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample if needed
    if sr != 16000:
        import librosa
        data_16k = librosa.resample(data, orig_sr=sr, target_sr=16000)
    else:
        data_16k = data

    data_int16 = (np.clip(data_16k, -1.0, 1.0) * 32767.0).astype(np.int16)
    sf.write(out_path, data_int16, 16000, subtype="PCM_16")

    # Hashes
    f_sha = hashlib.sha256()
    with open(out_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            f_sha.update(chunk)

    pcm_sha = hashlib.sha256(data_int16.tobytes()).hexdigest()
    dur = len(data_int16) / 16000.0

    return {
        "file_sha256": f_sha.hexdigest(),
        "pcm_sha256": pcm_sha,
        "duration": dur,
        "orig_sr": sr,
        "channels": 1,
    }

def process_omnilingual():
    print("Processing Meta Omnilingual ASR (kas_Arab)...")
    out_dir = os.path.join(PROCESSED_AUDIO_ROOT, "meta_omnilingual")
    ensure_dir(out_dir)

    omni_cache_files = sorted(glob.glob(
        "/home/musa/.cache/huggingface/hub/datasets--facebook--omnilingual-asr-corpus/snapshots/**/data/kas_Arab/*.parquet",
        recursive=True
    ))

    manifest_records = []
    clip_counter = 0

    for pfile in omni_cache_files:
        base = os.path.basename(pfile)
        split = "train" if "train" in base else ("dev" if "dev" in base else "test")
        tbl = pq.read_table(pfile)
        rows = tbl.to_pylist()
        print(f"  Extracting {len(rows)} clips from {base} (split={split})...")

        for idx, r in enumerate(rows):
            clip_counter += 1
            sid = f"omni_kas_{split}_{idx:04d}"
            out_wav = os.path.join(out_dir, f"{sid}.wav")
            audio_bytes = r["audio"]["bytes"]
            audio_stats = write_wav_16k_mono(audio_bytes, out_wav)

            rec = {
                "sample_id": sid,
                "source_dataset": "meta_omnilingual",
                "source_repository": "facebook/omnilingual-asr-corpus",
                "source_revision": "8648ba8946377697b427ae952076e49fc0e5e44d",
                "original_sample_id": r.get("segment_id") or f"{split}_{idx}",
                "original_audio_path": r["audio"].get("path") or f"kas_Arab/{split}_{idx}.wav",
                "canonical_audio_path": out_wav,
                "audio_sha256": audio_stats["file_sha256"],
                "pcm_sha256": audio_stats["pcm_sha256"],
                "duration_seconds": round(audio_stats["duration"], 3),
                "sample_rate": 16000,
                "channels": 1,
                "speaker_id": r.get("speaker_id", "unknown"),
                "speaker_metadata": {"iso_639_3": r.get("iso_639_3", "kas"), "glottocode": r.get("glottocode", "kash1277")},
                "language": "kas",
                "language_confidence": 1.0,
                "script": "perso_arabic_kashmiri",
                "transcript_original": r.get("raw_text", "").strip(),
                "prompt": r.get("prompt", "").strip(),
                "recording_domain": "spontaneous_prompted_dialogue",
                "license_id": "CC-BY-4.0",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "provenance": "Meta FAIR Omnilingual ASR, 2024",
                "original_split": split,
                "assigned_split": "train" if split == "train" else "evaluation_held_out",
                "eligibility_status": "training_eligible" if split == "train" else "evaluation_excluded",
            }
            manifest_records.append(rec)

    print(f"Meta Omnilingual done: {len(manifest_records)} clips extracted.")
    return manifest_records

def process_whisper_kashmiri_2():
    print("Processing Whisper Kashmiri Fine-tune Part 2...")
    out_dir = os.path.join(PROCESSED_AUDIO_ROOT, "whisper_kashmiri_2")
    ensure_dir(out_dir)

    manifest_records = []
    for split in ["train", "test"]:
        pfile = f"kashmiri_dataset/raw/external/whisper_kashmiri_2/data/{split}-00000-of-00001.parquet"
        if not os.path.exists(pfile):
            continue
        tbl = pq.read_table(pfile)
        rows = tbl.to_pylist()
        print(f"  Extracting {len(rows)} clips from {split}...")
        for idx, r in enumerate(rows):
            sid = f"whisper2_kas_{split}_{idx:04d}"
            out_wav = os.path.join(out_dir, f"{sid}.wav")
            audio_bytes = r["audio"]["bytes"]
            audio_stats = write_wav_16k_mono(audio_bytes, out_wav)

            rec = {
                "sample_id": sid,
                "source_dataset": "whisper_kashmiri_2",
                "source_repository": "muneebharoon/whisper-kashmiri-2",
                "source_revision": "initial_release",
                "original_sample_id": f"w2_{split}_{idx}",
                "original_audio_path": r["audio"].get("path") or f"{split}_{idx}.wav",
                "canonical_audio_path": out_wav,
                "audio_sha256": audio_stats["file_sha256"],
                "pcm_sha256": audio_stats["pcm_sha256"],
                "duration_seconds": round(audio_stats["duration"], 3),
                "sample_rate": 16000,
                "channels": 1,
                "speaker_id": "whisper_kashmiri_spk02",
                "speaker_metadata": {"domain": "literary_poetry_and_prose"},
                "language": "kas",
                "language_confidence": 1.0,
                "script": "perso_arabic_kashmiri",
                "transcript_original": r.get("text", "").strip(),
                "prompt": None,
                "recording_domain": "literary_poetry_and_prose",
                "license_id": "Apache-2.0",
                "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
                "provenance": "Muneeb Haroon Whisper Kashmiri Suite",
                "original_split": split,
                "assigned_split": "train" if split == "train" else "evaluation_held_out",
                "eligibility_status": "training_eligible" if split == "train" else "evaluation_excluded",
            }
            manifest_records.append(rec)

    print(f"Whisper Kashmiri 2 done: {len(manifest_records)} clips extracted.")
    return manifest_records

def process_sarvam_diarbench():
    print("Processing Sarvam Indic DiarBench (Strictly Evaluation Only)...")
    out_dir = os.path.join(PROCESSED_AUDIO_ROOT, "sarvam_diarbench")
    ensure_dir(out_dir)

    diar_cache = glob.glob(
        "/home/musa/.cache/huggingface/hub/datasets--sarvamai--indic-diarbench/snapshots/**/Kashmiri/*.parquet",
        recursive=True
    )
    manifest_records = []
    if not diar_cache:
        print("  Sarvam DiarBench parquet not found in cache.")
        return []

    tbl = pq.read_table(diar_cache[0])
    rows = tbl.to_pylist()
    print(f"  Extracting {len(rows)} recordings (strictly evaluation benchmark)...")
    for idx, r in enumerate(rows):
        rec_id = r.get("recording_id", f"rec_{idx}")
        sid = f"diarbench_kas_eval_{idx:03d}_{rec_id}"
        out_wav = os.path.join(out_dir, f"{sid}.wav")
        audio_bytes = r["audio"]["bytes"]
        audio_stats = write_wav_16k_mono(audio_bytes, out_wav)

        rec = {
            "sample_id": sid,
            "source_dataset": "sarvam_indic_diarbench",
            "source_repository": "sarvamai/indic-diarbench",
            "source_revision": "92877bad8aab6e598167d91c6ee02aa8ca6ede09",
            "original_sample_id": r.get("sample_id") or rec_id,
            "original_audio_path": r["audio"].get("path") or f"Kashmiri/{rec_id}.wav",
            "canonical_audio_path": out_wav,
            "audio_sha256": audio_stats["file_sha256"],
            "pcm_sha256": audio_stats["pcm_sha256"],
            "duration_seconds": round(audio_stats["duration"], 3),
            "sample_rate": 16000,
            "channels": 1,
            "speaker_id": f"multi_speaker_{r.get('num_speakers', 'multi')}",
            "speaker_metadata": {
                "num_speakers": r.get("num_speakers"),
                "num_segments": r.get("num_segments"),
                "dataset_type": r.get("dataset_type"),
            },
            "language": "kas",
            "language_confidence": 1.0,
            "script": "perso_arabic_annotated",
            "transcript_original": json.dumps(r.get("annotated_transcript", ""), ensure_ascii=False),
            "prompt": None,
            "recording_domain": "conversational_meeting_diarization",
            "license_id": "CC-BY-4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "provenance": "Sarvam AI Indic DiarBench, 2024",
            "original_split": "test",
            "assigned_split": "evaluation_benchmark_protected",
            "eligibility_status": "evaluation_excluded",  # STRICTLY EVALUATION ONLY!
        }
        manifest_records.append(rec)

    print(f"Sarvam DiarBench done: {len(manifest_records)} recordings extracted (EVALUATION ONLY).")
    return manifest_records

def main():
    omni_recs = process_omnilingual()
    w2_recs = process_whisper_kashmiri_2()
    diar_recs = process_sarvam_diarbench()

    all_downloads = omni_recs + w2_recs + diar_recs
    print(f"\nWriting total {len(all_downloads)} records to {DOWNLOAD_MANIFEST_PATH}...")
    with open(DOWNLOAD_MANIFEST_PATH, "w", encoding="utf-8") as f:
        for r in all_downloads:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Also archive raw copies to kashmiri_dataset/manifests/acquisition/download_manifest.jsonl
    archive_path = "kashmiri_dataset/manifests/acquisition/download_manifest.jsonl"
    with open(archive_path, "w", encoding="utf-8") as f:
        for r in all_downloads:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_hours = sum(r["duration_seconds"] for r in all_downloads) / 3600.0
    print(f"SUCCESS: {len(all_downloads)} clips written ({total_hours:.2f} total raw hours downloaded & standardized).")

if __name__ == "__main__":
    main()
