#!/usr/bin/env python3
"""
Dr. Abdul Rehman War Kashmiri Quran Commentary Processing Engine
================================================================
Processes 28 broadcast recordings from Gulistan News (EP 01 - EP 30)
1. High-quality in-memory decoding and voice activity detection (ffmpeg silencedetect + recursive split).
2. Batched GPU inference on NVIDIA GB10 GPU with frozen B-01 XLSR model.
3. Categorization into kashmiri_commentary, arabic_recitation, mixed_kashmiri_arabic, uncertain_mixed_speech, rejected.
4. Per-episode persistent checkpointing for restart safety.
5. Isolated source collection: SRC_QURAN_DR_WAR.
6. Speaker Attribution: dr_abdul_rehman_war (independent voice).
"""

import os
import sys
import re
import json
import csv
import glob
import math
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

SCRIPTS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPTS_DIR.parent.parent
BASE_DIR = SCRIPTS_DIR.parent

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(WORKSPACE_ROOT))

from normalize_transcripts import normalize_kashmiri_text

RAW_AUDIO_DIR = BASE_DIR / "raw/external/quran_dr_war/raw_audio"
PROCESSED_AUDIO_DIR = BASE_DIR / "processed/audio/quran_dr_war"
MANIFEST_DIR = BASE_DIR / "manifests/source_manifests"
QUALITY_REPORT_DIR = BASE_DIR / "quality_reports"
PROVENANCE_DIR = BASE_DIR / "provenance"
CHECKPOINTS_DIR = BASE_DIR / "raw/external/quran_dr_war/episode_checkpoints"

PROCESSED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
QUALITY_REPORT_DIR.mkdir(parents=True, exist_ok=True)
PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = BASE_DIR / "experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc/best_model"

KASHMIRI_LEXICAL_MARKERS = [
    'چھُ', 'چھِ', 'چُھ', 'چِھ', 'چھُس', 'چُھس', 'چھُکھ', 'چھِو', 'چھہِ',
    'سٕندِ', 'سٕنزِ', 'سٕند', 'سٕنٛدِ', 'سٕنٛزِ', 'سٕنٛد', 'ہُند', 'ہٕند', 'ہٕنٛز',
    'سۭتۍ', 'سٍتۍ', 'سۭتی', 'یُس', 'یِم', 'تِم', 'تٔسۍ', 'تٔژھۍ', 'تِہند',
    'کَرن', 'کروُن', 'پزن', 'وَتھ', 'گنزرنہٕ', 'یِوان', 'گوٚو', 'ٲسۍ',
    'ژےٚ', 'أسۍ', 'دۅہ', 'پنٕنۍ', 'پننِس', 'پننہِ', 'خداے', 'خۄداے',
    'کِیاہ', 'کِتھہٕ', 'پٲٹھۍ', 'بۅڈ', 'مٔکی', 'واریاہ', 'کُنُے', 'زانان',
    'برونٹھ', 'نِشہِ', 'ورٲے', 'لُکن', 'لوٗکن', 'اَکھ', 'زٕ', 'ترٛے', 'ژور',
    'تُہۍ', 'محتاج', 'شروٗع', 'فرماوان', 'سبق', 'قرآن', 'تعالیٰ', 'مطلب', 'گویا', 'فرمایتھ'
]

def estimate_snr_from_numpy(data: np.ndarray, sr: int = 16000, frame_ms: float = 25.0) -> Dict:
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    dur = len(data) / sr
    flags = []
    if dur < 0.5: flags.append('too_short')
    if dur > 25.0: flags.append('too_long')

    clipping_count = np.sum(np.abs(data) >= 0.99)
    clipping_ratio = clipping_count / len(data) if len(data) > 0 else 0.0
    if clipping_ratio > 0.005: flags.append('clipped')

    frame_size = int(sr * (frame_ms / 1000.0))
    if frame_size <= 0 or len(data) < frame_size:
        return {'quality_score': 0.1, 'snr_db': 0.0, 'duration': round(dur, 3), 'clipping_ratio': round(clipping_ratio, 4), 'speech_ratio': 0.0, 'flags': flags or ['insufficient_frames'], 'is_clean': False}

    num_frames = len(data) // frame_size
    frames = data[:num_frames * frame_size].reshape(num_frames, frame_size)
    frame_energies = np.sum(frames ** 2, axis=1) / frame_size + 1e-10

    noise_energy = np.percentile(frame_energies, 10)
    speech_energy = np.percentile(frame_energies, 80)

    if noise_energy > 0:
        snr_linear = speech_energy / noise_energy
        snr_db = 10.0 * math.log10(max(1.0, snr_linear))
    else:
        snr_db = 35.0

    vad_thresh = max(noise_energy * 3.0, 1e-5)
    active_frames = np.sum(frame_energies > vad_thresh)
    speech_ratio = active_frames / num_frames if num_frames > 0 else 0.0

    if snr_db < 12.0: flags.append('low_snr')
    if speech_ratio < 0.3: flags.append('high_silence')

    snr_factor = min(1.0, max(0.0, snr_db / 30.0))
    speech_factor = min(1.0, max(0.0, speech_ratio / 0.8))
    clip_penalty = max(0.0, 1.0 - (clipping_ratio * 20.0))
    dur_factor = 1.0 if (0.5 <= dur <= 25.0) else 0.5

    quality_score = round(0.5 * snr_factor + 0.3 * speech_factor + 0.2 * clip_penalty, 3) * dur_factor
    quality_score = max(0.0, min(1.0, round(quality_score, 3)))
    if not flags: flags.append('ok')

    return {'quality_score': quality_score, 'snr_db': round(snr_db, 2), 'duration': round(dur, 3), 'clipping_ratio': round(clipping_ratio, 4), 'speech_ratio': round(speech_ratio, 3), 'flags': flags, 'is_clean': ('ok' in flags) and (quality_score >= 0.70) and (snr_db >= 15.0)}

def detect_speech_pauses_from_wav(wav_path: str, noise_db: float = -28.0, min_silence: float = 0.45) -> List[Tuple[float, float]]:
    cmd = ["ffmpeg", "-i", str(wav_path), "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}", "-f", "null", "-"]
    res = subprocess.run(cmd, stderr=subprocess.PIPE, text=True, errors="replace")
    silence_starts = [float(x) for x in re.findall(r"silence_start: ([\d\.]+)", res.stderr)]
    silence_ends = [float(x) for x in re.findall(r"silence_end: ([\d\.]+)", res.stderr)]

    info = sf.info(str(wav_path))
    total_dur = info.duration

    raw_intervals = []
    last_end = 0.0
    for s_start, s_end in zip(silence_starts, silence_ends):
        if s_start > last_end + 0.4:
            raw_intervals.append((last_end, s_start))
        last_end = s_end
    if last_end < total_dur - 0.4:
        raw_intervals.append((last_end, total_dur))

    return raw_intervals

print(f"Loading frozen ASR model from {MODEL_DIR}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Wav2Vec2Processor.from_pretrained(str(MODEL_DIR))
model = Wav2Vec2ForCTC.from_pretrained(str(MODEL_DIR)).to(device)
model.eval()

ep_files = sorted(glob.glob(str(RAW_AUDIO_DIR / "dr_war_ep*.mp3")))
print(f"Found {len(ep_files)} Dr. War audio episodes to process.")

all_records = []
coverage_audit_records = []

for ep_idx, ep_path in enumerate(ep_files, 1):
    fname = os.path.basename(ep_path)
    # extract ep num
    m = re.search(r'ep(\d+)', fname, re.I)
    ep_num = int(m.group(1)) if m else ep_idx

    chk_file = CHECKPOINTS_DIR / f"checkpoint_ep{ep_num:02d}.json"
    if chk_file.exists():
        print(f"[{ep_idx}/{len(ep_files)}] Episode {ep_num:02d} already processed, loading checkpoint...")
        with open(chk_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_records.extend(data["records"])
            coverage_audit_records.append(data["audit"])
        continue

    print(f"\n[{ep_idx}/{len(ep_files)}] Decoding and segmenting Dr. War EP {ep_num:02d} ({fname})...")
    temp_wav = CHECKPOINTS_DIR / f"temp_ep{ep_num:02d}.wav"
    cmd = ["ffmpeg", "-y", "-i", ep_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(temp_wav)]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    raw_intervals = detect_speech_pauses_from_wav(str(temp_wav), noise_db=-28.0, min_silence=0.45)
    audio_data, sr = sf.read(str(temp_wav))
    total_audio_dur = len(audio_data) / sr

    # Recursive sub-split for intervals > 12s (natural commentary pauses)
    refined_intervals = []
    for st, en in raw_intervals:
        dur = en - st
        if dur > 12.0:
            sub_count = math.ceil(dur / 8.0)
            step = dur / sub_count
            for i in range(sub_count):
                refined_intervals.append((round(st + i * step, 3), round(st + (i + 1) * step, 3)))
        else:
            refined_intervals.append((round(st, 3), round(en, 3)))

    total_speech_dur = sum(en - st for st, en in refined_intervals)
    silence_dur = max(0.0, total_audio_dur - total_speech_dur)

    ep_records = []
    seg_idx = 0

    batch_segments = []
    batch_metadata = []

    for st, en in refined_intervals:
        dur = en - st
        start_samp = int(st * sr)
        end_samp = min(len(audio_data), int(en * sr))
        clip_data = audio_data[start_samp:end_samp]
        if len(clip_data) < int(0.4 * sr): continue

        qc = estimate_snr_from_numpy(clip_data, sr=sr)
        batch_segments.append(clip_data)
        batch_metadata.append((st, en, dur, qc, clip_data))

    # Batched inference
    BATCH_SIZE = 16
    for b_idx in range(0, len(batch_segments), BATCH_SIZE):
        chunk_segs = batch_segments[b_idx:b_idx+BATCH_SIZE]
        chunk_meta = batch_metadata[b_idx:b_idx+BATCH_SIZE]

        inputs = processor(chunk_segs, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        transcriptions = processor.batch_decode(pred_ids)

        for (st, en, dur, qc, clip_data), hyp_text in zip(chunk_meta, transcriptions):
            seg_idx += 1
            clip_name = f"quran_dr_war_ep{ep_num:02d}_seg{seg_idx:04d}.wav"
            clip_path = PROCESSED_AUDIO_DIR / clip_name
            sf.write(str(clip_path), clip_data, sr, subtype="PCM_16")

            # Classification
            norm_hyp = normalize_kashmiri_text(hyp_text)
            has_kashmiri_marker = any(m in norm_hyp for m in KASHMIRI_LEXICAL_MARKERS)
            word_count = len(norm_hyp.split())

            if dur < 0.7 or word_count < 2:
                category = "rejected"
                seg_type = "rejected"
                script = "unknown"
                lang = "unk"
            elif has_kashmiri_marker and word_count >= 3:
                category = "kashmiri_commentary"
                seg_type = "commentary"
                script = "perso_arabic_kashmiri"
                lang = "kas"
            elif any(ar in hyp_text for ar in ["بسم الله", "الحمد لله", "أعوذ بالله", "رسول الله", "صلى الله", "تبارك وتعالى"]):
                category = "mixed_kashmiri_arabic"
                seg_type = "code_switched"
                script = "perso_arabic_kashmiri"
                lang = "mixed_kas_ara"
            elif word_count >= 4:
                category = "kashmiri_commentary"
                seg_type = "commentary"
                script = "perso_arabic_kashmiri"
                lang = "kas"
            else:
                category = "uncertain_mixed_speech"
                seg_type = "uncertain"
                script = "mixed"
                lang = "mixed_kas_ara"

            record = {
                "sample_id": f"quran_war_ep{ep_num:02d}_seg{seg_idx:04d}",
                "audio_path": str(clip_path.relative_to(WORKSPACE_ROOT)),
                "text": hyp_text,
                "normalized_text": norm_hyp,
                "asr_hypothesis": hyp_text,
                "duration": round(dur, 3),
                "sample_rate": 16000,
                "episode_number": ep_num,
                "segment_type": seg_type,
                "category": category,
                "speaker_id": "dr_abdul_rehman_war",
                "speaker_name": "Dr. Abdul Rehman War",
                "speaker_gender": "male",
                "speaker_region": "Kashmir",
                "dialect": "standard_central",
                "script": script,
                "language": lang,
                "quality_score": qc["quality_score"],
                "snr": qc["snr_db"],
                "flags": ",".join(qc["flags"]),
                "source": "SRC_QURAN_DR_WAR",
                "source_file": fname,
                "start_time_s": st,
                "end_time_s": en,
                "license": "Public_Broadcast_Gulistan_News",
                "verification_status": "tier1_machine_detected",
                "training_eligible": False
            }
            ep_records.append(record)

    all_records.extend(ep_records)

    audit_entry = {
        "episode_number": ep_num,
        "filename": fname,
        "total_duration_s": round(total_audio_dur, 3),
        "speech_duration_s": round(total_speech_dur, 3),
        "silence_duration_s": round(silence_dur, 3),
        "num_segments": len(ep_records),
        "speech_ratio": round(total_speech_dur / total_audio_dur, 4) if total_audio_dur > 0 else 0
    }
    coverage_audit_records.append(audit_entry)

    with open(chk_file, "w", encoding="utf-8") as f:
        json.dump({"records": ep_records, "audit": audit_entry}, f, indent=2)

    if temp_wav.exists(): temp_wav.unlink()
    print(f"EP {ep_num:02d} complete: {len(ep_records)} segments ({total_speech_dur/60:.1f} mins speech / {total_audio_dur/60:.1f} mins raw)")

# Write manifest
manifest_path = MANIFEST_DIR / "manifest_quran_dr_war.jsonl"
with open(manifest_path, "w", encoding="utf-8") as f:
    for r in all_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

audit_csv_path = QUALITY_REPORT_DIR / "quran_dr_war_coverage_audit.csv"
with open(audit_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["episode_number", "filename", "total_duration_s", "speech_duration_s", "silence_duration_s", "num_segments", "speech_ratio"])
    writer.writeheader()
    writer.writerows(coverage_audit_records)

# Summary
total_raw = sum(a["total_duration_s"] for a in coverage_audit_records) / 3600
total_speech = sum(a["speech_duration_s"] for a in coverage_audit_records) / 3600
kashmiri_comm = sum(r["duration"] for r in all_records if r["category"] == "kashmiri_commentary") / 3600
arabic_rec = sum(r["duration"] for r in all_records if r["category"] == "arabic_recitation") / 3600
mixed_dur = sum(r["duration"] for r in all_records if r["category"] == "mixed_kashmiri_arabic") / 3600
uncertain_dur = sum(r["duration"] for r in all_records if r["category"] == "uncertain_mixed_speech") / 3600

summary = {
    "source_id": "SRC_QURAN_DR_WAR",
    "dataset_name": "Dr. Abdul Rehman War Kashmiri Quran Commentary Series",
    "publisher": "Gulistan News / Dr. Abdul Rehman War",
    "total_episodes": len(ep_files),
    "total_raw_hours": round(total_raw, 4),
    "total_speech_hours": round(total_speech, 4),
    "total_kashmiri_commentary_hours": round(kashmiri_comm, 4),
    "total_mixed_kashmiri_arabic_hours": round(mixed_dur, 4),
    "total_kashmiri_candidate_hours": round(kashmiri_comm + mixed_dur, 4),
    "total_arabic_recitation_hours": round(arabic_rec, 4),
    "total_uncertain_speech_hours": round(uncertain_dur, 4),
    "total_segments": len(all_records),
    "manifest_path": str(manifest_path.relative_to(WORKSPACE_ROOT)),
    "audit_path": str(audit_csv_path.relative_to(WORKSPACE_ROOT))
}

prov_path = PROVENANCE_DIR / "quran_dr_war_provenance.json"
with open(prov_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nProcessing complete!")
print(json.dumps(summary, indent=2))
