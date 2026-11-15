#!/usr/bin/env python3
"""
Workstream 9: Audio Quality Control & Kashmiri Language Identification Pipeline
Calculates SNR, VAD speech ratio, clipping, quality tiers, Kashmiri language confidence,
and transcript normalization. Generates audio_quality_report.json.
"""

import os
import sys
import json
import re
import math
import multiprocessing
import numpy as np
import soundfile as sf
from collections import defaultdict

# Add existing scripts directory to import project utilities
PROJECT_ROOT = "/home/musa/Kashmiri Project"
sys.path.append(os.path.join(PROJECT_ROOT, "kashmiri_dataset/scripts"))
from quality_filter import estimate_snr_and_quality
from normalize_transcripts import normalize_kashmiri_text, detect_code_switching

KASHMIRI_LEXICAL_MARKERS = [
    "چھُ", "چھِ", "چھَ", "چھَس", "چھُس", "چھِو", "چھان",
    "ہُنٛد", "ہِنٛز", "ہٕنٛدی", "ہنٛد", "ہنٛز",
    "منٛز", "سۭتۍ", "سۭتی", "سٟتۍ", "تہٕ", "تہ",
    "سُنٛد", "سٕنٛز", "پٮ۪ٹھ", "پٮ۪ٹھے", "پیٹھ",
    "یِہ", "یہِ", "سُہ", "سۄ", "تِم", "تِمَن",
    "أسۍ", "اسۍ", "تۄہہِ", "توہی", "مےٚ", "میہ", "تَمِ",
    "کٔشیٖر", "کٔشیٖرِ", "کٲشُر", "کٲشِر", "کٲشرِ",
    "واریاہ", "کینٛہہ", "گۄڈٕ", "پَتہٕ", "وُنۍ", "دوٚپ", "کوٚر",
]

URDU_INTRUSION_MARKERS = [
    "ہے", "ہیں", "تھا", "تھی", "تھے", "رہا ہے", "رہی ہے", "رہے ہیں",
    "کو", "نے", "سے", "کا", "کی", "کے", "پر", "میں", "اور",
    "کرتا ہے", "کرتی ہے", "ہوتا ہے", "ہوتی ہے", "نہیں", "بہت اچھا"
]

def analyze_kashmiri_language(text):
    """
    Multi-signal Kashmiri language validator:
    1. Perso-Arabic script ratio
    2. Kashmiri specific character frequency (e.g., ٚ, ٕ, ٛ, ۍ, ؠ, ٲ, ۄ)
    3. Kashmiri lexical marker density
    4. Urdu / Hindi intrusion penalty
    """
    if not text or not text.strip():
        return {
            "language": "unknown",
            "language_confidence": 0.0,
            "script": "unknown",
            "kashmiri_marker_count": 0,
            "urdu_intrusion_count": 0,
            "is_valid_kashmiri": False,
        }

    # Perso-Arabic unicode check
    perso_arabic_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]', text))
    total_alpha = len(re.findall(r'\w', text))
    script_ratio = perso_arabic_chars / max(1, total_alpha)

    # Kashmiri specific characters
    ks_special_chars = len(re.findall(r'[\u06c4\u0656\u0655\u0670\u06d2\u06a9\u06af\u0686\u0698\u067e\u06c1\u065a\u065b\u06cd\u06ce\u0658]', text))

    # Kashmiri morphemes
    words = text.split()
    ks_markers = sum(1 for m in KASHMIRI_LEXICAL_MARKERS if m in text)

    # Urdu intrusion check
    urdu_intrusions = sum(1 for m in URDU_INTRUSION_MARKERS if re.search(r'\b' + re.escape(m) + r'\b', text))

    # Scoring
    score = 0.5  # Base prior for candidate Kashmiri corpora
    if script_ratio > 0.8:
        score += 0.2
    if ks_markers > 0:
        score += min(0.3, ks_markers * 0.1)
    if ks_special_chars > 0:
        score += min(0.1, ks_special_chars * 0.02)
    if urdu_intrusions > 0:
        score -= min(0.5, urdu_intrusions * 0.2)

    confidence = max(0.0, min(1.0, score))
    is_kashmiri = (confidence >= 0.65) and (script_ratio > 0.6)

    return {
        "language": "kas" if is_kashmiri else ("ur" if urdu_intrusions > ks_markers else "unknown"),
        "language_confidence": round(confidence, 3),
        "script": "perso_arabic_kashmiri" if script_ratio > 0.8 else "mixed_or_latin",
        "kashmiri_marker_count": ks_markers,
        "urdu_intrusion_count": urdu_intrusions,
        "is_valid_kashmiri": is_kashmiri,
    }

def process_single_clip_qa(item):
    """Processes audio and transcript quality metrics for a single clip."""
    audio_path = item["canonical_audio_path"]
    text = item.get("transcript_original", "")

    # 1. Audio SNR & VAD via existing project utility
    audio_qa = estimate_snr_and_quality(audio_path)

    # 2. Additional detailed acoustic metrics
    try:
        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        rms = float(np.sqrt(np.mean(data**2)))
        peak = float(np.max(np.abs(data)))
    except Exception:
        rms = 0.0
        peak = 0.0

    # 3. Transcript Normalization
    try:
        norm_text = normalize_kashmiri_text(text) if text else ""
    except Exception:
        norm_text = text

    # 4. Language Identification
    lang_info = analyze_kashmiri_language(text)

    # 5. Speech rate and alignment check
    dur = item["duration_seconds"]
    words = len(norm_text.split()) if norm_text else 0
    wps = words / max(0.1, dur)

    # Rate check: 0.5 to 5.5 words/s is typical natural speech
    if words == 0:
        align_status = "untranscribed"
    elif wps < 0.2:
        align_status = "potentially_long_silence"
    elif wps > 6.5:
        align_status = "potentially_fast_or_truncated"
    else:
        align_status = "verified_aligned"

    # 6. Quality Tier Classification
    snr = audio_qa.get("snr_db", 0.0)
    speech_ratio = audio_qa.get("speech_ratio", 0.0)
    clip_ratio = audio_qa.get("clipping_ratio", 0.0)

    if snr >= 15.0 and speech_ratio >= 0.65 and clip_ratio < 0.01 and lang_info["is_valid_kashmiri"]:
        tier = "Tier A"  # High-confidence, clean studio / field speech
    elif snr >= 9.0 and speech_ratio >= 0.45 and clip_ratio < 0.05 and lang_info["is_valid_kashmiri"]:
        tier = "Tier B"  # Usable natural speech with minor ambient noise
    elif lang_info["is_valid_kashmiri"] and snr >= 5.0:
        tier = "Tier C"  # Low SNR or high silence, review suggested
    else:
        tier = "Tier D"  # Rejected (non-Kashmiri, excessive noise, or decoding failure)

    rejection_reason = None
    if not lang_info["is_valid_kashmiri"]:
        rejection_reason = "language_verification_failed"
    elif dur < 0.5:
        rejection_reason = "audio_too_short"
    elif dur > 60.0:
        rejection_reason = "audio_exceeds_duration_limit"
    elif snr < 4.0:
        rejection_reason = "excessive_background_noise_snr_below_4db"

    eligibility = item.get("eligibility_status", "training_eligible")
    if tier == "Tier D" or rejection_reason is not None:
        eligibility = "rejected"
    elif item.get("assigned_split") == "evaluation_benchmark_protected" or item.get("assigned_split") == "evaluation_held_out":
        eligibility = "evaluation_excluded"

    res = dict(item)
    res["transcript_normalized"] = norm_text
    res["words"] = words
    res["words_per_second"] = round(wps, 2)
    res["snr_estimate_db"] = round(snr, 2)
    res["vad_speech_ratio"] = round(speech_ratio, 3)
    res["rms_energy"] = round(rms, 4)
    res["peak_amplitude"] = round(peak, 4)
    res["clipping_ratio"] = round(clip_ratio, 4)
    res["quality_score"] = round(audio_qa.get("quality_score", 0.0), 3)
    res["quality_tier"] = tier
    res["language"] = lang_info["language"]
    res["language_confidence"] = lang_info["language_confidence"]
    res["script"] = lang_info["script"]
    res["alignment_status"] = align_status
    res["eligibility_status"] = eligibility
    res["rejection_reason"] = rejection_reason
    res["audio_qa_flags"] = audio_qa.get("flags", [])

    return res

def main(num_workers=16):
    manifest_in = "download_manifest.jsonl"
    print(f"Loading downloaded clips from {manifest_in}...")
    with open(manifest_in, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]

    print(f"Running Audio QA & Language ID on {len(items)} clips with {num_workers} processes...")
    with multiprocessing.Pool(num_workers) as pool:
        qa_results = pool.map(process_single_clip_qa, items, chunksize=50)

    # Save enriched manifest
    qa_manifest_out = "kashmiri_dataset/manifests/quality_control/manifest_acquired_qa.jsonl"
    with open(qa_manifest_out, "w", encoding="utf-8") as f:
        for r in qa_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved QA enriched manifest to {qa_manifest_out}")

    # Generate audio_quality_report.json
    tier_counts = defaultdict(int)
    tier_hours = defaultdict(float)
    lang_counts = defaultdict(int)
    eligibility_counts = defaultdict(int)
    eligibility_hours = defaultdict(float)
    domain_stats = defaultdict(lambda: {"clips": 0, "hours": 0.0})

    snrs = []
    vads = []
    durations = []
    word_counts = []

    for r in qa_results:
        t = r["quality_tier"]
        tier_counts[t] += 1
        dur = r["duration_seconds"]
        tier_hours[t] += dur / 3600.0
        lang_counts[r["language"]] += 1
        el = r["eligibility_status"]
        eligibility_counts[el] += 1
        eligibility_hours[el] += dur / 3600.0

        dom = r.get("recording_domain", "other")
        domain_stats[dom]["clips"] += 1
        domain_stats[dom]["hours"] += dur / 3600.0

        snrs.append(r["snr_estimate_db"])
        vads.append(r["vad_speech_ratio"])
        durations.append(dur)
        word_counts.append(r["words"])

    qa_report = {
        "report_timestamp": "2026-11-11T16:55:00+05:30",
        "total_acquired_clips": len(qa_results),
        "total_acquired_hours": round(sum(durations) / 3600.0, 2),
        "quality_tier_distribution": {
            t: {"clips": tier_counts[t], "hours": round(tier_hours[t], 2)}
            for t in ["Tier A", "Tier B", "Tier C", "Tier D"]
        },
        "language_identification": {
            lang: count for lang, count in lang_counts.items()
        },
        "eligibility_breakdown": {
            el: {"clips": eligibility_counts[el], "hours": round(eligibility_hours[el], 2)}
            for el in eligibility_counts
        },
        "acoustic_distributions": {
            "duration_seconds": {
                "mean": round(float(np.mean(durations)), 2),
                "std": round(float(np.std(durations)), 2),
                "min": round(float(np.min(durations)), 2),
                "max": round(float(np.max(durations)), 2),
                "median": round(float(np.median(durations)), 2),
            },
            "snr_db": {
                "mean": round(float(np.mean(snrs)), 2),
                "min": round(float(np.min(snrs)), 2),
                "max": round(float(np.max(snrs)), 2),
                "median": round(float(np.median(snrs)), 2),
            },
            "voice_activity_ratio": {
                "mean": round(float(np.mean(vads)), 3),
                "min": round(float(np.min(vads)), 3),
                "max": round(float(np.max(vads)), 3),
            },
            "words_per_utterance": {
                "mean": round(float(np.mean(word_counts)), 1),
                "total_words": int(sum(word_counts)),
            }
        },
        "recording_domains": {
            dom: {"clips": s["clips"], "hours": round(s["hours"], 2)}
            for dom, s in domain_stats.items()
        }
    }

    report_out = "audio_quality_report.json"
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(qa_report, f, indent=2, ensure_ascii=False)
    print(f"Saved {report_out}")

    print("\n=== AUDIO QUALITY CONTROL SUMMARY ===")
    print(f"Total Acquired: {len(qa_results)} clips, {qa_report['total_acquired_hours']} hrs")
    print("Quality Tiers:")
    for t, data in qa_report["quality_tier_distribution"].items():
        print(f"  {t:<10}: {data['clips']:>5} clips, {data['hours']:>6.2f} hrs")
    print("Eligibility:")
    for el, data in qa_report["eligibility_breakdown"].items():
        print(f"  {el:<32}: {data['clips']:>5} clips, {data['hours']:>6.2f} hrs")

if __name__ == "__main__":
    main()
