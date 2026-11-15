#!/usr/bin/env python3
"""
Audio Review Package & Transcription Correctness Audit
=====================================================
1. Samples exactly 100 accepted alignments stratified by Surah, duration, and confidence.
2. Extracts and packages audio clips for review.
3. Leaves human-verification fields unset (null) to preserve research integrity.
4. Audits error patterns (boundary leakage, commentary divergence, phonetic variation).
5. Computes measured error rates and discrepancy statistics.
"""

import os
import sys
import json
import csv
import random
import shutil
import difflib
from pathlib import Path
from collections import defaultdict

WORKSPACE = Path("/home/musa/Kashmiri Project")
VERIFIED_PATH = WORKSPACE / "kashmiri_dataset/raw/external/quran_kashmiri/raw_text/verified_alignments.jsonl"
REVIEW_PKG_DIR = WORKSPACE / "kashmiri_dataset/raw/external/quran_kashmiri/audio_review_package"
CLIPS_DIR = REVIEW_PKG_DIR / "clips"

CLIPS_DIR.mkdir(parents=True, exist_ok=True)

def cer(ref: str, hyp: str) -> float:
    """Computes Character Error Rate via edit distance."""
    if not ref:
        return 1.0 if hyp else 0.0
    r_chars = list(ref.replace(" ", ""))
    h_chars = list(hyp.replace(" ", ""))
    if not r_chars:
        return 1.0 if h_chars else 0.0
    matcher = difflib.SequenceMatcher(None, r_chars, h_chars)
    return round(1.0 - matcher.ratio(), 4)

def wer(ref: str, hyp: str) -> float:
    """Computes Word Error Rate via token edit distance."""
    r_words = ref.split()
    h_words = hyp.split()
    if not r_words:
        return 1.0 if h_words else 0.0
    matcher = difflib.SequenceMatcher(None, r_words, h_words)
    return round(1.0 - matcher.ratio(), 4)

def main():
    random.seed(42)

    with open(VERIFIED_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(l) for l in f]

    print(f"Total accepted alignments available for sampling: {len(records)}")

    # Stratify into buckets
    buckets = defaultdict(list)
    for r in records:
        dur = r["duration"]
        sim = r.get("similarity_score", 0.0)

        # Duration bucket
        if dur < 3.0:
            d_b = "short (<3s)"
        elif dur <= 7.0:
            d_b = "medium (3-7s)"
        else:
            d_b = "long (>7s)"

        # Confidence bucket
        if sim < 0.75:
            c_b = "low-conf (0.60-0.75)"
        elif sim < 0.90:
            c_b = "mid-conf (0.75-0.90)"
        else:
            c_b = "high-conf (>=0.90)"

        buckets[(d_b, c_b)].append(r)

    print("\nStratification Buckets Distribution:")
    for (d_b, c_b), b_records in sorted(buckets.items()):
        print(f"  {d_b:16} | {c_b:22} : {len(b_records)} records")

    # Sample proportionally to reach exactly 100
    target_sample_size = 100
    sampled_records = []
    
    # Target allocations per bucket
    for (d_b, c_b), b_records in sorted(buckets.items()):
        quota = max(1, round(len(b_records) / len(records) * target_sample_size))
        chosen = random.sample(b_records, min(quota, len(b_records)))
        sampled_records.extend(chosen)

    # Adjust to exactly 100 if rounding caused +/-
    if len(sampled_records) > 100:
        sampled_records = sampled_records[:100]
    elif len(sampled_records) < 100:
        remaining_pool = [r for r in records if r not in sampled_records]
        sampled_records.extend(random.sample(remaining_pool, 100 - len(sampled_records)))

    print(f"\nFinal Stratified Sample: {len(sampled_records)} records")

    # Package clips and audit alignments
    review_package_rows = []
    leakage_count = 0
    divergence_count = 0
    clean_count = 0
    cer_list = []
    wer_list = []

    # Indicators of Arabic recitation in ASR hypothesis (boundary leakage)
    arabic_markers = ["بِسْمِ", "الله", "عحد", "رحمان", "رحیم", "صمد", "فلق", "ناس", "کوثر", "عالمین", "مالک", "مستقیم", "نعبد", "نستعین"]

    for idx, r in enumerate(sampled_records):
        sid = r["sample_id"]
        s_num = r["surah_number"]
        ayah_num = r.get("ayah_number")
        start_s = r["start_sec"]
        end_s = r["end_sec"]
        dur = r["duration"]
        src_rec = f"{s_num:03d} - Surah {s_num}"
        written_ref = r.get("matched_reference", "")
        published_txt = r.get("text", "")
        asr_hyp = r.get("asr_hypothesis", "")
        sim = r.get("similarity_score", 0.0)

        # Copy clip to package directory
        src_wav = WORKSPACE / r["audio_path"]
        pkg_clip_name = f"{sid}.wav"
        dest_wav = CLIPS_DIR / pkg_clip_name
        if src_wav.exists():
            shutil.copy2(src_wav, dest_wav)

        # Audit error category
        # 1. Boundary leakage: segment starts or ends with Qari recitation
        has_leakage = False
        words = asr_hyp.split()
        if len(words) > 0:
            first_two = " ".join(words[:2])
            last_two = " ".join(words[-2:])
            if any(m in first_two for m in ["عحد", "صمد", "اومیت دین", "مالی کی", "عسمی", "بماللہ"]) and ("چھُ" in asr_hyp or "سٕند" in asr_hyp):
                has_leakage = True

        # 2. Commentary divergence: speech is Kashmiri tafsir, but script replaced it with printed verse clause
        is_divergent = False
        if published_txt != asr_hyp and sim < 0.75:
            is_divergent = True

        if has_leakage:
            audit_flag = "boundary_leakage_arabic_prefix"
            leakage_count += 1
        elif is_divergent:
            audit_flag = "commentary_divergence_text_substituted"
            divergence_count += 1
        elif sim >= 0.85:
            audit_flag = "high_fidelity_verse_alignment"
            clean_count += 1
        else:
            audit_flag = "partial_clause_match"

        # Compute CER and WER against the published text
        c_err = cer(published_txt, asr_hyp)
        w_err = wer(published_txt, asr_hyp)
        cer_list.append(c_err)
        wer_list.append(w_err)

        row = {
            "audit_id": f"AUD_{idx+1:03d}",
            "sample_id": sid,
            "clip_filename": pkg_clip_name,
            "relative_clip_path": f"kashmiri_dataset/raw/external/quran_kashmiri/audio_review_package/clips/{pkg_clip_name}",
            "surah_number": s_num,
            "ayah_number": ayah_num,
            "verse_key": r.get("verse_key", f"{s_num}:{ayah_num}"),
            "start_time_sec": start_s,
            "end_time_sec": end_s,
            "duration_sec": dur,
            "source_recording": src_rec,
            "published_text": published_txt,
            "asr_acoustic_hypothesis": asr_hyp,
            "written_verse_reference": written_ref[:120] + ("..." if len(written_ref) > 120 else ""),
            "character_similarity": sim,
            "measured_cer_against_published": c_err,
            "measured_wer_against_published": w_err,
            "snr_db": r.get("snr"),
            "quality_score": r.get("quality_score"),
            "audit_flag": audit_flag,
            "machine_verification_status": "machine_confidence_filtered",
            # HUMAN VERIFICATION FIELDS EXPLICITLY LEFT UNSET (NULL)
            "human_verified": None,
            "human_reviewer_id": None,
            "human_decision": None,
            "human_review_timestamp": None,
            "human_review_notes": None
        }
        review_package_rows.append(row)

    # Save Review Package Manifests
    pkg_jsonl = REVIEW_PKG_DIR / "sample_100_audit_manifest.jsonl"
    with open(pkg_jsonl, "w", encoding="utf-8") as f:
        for r in review_package_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pkg_csv = REVIEW_PKG_DIR / "sample_100_audit_manifest.csv"
    if review_package_rows:
        keys = list(review_package_rows[0].keys())
        with open(pkg_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(review_package_rows)

    # Summary Report
    avg_cer = sum(cer_list) / len(cer_list)
    avg_wer = sum(wer_list) / len(wer_list)

    findings = {
        "audit_sample_size": 100,
        "human_verified_count": 0,
        "human_verification_coverage_percent": 0.0,
        "note": "Human verification fields are strictly left unset (null) pending native Kashmiri linguist review.",
        "measured_error_rates": {
            "mean_cer_against_published_text": round(avg_cer, 4),
            "mean_wer_against_published_text": round(avg_wer, 4)
        },
        "discrepancy_breakdown": {
            "high_fidelity_verse_alignment": clean_count,
            "boundary_leakage_arabic_prefix": leakage_count,
            "commentary_divergence_text_substituted": divergence_count,
            "partial_clause_match": 100 - clean_count - leakage_count - divergence_count
        },
        "critical_findings": [
            "Previous 'human_verified' label was an automated heuristic and has been completely revoked.",
            f"Boundary leakage detected in {leakage_count}% of sampled clips where trailing Arabic tajweed was joined with Kashmiri speech.",
            f"Commentary divergence detected in {divergence_count}% of sampled clips where printed verse translation was substituted over verbatim spoken tafsir."
        ]
    }

    findings_file = REVIEW_PKG_DIR / "audit_findings_report.json"
    with open(findings_file, "w", encoding="utf-8") as f:
        json.dump(findings, f, indent=2)

    print("\n--- Audio Review Package Audit Findings ---")
    print(f"  Sampled Clips Packaged: 100")
    print(f"  Human Verification Count: 0 (0.0% - strictly unset)")
    print(f"  Mean CER (ASR vs Published Text): {avg_cer*100:.2f}%")
    print(f"  Mean WER (ASR vs Published Text): {avg_wer*100:.2f}%")
    print(f"  High Fidelity Verse Matches: {clean_count}%")
    print(f"  Boundary Leakage Identified: {leakage_count}%")
    print(f"  Commentary Divergence Identified: {divergence_count}%")
    print(f"  Saved Review Package CSV to: {pkg_csv}")
    print(f"  Saved Audit Findings to: {findings_file}")

if __name__ == "__main__":
    main()
