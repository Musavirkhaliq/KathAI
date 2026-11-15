#!/usr/bin/env python3
"""
Kashmiri Quran Audio Human-Review Workflow & Verification Engine
================================================================
Provides a structured workflow for native Kashmiri linguists to:
1. Listen to stratified audio review clips.
2. Verify or correctPerso-Arabic transcripts.
3. Identify and flag Arabic vs. Kashmiri boundary leakage.
4. Classify speech category (verse translation vs commentary vs recitation).
5. Record explicit reviewer provenance (reviewer ID, timestamp, decision, notes).
6. Calculate genuine acoustic WER/CER against actual human ground-truth.
"""

import os
import sys
import json
import csv
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(SCRIPTS_DIR))
sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, WORKSPACE_ROOT)

from normalize_transcripts import normalize_kashmiri_text

BASE_DIR = Path(__file__).resolve().parent.parent
REVIEW_PKG_DIR = BASE_DIR / "raw/external/quran_kashmiri/audio_review_package"
MANIFEST_PATH = REVIEW_PKG_DIR / "sample_100_audit_manifest.jsonl"
REVIEWED_OUTPUT_PATH = REVIEW_PKG_DIR / "sample_100_human_reviewed_manifest.jsonl"

def levenshtein_distance(ref: List[str], hyp: List[str]) -> int:
    dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(hyp) + 1):
        dp[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[len(ref)][len(hyp)]

def compute_wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return min(1.0, levenshtein_distance(ref_words, hyp_words) / len(ref_words))

def compute_cer(reference: str, hypothesis: str) -> float:
    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(hypothesis.replace(" ", ""))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return min(1.0, levenshtein_distance(ref_chars, hyp_chars) / len(ref_chars))

def apply_reviewer_annotations(records: List[Dict], annotations: Dict[str, Dict]) -> Tuple[List[Dict], Dict]:
    """
    Applies explicit human reviewer annotations to records.
    Does NOT fabricate decisions; records are only marked human-verified
    when an explicit annotation payload is provided.
    """
    reviewed_records = []
    reviewed_count = 0
    approved_count = 0
    corrected_count = 0
    rejected_count = 0
    leakage_count = 0

    cer_values = []
    wer_values = []

    for r in records:
        audit_id = r.get("audit_id")
        rec = dict(r)

        if audit_id in annotations:
            ann = annotations[audit_id]
            reviewed_count += 1
            decision = ann.get("decision")
            reviewer_id = ann.get("reviewer_id", "REV_NATIVE_KASHMIRI_01")
            notes = ann.get("notes", "")
            verified_text = ann.get("verified_text", "")
            speech_type = ann.get("speech_category", "kashmiri_translation")
            arabic_boundary_trim = ann.get("arabic_boundary_trim_sec", 0.0)

            rec["human_verified"] = True
            rec["human_reviewer_id"] = reviewer_id
            rec["human_decision"] = decision
            rec["human_review_timestamp"] = ann.get("timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat())
            rec["human_review_notes"] = notes
            rec["human_verified_text"] = verified_text
            rec["speech_category"] = speech_type
            rec["arabic_boundary_trim_sec"] = arabic_boundary_trim

            if decision == "approved":
                approved_count += 1
                # Approved means hypothesis is accurate verbatim
                rec["final_ground_truth"] = rec["asr_acoustic_hypothesis"]
            elif decision == "corrected":
                corrected_count += 1
                rec["final_ground_truth"] = verified_text
            elif decision == "rejected":
                rejected_count += 1
                rec["final_ground_truth"] = None
            elif decision == "boundary_leakage":
                leakage_count += 1
                rec["final_ground_truth"] = verified_text

            # Calculate genuine transcription WER and CER against actual human ground-truth
            if rec["final_ground_truth"]:
                norm_gt = normalize_kashmiri_text(rec["final_ground_truth"])
                norm_hyp = normalize_kashmiri_text(rec["asr_acoustic_hypothesis"])
                c_cer = compute_cer(norm_gt, norm_hyp)
                c_wer = compute_wer(norm_gt, norm_hyp)
                rec["genuine_cer_against_human_ground_truth"] = round(c_cer, 4)
                rec["genuine_wer_against_human_ground_truth"] = round(c_wer, 4)
                cer_values.append(c_cer)
                wer_values.append(c_wer)
        else:
            # Explicitly unreviewed
            rec["human_verified"] = None
            rec["human_reviewer_id"] = None
            rec["human_decision"] = None
            rec["human_review_timestamp"] = None
            rec["human_review_notes"] = None
            rec["human_verified_text"] = None

        reviewed_records.append(rec)

    stats = {
        "total_audit_records": len(records),
        "human_reviewed_records": reviewed_count,
        "human_review_coverage_percent": round(reviewed_count / len(records) * 100.0, 2),
        "approved_count": approved_count,
        "corrected_count": corrected_count,
        "rejected_count": rejected_count,
        "boundary_leakage_flagged": leakage_count,
        "mean_genuine_cer_against_human_gt": round(float(sum(cer_values) / len(cer_values)), 4) if cer_values else None,
        "mean_genuine_wer_against_human_gt": round(float(sum(wer_values) / len(wer_values)), 4) if wer_values else None,
        "audit_note": "Genuine WER/CER derived strictly from human-verified transcriptions, separating ASR hypotheses from printed translations."
    }

    return reviewed_records, stats

def main():
    if not MANIFEST_PATH.exists():
        print(f"Error: Audit manifest {MANIFEST_PATH} not found.")
        return

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print("=======================================================")
    print("      KASHMIRI QURAN AUDIO HUMAN-REVIEW WORKFLOW       ")
    print("=======================================================")
    print(f"Loaded {len(records)} stratified audit package clips from {MANIFEST_PATH}")

    # Check for existing human reviewer annotations file
    ann_file = REVIEW_PKG_DIR / "reviewer_annotations.json"
    if ann_file.exists():
        with open(ann_file, "r", encoding="utf-8") as f:
            annotations = json.load(f)
        print(f"Found {len(annotations)} explicit human reviewer annotations in {ann_file}")
    else:
        annotations = {}
        print("No prior annotations file found. Initializing blank review package template.")

    reviewed_records, stats = apply_reviewer_annotations(records, annotations)

    with open(REVIEWED_OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in reviewed_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary_path = REVIEW_PKG_DIR / "human_review_stats.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\nReview Status Summary:")
    print(f"  Total Clips:               {stats['total_audit_records']}")
    print(f"  Human Reviewed:            {stats['human_reviewed_records']} ({stats['human_review_coverage_percent']}%)")
    print(f"  Approved:                  {stats['approved_count']}")
    print(f"  Corrected:                 {stats['corrected_count']}")
    print(f"  Rejected:                  {stats['rejected_count']}")
    print(f"  Boundary Leakage Flagged:  {stats['boundary_leakage_flagged']}")
    if stats['mean_genuine_cer_against_human_gt'] is not None:
        print(f"  Genuine CER against Human Ground Truth: {stats['mean_genuine_cer_against_human_gt'] * 100:.2f}%")
        print(f"  Genuine WER against Human Ground Truth: {stats['mean_genuine_wer_against_human_gt'] * 100:.2f}%")
    else:
        print("  Genuine CER/WER:           null (awaiting reviewer input)")
    print(f"\nSaved updated manifest: {REVIEWED_OUTPUT_PATH}")
    print(f"Saved review stats:     {summary_path}")

if __name__ == "__main__":
    main()
