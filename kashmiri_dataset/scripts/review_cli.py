#!/usr/bin/env python3
"""
Interactive CLI Review Interface for Kashmiri Quran Audio Review Package
========================================================================
Enables native Kashmiri speakers to listen to audit clips, verify transcripts,
flag Arabic Tajweed boundary leakage, and log human ground-truth.

Usage:
  python3 kashmiri_dataset/scripts/review_cli.py [--reviewer-id YOUR_NAME] [--limit 10]
"""

import os
import sys
import json
import argparse
import subprocess
import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REVIEW_PKG_DIR = BASE_DIR / "raw/external/quran_kashmiri/audio_review_package"
MANIFEST_PATH = REVIEW_PKG_DIR / "sample_100_audit_manifest.jsonl"
ANN_PATH = REVIEW_PKG_DIR / "reviewer_annotations.json"

def play_audio(clip_path: str):
    """Attempts to play audio using ffplay, aplay, or paplay."""
    if not os.path.exists(clip_path):
        print(f"  [Warning: Clip not found at {clip_path}]")
        return
    for player, args in [("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet"]), ("aplay", ["-q"]), ("paplay", [])]:
        if subprocess.run(["which", player], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            try:
                subprocess.run([player] + args + [clip_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except Exception:
                pass
    print("  [Notice: No terminal audio player available. Open clip manually in media player.]")

def main():
    parser = argparse.ArgumentParser(description="Review Kashmiri Quran Audio Clips")
    parser.add_argument("--reviewer-id", default="reviewer_musa_native", help="Reviewer identity / name")
    parser.add_argument("--limit", type=int, default=10, help="Number of clips to review in this session")
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        print(f"Error: {MANIFEST_PATH} not found.")
        return

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    annotations = {}
    if ANN_PATH.exists():
        try:
            with open(ANN_PATH, "r", encoding="utf-8") as f:
                annotations = json.load(f)
        except Exception:
            annotations = {}

    unreviewed = [r for r in records if r["audit_id"] not in annotations]
    print("\n" + "=" * 65)
    print("    KASHMIRI QUR'AN AUDIO VERIFICATION TERMINAL WORKFLOW")
    print("=" * 65)
    print(f"Reviewer:             {args.reviewer_id}")
    print(f"Total Audit Clips:    {len(records)}")
    print(f"Already Reviewed:     {len(annotations)}")
    print(f"Remaining Unreviewed: {len(unreviewed)}")
    print("=" * 65)

    if not unreviewed:
        print("\nAll 100 audit package clips have been reviewed! Calculating final metrics...")
        cmd = [sys.executable, str(BASE_DIR / "scripts/quran_review_interface.py"), "--annotations", str(ANN_PATH)]
        subprocess.run(cmd)
        return

    to_process = unreviewed[:args.limit]
    print(f"\nBeginning session for {len(to_process)} clips. Press Ctrl+C at any time to exit.\n")

    for idx, r in enumerate(to_process, 1):
        audit_id = r["audit_id"]
        clip_rel = r.get("relative_clip_path")
        clip_path = str(BASE_DIR.parent / clip_rel) if not os.path.exists(str(BASE_DIR / clip_rel)) else str(BASE_DIR / clip_rel)

        print(f"\n---------------------------------------------------------")
        print(f"[{idx}/{len(to_process)}] Clip ID: {audit_id} | Surah {r['surah_number']} Ayah {r.get('verse_key', '')} ({r['duration_sec']}s)")
        print(f"File: {os.path.basename(clip_path)}")
        print(f"Published (Written):   {r.get('published_text', '—')}")
        print(f"ASR Hypothesis:        {r.get('asr_acoustic_hypothesis', '')}")
        print(f"Similarity Score:      {r.get('character_similarity', 0.0)}")
        print(f"---------------------------------------------------------")

        # Auto-play clip once
        print("▶ Playing audio clip...")
        play_audio(clip_path)

        while True:
            print("\nOptions:")
            print("  [A]pprove (Verse Match)     [O]k (Spoken Commentary)")
            print("  [C]orrect (Edit Text)       [B]oundary Leakage (Arabic attached)")
            print("  [P]lay audio again          [R]eject (Noise/Non-Kashmiri)")
            print("  [S]kip for later            [Q]uit & Save")
            choice = input("Enter decision [A/O/C/B/P/R/S/Q]: ").strip().lower()

            if choice == "p":
                print("▶ Playing audio clip...")
                play_audio(clip_path)
                continue
            elif choice == "a":
                annotations[audit_id] = {
                    "audit_id": audit_id,
                    "decision": "approved",
                    "reviewer_id": args.reviewer_id,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "verified_text": r.get("published_text") or r.get("asr_acoustic_hypothesis"),
                    "speech_category": "kashmiri_translation",
                    "arabic_boundary_trim_sec": 0.0,
                    "notes": "Human verified: exact verse translation match."
                }
                print(f"✓ Recorded: APPROVED (Verse Match)")
                break
            elif choice == "o":
                annotations[audit_id] = {
                    "audit_id": audit_id,
                    "decision": "approved",
                    "reviewer_id": args.reviewer_id,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "verified_text": r.get("asr_acoustic_hypothesis"),
                    "speech_category": "kashmiri_commentary",
                    "arabic_boundary_trim_sec": 0.0,
                    "notes": "Human verified: authentic spoken Kashmiri Tafsir commentary."
                }
                print(f"✓ Recorded: APPROVED (Spoken Commentary)")
                break
            elif choice == "c":
                print(f"\nCurrent ASR text:\n  {r.get('asr_acoustic_hypothesis')}")
                corr_text = input("Enter corrected Perso-Arabic text: ").strip()
                if not corr_text:
                    corr_text = r.get("asr_acoustic_hypothesis")
                annotations[audit_id] = {
                    "audit_id": audit_id,
                    "decision": "corrected",
                    "reviewer_id": args.reviewer_id,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "verified_text": corr_text,
                    "speech_category": "kashmiri_translation",
                    "arabic_boundary_trim_sec": 0.0,
                    "notes": "Human verified with manual text correction."
                }
                print(f"✓ Recorded: CORRECTED text")
                break
            elif choice == "b":
                trim = input("Trim Arabic prefix duration in seconds (default 1.0): ").strip()
                try:
                    trim_sec = float(trim) if trim else 1.0
                except ValueError:
                    trim_sec = 1.0
                annotations[audit_id] = {
                    "audit_id": audit_id,
                    "decision": "boundary_leakage",
                    "reviewer_id": args.reviewer_id,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "verified_text": r.get("asr_acoustic_hypothesis"),
                    "speech_category": "uncertain_mixed_speech",
                    "arabic_boundary_trim_sec": trim_sec,
                    "notes": f"Flagged Arabic Tajweed leakage (trimmed {trim_sec}s)."
                }
                print(f"✓ Recorded: BOUNDARY LEAKAGE flagged")
                break
            elif choice == "r":
                reason = input("Reason for rejection: ").strip() or "Low quality / acoustic noise"
                annotations[audit_id] = {
                    "audit_id": audit_id,
                    "decision": "rejected",
                    "reviewer_id": args.reviewer_id,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "verified_text": "",
                    "speech_category": "rejected",
                    "arabic_boundary_trim_sec": 0.0,
                    "notes": reason
                }
                print(f"✗ Recorded: REJECTED")
                break
            elif choice == "s":
                print(f"Skipping {audit_id}.")
                break
            elif choice == "q":
                print("\nSaving progress and exiting...")
                with open(ANN_PATH, "w", encoding="utf-8") as f:
                    json.dump(annotations, f, ensure_ascii=False, indent=2)
                print(f"Saved {len(annotations)} annotations to {ANN_PATH}")
                return
            else:
                print("Invalid choice. Please enter A, O, C, B, P, R, S, or Q.")

        # Save progress after each clip
        with open(ANN_PATH, "w", encoding="utf-8") as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 65)
    print(f"Session complete! Recorded {len(annotations)} total reviews to {ANN_PATH}")
    print("Updating human verification manifest and calculating genuine WER/CER...")
    print("=" * 65)
    cmd = [sys.executable, str(BASE_DIR / "scripts/quran_review_interface.py"), "--annotations", str(ANN_PATH)]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
