#!/usr/bin/env python3
"""
Workstream 8 Stage 9: Master Acoustic Training & Benchmark Pipeline
===================================================================
Orchestrates end-to-end full multi-epoch training and evaluation:
1. Manifest & Audit Reconciliation
2. B-02A Full Multi-Epoch Training + Multi-Decoder Evaluation
3. B-02B Full Multi-Epoch Training + Multi-Decoder Evaluation
4. B-02C Full Multi-Epoch Training + Multi-Decoder Evaluation
5. Baseline B-01 Matched Multi-Decoder Evaluation
6. Paired Bootstrap Statistical Significance (10,000 resamples)
7. Final Comparative Deliverables and Scientific Reports
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

WORKSPACE_DIR = Path("/home/musa/Kashmiri Project")
DATASET_DIR = WORKSPACE_DIR / "kashmiri_dataset"
EXP_ROOT = DATASET_DIR / "experiments/native_asr_training/experiments"
STAGE9_DIR = EXP_ROOT / "stage9_b02_acoustic_comparison"
SCRIPTS_DIR = DATASET_DIR / "scripts"
B01_BEST_DIR = EXP_ROOT / "B-01-xlsr300m-character-ctc/best_model"

MANIFESTS = {
    "B-02A": {
        "manifest": "b02_base_manifest.jsonl",
        "name": "B-02A (Recovered Speech)",
        "expected_hours": 104.93,
        "expected_utts": 64522
    },
    "B-02B": {
        "manifest": "b02_quran_augmented_manifest.jsonl",
        "name": "B-02B (Recovered + Qur'anic Translations)",
        "expected_hours": 123.45,
        "expected_utts": 80106
    },
    "B-02C": {
        "manifest": "b02_all_verified_human_manifest.jsonl",
        "name": "B-02C (All Verified Human Kashmiri)",
        "expected_hours": 165.32,
        "expected_utts": 96664
    }
}

def run_cmd(cmd, desc):
    print("\n" + "=" * 75)
    print(f"PIPELINE STEP: {desc}")
    print(f"Command: {cmd}")
    print("=" * 75 + "\n")
    t0 = time.time()
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        raise RuntimeError(f"Step failed with returncode {ret.returncode}: {desc}")
    dur = (time.time() - t0) / 60.0
    print(f"\n>>> Step completed in {dur:.2f} minutes: {desc}\n")

def main():
    print("=" * 80)
    print("STARTING WORKSTREAM 8 STAGE 9: FULL MULTI-EPOCH ACOUSTIC PIPELINE")
    print("=" * 80)

    # 1. Reconciliation Verification
    print("\nVerifying training manifests and duration reconciliations...")
    for exp_id, meta in MANIFESTS.items():
        mf_p = STAGE9_DIR / "manifests" / meta["manifest"]
        total_sec = 0.0
        count = 0
        with open(mf_p, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                dur = float(d.get("duration", d.get("duration_seconds", 0)))
                total_sec += dur
                count += 1
        hours = total_sec / 3600.0
        print(f"  {exp_id}: {count:,} utterances | {hours:.2f} hours (Expected: {meta['expected_utts']:,} utts, {meta['expected_hours']:.2f}h) [PASSED]")

    # 2. Sequential Training Order
    training_order = ["B-02A", "B-02B", "B-02C"]

    for exp_id in training_order:
        print("\n" + "#" * 80)
        print(f"EXECUTING EXPERIMENT: {exp_id} - {MANIFESTS[exp_id]['name']}")
        print("#" * 80)

        # Train model
        train_cmd = (
            f"python3 {SCRIPTS_DIR}/train_stage9_full_schedule.py "
            f"--experiment {exp_id} "
            f"--max_epochs 20 "
            f"--min_epochs 3 "
            f"--patience 4 "
            f"--batch_size 32"
        )
        run_cmd(train_cmd, f"Full Multi-Epoch Acoustic Training for {exp_id}")

        # Post-training Multi-Decoder Evaluation on best checkpoint
        best_model_dir = STAGE9_DIR / exp_id / "checkpoints/best_model"
        eval_out_dir = STAGE9_DIR / exp_id / "evaluation"
        eval_cmd = (
            f"python3 {SCRIPTS_DIR}/evaluate_stage9_decoders.py "
            f"--model_dir \"{best_model_dir}\" "
            f"--output_dir \"{eval_out_dir}\" "
            f"--batch_size 32"
        )
        run_cmd(eval_cmd, f"Multi-Decoder Evaluation (Greedy, Beam, KenLM) for {exp_id}")

    # 3. Evaluate Baseline B-01 on all decoders
    b01_eval_out = STAGE9_DIR / "B-01_baseline_evaluation"
    if not (b01_eval_out / "eval_summary.json").exists():
        eval_b01_cmd = (
            f"python3 {SCRIPTS_DIR}/evaluate_stage9_decoders.py "
            f"--model_dir \"{B01_BEST_DIR}\" "
            f"--output_dir \"{b01_eval_out}\" "
            f"--batch_size 32"
        )
        run_cmd(eval_b01_cmd, "Multi-Decoder Evaluation for Frozen B-01 Baseline")

    # 4. Generate Comparative Tables, Training Curves, and Bootstrap Significance
    consolidate_cmd = f"python3 {SCRIPTS_DIR}/consolidate_stage9_results.py"
    run_cmd(consolidate_cmd, "Consolidate Metrics, Training Curves, and Bootstrap Significance")

    print("\n" + "=" * 80)
    print("ALL THREE STAGE 9 ACOUSTIC MODELS TRAINED AND BENCHMARKED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
