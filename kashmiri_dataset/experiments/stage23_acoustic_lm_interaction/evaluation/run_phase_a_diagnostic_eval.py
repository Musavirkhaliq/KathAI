#!/usr/bin/env python3
"""
Phase A & Diagnostic Runner for Stage 23.
Runs E-03 and E22-01 on dev_calibration, dev_b_waaz, and dev_c_clean.
"""

import os, sys, time

sys.path.insert(0, "/home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage23_acoustic_lm_interaction/evaluation")
from stage23_analysis_engine import run_stage23_analysis

MODELS = {
    "E-03": {
        "checkpoint": "/home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage15_mwer_verification_and_scaling/checkpoints/E-03/best_calib",
    },
    "E22-01": {
        "checkpoint": "/home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage22_data_expanded_training/checkpoints/E22-01/best_model",
    }
}

SPLITS = {
    "dev_calibration": "/home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage11_robustness_optimization/data_splits/dev_calibration.jsonl",
    "dev_b_waaz": "/home/musa/Kashmiri Project/kashmiri_dataset/releases/asr_training_v2_waaz/validation_manifests/dev_b_waaz_val.jsonl",
    "dev_c_clean": "/home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage23_acoustic_lm_interaction/evaluation/dev_c_clean.jsonl"
}

OUT_DIR = "/home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage23_acoustic_lm_interaction/evaluation"
LM_PATH = "/home/musa/Kashmiri Project/kashmiri_dataset/experiments/native_asr_training/experiments/stage7_ctc_lm_decoding/language_models/lm_word_4gram_domain_d.arpa"
UNIGRAMS_PATH = "/home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage22b_decoder_and_independent_test/language_model/unigrams.txt"

def main():
    t_start = time.time()
    for split_name, split_path in SPLITS.items():
        for m_name, m_cfg in MODELS.items():
            out_file = os.path.join(OUT_DIR, f"{split_name}_{m_name.lower().replace('-', '')}_predictions.jsonl")
            print(f"\n>>> Running {m_name} on {split_name} -> {out_file}")
            run_stage23_analysis(
                model_name=m_name,
                checkpoint_path=m_cfg["checkpoint"],
                manifest_path=split_path,
                out_jsonl_path=out_file,
                lm_path=LM_PATH,
                unigrams_path=UNIGRAMS_PATH,
                alpha=0.30,
                beta=-1.00,
                beam_width=64,
                num_workers=4
            )
    print(f"\nAll Phase A diagnostic runs completed in {time.time()-t_start:.1f}s.")

if __name__ == "__main__":
    main()
