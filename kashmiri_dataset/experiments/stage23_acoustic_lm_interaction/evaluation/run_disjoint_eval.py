#!/usr/bin/env python3
"""
Runs validation on dev_a_disjoint.jsonl (6,257 utts) for:
1. Stage 15 E-03 (Baseline)
2. Stage 22 E22-01 (Baseline, T=1.00)
3. Stage 22 E22-01 (Calibrated, T=1.08)
"""

import os, sys, time

sys.path.insert(0, "/home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage23_acoustic_lm_interaction/evaluation")
from stage23_analysis_engine import run_stage23_analysis

E03_CKPT = "/home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage15_mwer_verification_and_scaling/checkpoints/E-03/best_calib"
E22_CKPT = "/home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage22_data_expanded_training/checkpoints/E22-01/best_model"

LM_PATH = "/home/musa/Kashmiri Project/kashmiri_dataset/experiments/native_asr_training/experiments/stage7_ctc_lm_decoding/language_models/lm_word_4gram_domain_d.arpa"
UNIGRAMS_PATH = "/home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage22b_decoder_and_independent_test/language_model/unigrams.txt"

MANIFEST = "/home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage23_acoustic_lm_interaction/evaluation/dev_a_disjoint.jsonl"
OUT_DIR = "/home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage23_acoustic_lm_interaction/evaluation"

def main():
    runs = [
        ("E22-01_Calib_T108", E22_CKPT, 1.08, os.path.join(OUT_DIR, "dev_a_disjoint_e2201_calib_t108.jsonl")),
        ("E22-01_Base", E22_CKPT, 1.00, os.path.join(OUT_DIR, "dev_a_disjoint_e2201_base.jsonl")),
        ("E-03_Base", E03_CKPT, 1.00, os.path.join(OUT_DIR, "dev_a_disjoint_e03_base.jsonl")),
    ]
    
    t0 = time.time()
    for name, ckpt, temp, out_f in runs:
        print(f"\n=======================================================")
        print(f"Starting run: {name} (T={temp})")
        print(f"=======================================================")
        run_stage23_analysis(
            model_name=name,
            checkpoint_path=ckpt,
            manifest_path=MANIFEST,
            out_jsonl_path=out_f,
            lm_path=LM_PATH,
            unigrams_path=UNIGRAMS_PATH,
            alpha=0.30,
            beta=-1.00,
            beam_width=64,
            temperature=temp,
            num_workers=4
        )
    print(f"\nAll disjoint evaluations completed in {time.time()-t0:.1f}s.")

if __name__ == "__main__":
    main()
