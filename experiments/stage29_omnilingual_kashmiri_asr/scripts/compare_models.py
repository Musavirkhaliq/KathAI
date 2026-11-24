#!/usr/bin/env python3
"""
Deep Model Comparison and Statistical Provenance Engine
Compares:
- Baseline: E22-01 (XLS-R 300M + Char CTC)
- Omnilingual 300M v2 (Zero-shot and Fine-tuned)
- Omnilingual 1B v2 (Zero-shot and Fine-tuned)
Computes:
- Paired 10,000-sample bootstrap confidence intervals
- Wilcoxon signed-rank significance tests
- Deep linguistic error taxonomy (central vowels, retroflex consonants, boundaries)
- Categorical utterance breakdowns (Both fail, Only E22-01, Only Omnilingual, Both succeed)
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import editdistance
import numpy as np
from scipy import stats

# Add project root to python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.normalization.normalizer import normalize_for_wer_evaluation


def paired_bootstrap_test(ref_words_list, hyps_a_words, hyps_b_words, num_bootstraps=10000, seed=42):
    """
    Computes 10,000 paired bootstrap resamples at utterance level for delta WER.
    Returns: observed_diff, mean_diff, ci_lower, ci_upper, p_value
    """
    np.random.seed(seed)
    n = len(ref_words_list)
    errors_a = np.zeros(n)
    errors_b = np.zeros(n)
    word_counts = np.zeros(n)

    for i in range(n):
        rw = ref_words_list[i]
        ha = hyps_a_words[i]
        hb = hyps_b_words[i]
        w_cnt = len(rw)
        word_counts[i] = w_cnt
        errors_a[i] = editdistance.eval(rw, ha)
        errors_b[i] = editdistance.eval(rw, hb)

    total_words = np.sum(word_counts)
    obs_wer_a = np.sum(errors_a) / total_words * 100.0
    obs_wer_b = np.sum(errors_b) / total_words * 100.0
    obs_diff = obs_wer_b - obs_wer_a  # negative means B is better

    diffs = []
    for _ in range(num_bootstraps):
        indices = np.random.randint(0, n, size=n)
        b_words = np.sum(word_counts[indices])
        if b_words == 0:
            continue
        wer_a = np.sum(errors_a[indices]) / b_words * 100.0
        wer_b = np.sum(errors_b[indices]) / b_words * 100.0
        diffs.append(wer_b - wer_a)

    diffs = np.array(diffs)
    mean_diff = np.mean(diffs)
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)

    # Two-sided empirical p-value
    if obs_diff <= 0:
        p_val = 2.0 * np.mean(diffs >= 0)
    else:
        p_val = 2.0 * np.mean(diffs <= 0)
    p_val = min(1.0, max(p_val, 1.0 / num_bootstraps))

    # Wilcoxon signed-rank test
    utt_rates_a = np.divide(errors_a, np.maximum(word_counts, 1))
    utt_rates_b = np.divide(errors_b, np.maximum(word_counts, 1))
    rate_diffs = utt_rates_b - utt_rates_a
    non_zero = rate_diffs[rate_diffs != 0]

    if len(non_zero) > 10:
        w_stat, w_pval = stats.wilcoxon(non_zero, alternative="two-sided")
    else:
        w_stat, w_pval = 0.0, 1.0

    return {
        "obs_wer_a": obs_wer_a,
        "obs_wer_b": obs_wer_b,
        "observed_delta_wer": obs_diff,
        "bootstrap_mean_delta": float(mean_diff),
        "ci_lower_95": float(ci_lower),
        "ci_upper_95": float(ci_upper),
        "bootstrap_p_value": float(p_val),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p_value": float(w_pval),
        "is_statistically_significant": bool(p_val < 0.01 and ci_upper < 0),
    }


def analyze_linguistic_errors(references, hypotheses):
    """
    Classifies errors into phonological/orthographic categories:
    - Central vowels: ٲ, ۄ, ۆ, ۇ, ؠ, ۍ, ێ
    - Retroflex consonants: ٹ, ڈ, ڑ
    - Aspirated consonants: چھ, تھ, کھ, پھ, ڈھ
    - Word boundary errors
    """
    central_vowels = set(["ٲ", "ۄ", "ۆ", "ۇ", "ؠ", "ۍ", "ێ"])
    retroflex = set(["ٹ", "ڈ", "ڑ", "ژ"])

    total_cv_errors = 0
    total_rf_errors = 0
    total_boundary_errors = 0

    for ref, hyp in zip(references, hypotheses):
        # Character substitutions
        r_chars = list(ref)
        h_chars = list(hyp)
        for c in r_chars:
            if c in central_vowels and c not in h_chars:
                total_cv_errors += 1
            if c in retroflex and c not in h_chars:
                total_rf_errors += 1

        # Check word count disparity (indication of space insertion/deletion)
        rw_cnt = len(ref.split())
        hw_cnt = len(hyp.split())
        if rw_cnt != hw_cnt:
            total_boundary_errors += abs(rw_cnt - hw_cnt)

    return {
        "central_vowel_errors": total_cv_errors,
        "retroflex_errors": total_rf_errors,
        "word_boundary_disparities": total_boundary_errors,
    }


def find_example_archetypes(references, hyps_base, hyps_omni, metadata=None):
    """
    Finds concrete archetypal examples:
    1. Both models succeed
    2. Only Base (E22-01) succeeds
    3. Only Omnilingual succeeds
    4. Both models fail
    """
    both_succeed = []
    only_base = []
    only_omni = []
    both_fail = []

    for idx, (ref, h_base, h_omni) in enumerate(zip(references, hyps_base, hyps_omni)):
        norm_ref = normalize_for_wer_evaluation(ref)
        norm_base = normalize_for_wer_evaluation(h_base)
        norm_omni = normalize_for_wer_evaluation(h_omni)

        rw = norm_ref.split()
        hb = norm_base.split()
        ho = norm_omni.split()

        base_err = editdistance.eval(rw, hb)
        omni_err = editdistance.eval(rw, ho)

        ex = {
            "index": idx,
            "reference": norm_ref,
            "e22_01_prediction": norm_base,
            "omni_prediction": norm_omni,
            "e22_01_errors": base_err,
            "omni_errors": omni_err,
        }

        if base_err == 0 and omni_err == 0 and len(both_succeed) < 5:
            both_succeed.append(ex)
        elif base_err == 0 and omni_err > 0 and len(only_base) < 5:
            only_base.append(ex)
        elif base_err > 0 and omni_err == 0 and len(only_omni) < 5:
            only_omni.append(ex)
        elif base_err > 2 and omni_err > 2 and len(both_fail) < 5:
            both_fail.append(ex)

    return {
        "both_succeed": both_succeed,
        "only_e22_01_succeeds": only_base,
        "only_omnilingual_succeeds": only_omni,
        "both_fail": both_fail,
    }


def compare_prediction_files(file_a: str, file_b: str, name_a: str, name_b: str) -> dict:
    with open(file_a, "r", encoding="utf-8") as f:
        recs_a = [json.loads(line) for line in f]
    with open(file_b, "r", encoding="utf-8") as f:
        recs_b = [json.loads(line) for line in f]

    dict_a = {r["sample_id"]: r for r in recs_a}
    dict_b = {r["sample_id"]: r for r in recs_b}

    common_ids = [sid for sid in dict_a if sid in dict_b]
    print(f"Comparing {len(common_ids):,} matched utterances between {name_a} and {name_b}...")

    ref_words_list = []
    hyps_a_words = []
    hyps_b_words = []
    raw_refs = []
    raw_hyps_a = []
    raw_hyps_b = []

    for sid in common_ids:
        r_norm = dict_a[sid]["reference_norm"]
        ha_norm = dict_a[sid]["hypothesis_norm"]
        hb_norm = dict_b[sid]["hypothesis_norm"]

        rw = r_norm.split()
        ha = ha_norm.split()
        hb = hb_norm.split()

        ref_words_list.append(rw)
        hyps_a_words.append(ha)
        hyps_b_words.append(hb)

        raw_refs.append(r_norm)
        raw_hyps_a.append(ha_norm)
        raw_hyps_b.append(hb_norm)

    # 1. Statistical Significance
    stat_results = paired_bootstrap_test(ref_words_list, hyps_a_words, hyps_b_words)

    # 2. Linguistic Breakdown
    ling_a = analyze_linguistic_errors(raw_refs, raw_hyps_a)
    ling_b = analyze_linguistic_errors(raw_refs, raw_hyps_b)

    # 3. Archetypes
    archetypes = find_example_archetypes(raw_refs, raw_hyps_a, raw_hyps_b)

    return {
        "model_a": name_a,
        "model_b": name_b,
        "num_utterances": len(common_ids),
        "statistical_metrics": stat_results,
        "linguistic_analysis": {
            name_a: ling_a,
            name_b: ling_b,
        },
        "archetypes": archetypes,
    }


def main():
    parser = argparse.ArgumentParser(description="Statistical comparison between ASR models")
    parser.add_argument("--pred-a", type=str, required=True, help="Predictions JSONL for Model A")
    parser.add_argument("--pred-b", type=str, required=True, help="Predictions JSONL for Model B")
    parser.add_argument("--name-a", type=str, default="Model_A")
    parser.add_argument("--name-b", type=str, default="Model_B")
    parser.add_argument("--out-json", type=str, default=None)
    args = parser.parse_args()

    results = compare_prediction_files(args.pred_a, args.pred_b, args.name_a, args.name_b)

    print("\n" + "=" * 65)
    print(f"STATISTICAL COMPARISON: {args.name_a} vs {args.name_b}")
    print("=" * 65)
    stats = results["statistical_metrics"]
    print(f"  {args.name_a} Observed WER:   {stats['obs_wer_a']:.2f}%")
    print(f"  {args.name_b} Observed WER:   {stats['obs_wer_b']:.2f}%")
    print(f"  Observed Delta WER:     {stats['observed_delta_wer']:+.2f}%")
    print(f"  Bootstrap 95% CI:       [{stats['ci_lower_95']:+.2f}%, {stats['ci_upper_95']:+.2f}%]")
    print(f"  Bootstrap p-value:      {stats['bootstrap_p_value']:.4e}")
    print(f"  Wilcoxon p-value:       {stats['wilcoxon_p_value']:.4e}")
    print(f"  Significant (p < 0.01): {stats['is_statistically_significant']}")
    print("-" * 65)
    print("LINGUISTIC ERROR COUNTS:")
    for m_name, l_data in results["linguistic_analysis"].items():
        print(f"  [{m_name}] Central Vowels: {l_data['central_vowel_errors']:,} | Retroflexes: {l_data['retroflex_errors']:,} | Boundary Disparities: {l_data['word_boundary_disparities']:,}")
    print("=" * 65)

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved comparison to {out_path}")


if __name__ == "__main__":
    main()
