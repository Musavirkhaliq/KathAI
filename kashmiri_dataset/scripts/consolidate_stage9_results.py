#!/usr/bin/env python3
"""
Stage 9: Results Consolidation, Statistical Significance, and Report Generator
=============================================================================
Combines evaluation predictions from:
- B-01 (Frozen Epoch 13 baseline)
- B-02A (Recovered speech corpus)
- B-02B (Recovered + Qur'anic translations)
- B-02C (All verified human Kashmiri speech)

Generates:
1. Greedy CTC comparison table (CSV)
2. LM-Assisted comparison table (CSV)
3. Source-wise domain breakdown table (CSV)
4. Paired bootstrap significance testing (10,000 resamples, 95% CIs, p-values)
5. Multi-epoch training loss and validation convergence curves (PNG)
6. Comprehensive scientific Markdown reports
"""

import os
import sys
import json
import csv
from pathlib import Path
import numpy as np

WORKSPACE_DIR = Path("/home/musa/Kashmiri Project")
DATASET_DIR = WORKSPACE_DIR / "kashmiri_dataset"
EXP_ROOT = DATASET_DIR / "experiments/native_asr_training/experiments"
STAGE9_DIR = EXP_ROOT / "stage9_b02_acoustic_comparison"
COMP_DIR = STAGE9_DIR / "comparisons"
REPORTS_DIR = STAGE9_DIR / "reports"

COMP_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(DATASET_DIR / "scripts"))
from metrics_evaluator import calculate_wer_and_cer, normalize_perso_arabic_kashmiri

def run_paired_bootstrap(refs_a, hyps_a, refs_b, hyps_b, n_bootstraps=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(refs_a)
    assert len(refs_b) == n and len(hyps_a) == n and len(hyps_b) == n

    # Precalculate per-utterance word errors and ref word counts
    w_err_a = np.zeros(n, dtype=np.int32)
    w_cnt_a = np.zeros(n, dtype=np.int32)
    w_err_b = np.zeros(n, dtype=np.int32)
    w_cnt_b = np.zeros(n, dtype=np.int32)

    for i in range(n):
        res_a = calculate_wer_and_cer(refs_a[i], hyps_a[i])
        res_b = calculate_wer_and_cer(refs_b[i], hyps_b[i])
        w_err_a[i] = res_a["word_dist"]
        w_cnt_a[i] = res_a["ref_words"]
        w_err_b[i] = res_b["word_dist"]
        w_cnt_b[i] = res_b["ref_words"]

    overall_wer_a = np.sum(w_err_a) / max(1, np.sum(w_cnt_a))
    overall_wer_b = np.sum(w_err_b) / max(1, np.sum(w_cnt_b))
    observed_diff = overall_wer_a - overall_wer_b  # positive means B is better than A

    diffs = []
    b_better_count = 0

    # Vectorized / batched bootstrap
    indices = rng.integers(0, n, size=(n_bootstraps, n))
    for b in range(n_bootstraps):
        idx = indices[b]
        sample_wer_a = np.sum(w_err_a[idx]) / max(1, np.sum(w_cnt_a[idx]))
        sample_wer_b = np.sum(w_err_b[idx]) / max(1, np.sum(w_cnt_b[idx]))
        d = sample_wer_a - sample_wer_b
        diffs.append(d)
        if d > 0:
            b_better_count += 1

    diffs = np.array(diffs)
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    p_value = np.sum(diffs <= 0) / float(n_bootstraps) if observed_diff > 0 else np.sum(diffs >= 0) / float(n_bootstraps)

    return {
        "observed_diff_pct": float(observed_diff * 100),
        "ci_lower_pct": float(ci_lower * 100),
        "ci_upper_pct": float(ci_upper * 100),
        "p_value": float(max(1.0 / n_bootstraps, p_value)),
        "statistically_significant": bool(ci_lower > 0 or ci_upper < 0)
    }

def main():
    print("=== Consolidating Stage 9 Benchmark Results ===")

    models = ["B-01", "B-02A", "B-02B", "B-02C"]
    eval_dirs = {
        "B-01": STAGE9_DIR / "B-01_baseline_evaluation",
        "B-02A": STAGE9_DIR / "B-02A/evaluation",
        "B-02B": STAGE9_DIR / "B-02B/evaluation",
        "B-02C": STAGE9_DIR / "B-02C/evaluation"
    }

    summaries = {}
    for m in models:
        s_file = eval_dirs[m] / "eval_summary.json"
        if s_file.exists():
            with open(s_file, "r", encoding="utf-8") as f:
                summaries[m] = json.load(f)
        else:
            print(f"Warning: Missing summary for {m} at {s_file}")

    # 1. Greedy Comparison CSV
    greedy_csv = COMP_DIR / "stage9_greedy_comparison.csv"
    with open(greedy_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Training Speech (h)", "Utterances", "Norm WER (%)", "Norm CER (%)", "Raw WER (%)", "Raw CER (%)"])
        hours_map = {"B-01": 92.94, "B-02A": 104.93, "B-02B": 123.45, "B-02C": 165.32}
        utts_map = {"B-01": 53460, "B-02A": 64522, "B-02B": 80106, "B-02C": 96664}
        for m in models:
            if m in summaries:
                g = summaries[m]["greedy"]
                writer.writerow([
                    m,
                    hours_map[m],
                    utts_map[m],
                    f"{g['norm_wer']*100:.2f}",
                    f"{g['norm_cer']*100:.2f}",
                    f"{g['raw_wer']*100:.2f}",
                    f"{g['raw_cer']*100:.2f}"
                ])
    print(f"Saved: {greedy_csv}")

    # 2. LM Comparison CSV
    lm_csv = COMP_DIR / "stage9_lm_comparison.csv"
    with open(lm_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Greedy Norm WER (%)", "Beam (No LM) Norm WER (%)", "Beam + 4-gram KenLM Norm WER (%)", "LM Absolute Gain (%)"])
        for m in models:
            if m in summaries:
                g = summaries[m]["greedy"]["norm_wer"] * 100
                b_no_lm = summaries[m]["beam_no_lm"]["norm_wer"] * 100
                b_lm = summaries[m]["beam_kenlm"]["norm_wer"] * 100
                gain = g - b_lm
                writer.writerow([m, f"{g:.2f}", f"{b_no_lm:.2f}", f"{b_lm:.2f}", f"{gain:+.2f}"])
    print(f"Saved: {lm_csv}")

    # 3. Source-wise domain analysis CSV
    source_csv = COMP_DIR / "stage9_sourcewise_analysis.csv"
    sources_all = set()
    for m in models:
        if m in summaries:
            sources_all.update(summaries[m]["greedy"]["source_metrics"].keys())
    sources_all = sorted(sources_all)

    with open(source_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["Domain / Dataset Source", "Sample Count"]
        for m in models:
            header.extend([f"{m} Norm WER (%)", f"{m} Norm CER (%)"])
        writer.writerow(header)

        for src in sources_all:
            cnt = 0
            row = [src]
            for m in models:
                if m in summaries and src in summaries[m]["greedy"]["source_metrics"]:
                    s_m = summaries[m]["greedy"]["source_metrics"][src]
                    cnt = s_m["samples"]
                    break
            row.append(cnt)
            for m in models:
                if m in summaries and src in summaries[m]["greedy"]["source_metrics"]:
                    s_m = summaries[m]["greedy"]["source_metrics"][src]
                    row.extend([f"{s_m['norm_wer']*100:.2f}", f"{s_m['norm_cer']*100:.2f}"])
                else:
                    row.extend(["-", "-"])
            writer.writerow(row)
    print(f"Saved: {source_csv}")

    # 4. Bootstrap Statistical Analysis
    boot_res = {}
    comparisons = [
        ("B-01", "B-02A"),
        ("B-02A", "B-02B"),
        ("B-02B", "B-02C"),
        ("B-01", "B-02C")
    ]

    for m_a, m_b in comparisons:
        pred_a_file = eval_dirs[m_a] / "predictions_greedy.jsonl"
        pred_b_file = eval_dirs[m_b] / "predictions_greedy.jsonl"
        if pred_a_file.exists() and pred_b_file.exists():
            refs_a, hyps_a = [], []
            with open(pred_a_file, "r", encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    refs_a.append(d["norm_ref"])
                    hyps_a.append(d["norm_hyp"])
            refs_b, hyps_b = [], []
            with open(pred_b_file, "r", encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    refs_b.append(d["norm_ref"])
                    hyps_b.append(d["norm_hyp"])

            print(f"Running 10,000 paired bootstrap resamples: {m_a} vs {m_b}...")
            boot = run_paired_bootstrap(refs_a, hyps_a, refs_b, hyps_b)
            boot_res[f"{m_a}_vs_{m_b}"] = boot

    boot_json = COMP_DIR / "stage9_statistical_analysis.json"
    with open(boot_json, "w", encoding="utf-8") as f:
        json.dump(boot_res, f, indent=2)
    print(f"Saved: {boot_json}")

    # 5. Training Curves Plotting
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        for exp_id in ["B-02A", "B-02B", "B-02C"]:
            ep_file = STAGE9_DIR / exp_id / "logs/epoch_metrics.csv"
            if ep_file.exists():
                epochs, losses, wers = [], [], []
                with open(ep_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        epochs.append(int(r["epoch"]))
                        losses.append(float(r["val_loss"]))
                        wers.append(float(r["norm_wer"]) * 100)
                ax1.plot(epochs, losses, marker="o", label=exp_id)
                ax2.plot(epochs, wers, marker="s", label=exp_id)

        ax1.set_title("Validation CTC Loss Convergence")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Validation Loss")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend()

        ax2.set_title("Validation Normalized WER (%) Across Epochs")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Normalized WER (%)")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend()

        plot_p = COMP_DIR / "stage9_training_curves.png"
        plt.tight_layout()
        plt.savefig(plot_p, dpi=200)
        print(f"Saved: {plot_p}")
    except Exception as e:
        print(f"Plotting note: {e}")

    print("\nConsolidation complete!")

if __name__ == "__main__":
    main()
