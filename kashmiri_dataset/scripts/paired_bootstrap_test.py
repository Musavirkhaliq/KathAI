import json, time, sys
from pathlib import Path
import numpy as np

sys.path.append("kashmiri_dataset/scripts")
from metrics_evaluator import calculate_wer_and_cer, normalize_perso_arabic_kashmiri

B01_PREDS_PATH = Path("kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc/full_val_b01_predictions.jsonl")
INDIC_PREDS_PATH = Path("kashmiri_dataset/experiments/native_asr_training/experiments/A-01-indicconformer-kashmiri-finetune/full_val_indicconformer_predictions.jsonl")
OUT_JSON = Path("kashmiri_dataset/experiments/native_asr_training/experiments/A-01-indicconformer-kashmiri-finetune/paired_bootstrap_analysis.json")

def load_preds(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data[item["sample_id"]] = item
    return data

def run_analysis():
    print("Loading prediction records...")
    b01_data = load_preds(B01_PREDS_PATH)
    indic_data = load_preds(INDIC_PREDS_PATH)
    
    common_ids = sorted(list(set(b01_data.keys()) & set(indic_data.keys())))
    print(f"Matched {len(common_ids):,} common samples across both models.")
    
    samples = []
    for sid in common_ids:
        b = b01_data[sid]
        ind = indic_data[sid]
        
        # Verify reference equality after normalization
        ref_norm_b = b["ref_norm"]
        ref_norm_ind = ind["ref_norm"]
        assert ref_norm_b == ref_norm_ind, f"Reference mismatch for {sid}: '{ref_norm_b}' vs '{ref_norm_ind}'"
        
        m_b = calculate_wer_and_cer(ref_norm_b, b["hyp_norm"])
        m_ind = calculate_wer_and_cer(ref_norm_ind, ind["hyp_norm"])
        
        samples.append({
            "sample_id": sid,
            "source": b.get("source", "unknown"),
            "ref_words": m_b["ref_words"],
            "b01_word_dist": m_b["word_dist"],
            "indic_word_dist": m_ind["word_dist"],
            "ref_chars": m_b["ref_chars"],
            "b01_char_dist": m_b["char_dist"],
            "indic_char_dist": m_ind["char_dist"]
        })
        
    N = len(samples)
    ref_words_arr = np.array([s["ref_words"] for s in samples], dtype=np.int64)
    b01_wd_arr = np.array([s["b01_word_dist"] for s in samples], dtype=np.int64)
    indic_wd_arr = np.array([s["indic_word_dist"] for s in samples], dtype=np.int64)
    
    ref_chars_arr = np.array([s["ref_chars"] for s in samples], dtype=np.int64)
    b01_cd_arr = np.array([s["b01_char_dist"] for s in samples], dtype=np.int64)
    indic_cd_arr = np.array([s["indic_char_dist"] for s in samples], dtype=np.int64)
    
    # Overall metrics
    total_ref_words = np.sum(ref_words_arr)
    b01_wer = np.sum(b01_wd_arr) / total_ref_words
    indic_wer = np.sum(indic_wd_arr) / total_ref_words
    diff_wer = b01_wer - indic_wer
    
    total_ref_chars = np.sum(ref_chars_arr)
    b01_cer = np.sum(b01_cd_arr) / total_ref_chars
    indic_cer = np.sum(indic_cd_arr) / total_ref_chars
    diff_cer = b01_cer - indic_cer
    
    print(f"Overall Metrics (N={N}):")
    print(f"  B-01 (XLS-R 300M):         Norm WER = {b01_wer*100:.2f}%, Norm CER = {b01_cer*100:.2f}%")
    print(f"  IndicConformer Pretrained: Norm WER = {indic_wer*100:.2f}%, Norm CER = {indic_cer*100:.2f}%")
    print(f"  Difference (B01 - Indic):  ΔWER = {diff_wer*100:+.2f}%, ΔCER = {diff_cer*100:+.2f}%")
    
    # Paired Bootstrap
    num_bootstrap = 10000
    print(f"Running {num_bootstrap:,} paired bootstrap iterations...")
    np.random.seed(42)
    
    b01_boot_wers = np.zeros(num_bootstrap)
    indic_boot_wers = np.zeros(num_bootstrap)
    diff_boot_wers = np.zeros(num_bootstrap)
    
    b01_boot_cers = np.zeros(num_bootstrap)
    indic_boot_cers = np.zeros(num_bootstrap)
    diff_boot_cers = np.zeros(num_bootstrap)
    
    for i in range(num_bootstrap):
        idx = np.random.randint(0, N, size=N)
        rw_sum = np.sum(ref_words_arr[idx])
        b_w = np.sum(b01_wd_arr[idx]) / rw_sum
        ind_w = np.sum(indic_wd_arr[idx]) / rw_sum
        
        rc_sum = np.sum(ref_chars_arr[idx])
        b_c = np.sum(b01_cd_arr[idx]) / rc_sum
        ind_c = np.sum(indic_cd_arr[idx]) / rc_sum
        
        b01_boot_wers[i] = b_w
        indic_boot_wers[i] = ind_w
        diff_boot_wers[i] = b_w - ind_w
        
        b01_boot_cers[i] = b_c
        indic_boot_cers[i] = ind_c
        diff_boot_cers[i] = b_c - ind_c
        
    p_wer_diff = 2 * min(np.mean(diff_boot_wers <= 0), np.mean(diff_boot_wers >= 0))
    p_cer_diff = 2 * min(np.mean(diff_boot_cers <= 0), np.mean(diff_boot_cers >= 0))
    
    ci_b01_wer = [float(np.percentile(b01_boot_wers, 2.5)), float(np.percentile(b01_boot_wers, 97.5))]
    ci_indic_wer = [float(np.percentile(indic_boot_wers, 2.5)), float(np.percentile(indic_boot_wers, 97.5))]
    ci_diff_wer = [float(np.percentile(diff_boot_wers, 2.5)), float(np.percentile(diff_boot_wers, 97.5))]
    
    ci_b01_cer = [float(np.percentile(b01_boot_cers, 2.5)), float(np.percentile(b01_boot_cers, 97.5))]
    ci_indic_cer = [float(np.percentile(indic_boot_cers, 2.5)), float(np.percentile(indic_boot_cers, 97.5))]
    ci_diff_cer = [float(np.percentile(diff_boot_cers, 2.5)), float(np.percentile(diff_boot_cers, 97.5))]
    
    # Source breakdown
    sources = sorted(list(set(s["source"] for s in samples)))
    source_results = {}
    for src in sources:
        src_samples = [s for s in samples if s["source"] == src]
        s_rw = sum(s["ref_words"] for s in src_samples)
        s_b_wd = sum(s["b01_word_dist"] for s in src_samples)
        s_ind_wd = sum(s["indic_word_dist"] for s in src_samples)
        
        s_rc = sum(s["ref_chars"] for s in src_samples)
        s_b_cd = sum(s["b01_char_dist"] for s in src_samples)
        s_ind_cd = sum(s["indic_char_dist"] for s in src_samples)
        
        source_results[src] = {
            "num_samples": len(src_samples),
            "b01_wer": round(s_b_wd / max(s_rw, 1), 4),
            "indic_wer": round(s_ind_wd / max(s_rw, 1), 4),
            "b01_cer": round(s_b_cd / max(s_rc, 1), 4),
            "indic_cer": round(s_ind_cd / max(s_rc, 1), 4)
        }
        
    out = {
        "num_samples": N,
        "metrics": {
            "b01_xlsr300m": {
                "normalized_wer": round(float(b01_wer), 4),
                "wer_95_ci": [round(x, 4) for x in ci_b01_wer],
                "normalized_cer": round(float(b01_cer), 4),
                "cer_95_ci": [round(x, 4) for x in ci_b01_cer]
            },
            "indicconformer_pretrained": {
                "normalized_wer": round(float(indic_wer), 4),
                "wer_95_ci": [round(x, 4) for x in ci_indic_wer],
                "normalized_cer": round(float(indic_cer), 4),
                "cer_95_ci": [round(x, 4) for x in ci_indic_cer]
            },
            "difference_b01_minus_indic": {
                "delta_wer": round(float(diff_wer), 4),
                "delta_wer_95_ci": [round(x, 4) for x in ci_diff_wer],
                "p_value_wer": round(float(p_wer_diff), 5),
                "delta_cer": round(float(diff_cer), 4),
                "delta_cer_95_ci": [round(x, 4) for x in ci_diff_cer],
                "p_value_cer": round(float(p_cer_diff), 5)
            }
        },
        "source_breakdown": source_results
    }
    
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        
    print(f"\nSaved significance results to {OUT_JSON}")
    print(f"ΔWER 95% CI: [{ci_diff_wer[0]*100:+.2f}%, {ci_diff_wer[1]*100:+.2f}%], p-value = {p_wer_diff:.5f}")
    print(f"ΔCER 95% CI: [{ci_diff_cer[0]*100:+.2f}%, {ci_diff_cer[1]*100:+.2f}%], p-value = {p_cer_diff:.5f}")

if __name__ == "__main__":
    run_analysis()
