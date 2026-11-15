import json, sys, csv
from pathlib import Path
import numpy as np
from collections import defaultdict

sys.path.append("kashmiri_dataset/scripts")
from metrics_evaluator import calculate_wer_and_cer, normalize_perso_arabic_kashmiri

STAGE6_DIR = Path("kashmiri_dataset/experiments/native_asr_training/experiments/stage6_held_out_benchmark")
B01_PREDS_PATH = STAGE6_DIR / "b01_test_predictions.jsonl"
INDIC_PREDS_PATH = STAGE6_DIR / "indicconformer_test_predictions.jsonl"

def load_predictions(path):
    preds = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            preds[item["sample_id"]] = item
    return preds

def detailed_levenshtein(ref_tokens, hyp_tokens):
    # Returns (substitutions, deletions, insertions, distance)
    d = np.zeros((len(ref_tokens) + 1, len(hyp_tokens) + 1), dtype=int)
    for i in range(len(ref_tokens) + 1):
        d[i, 0] = i
    for j in range(len(hyp_tokens) + 1):
        d[0, j] = j
        
    for i in range(1, len(ref_tokens) + 1):
        for j in range(1, len(hyp_tokens) + 1):
            if ref_tokens[i-1] == hyp_tokens[j-1]:
                d[i, j] = d[i-1, j-1]
            else:
                d[i, j] = min(
                    d[i-1, j] + 1,      # deletion
                    d[i, j-1] + 1,      # insertion
                    d[i-1, j-1] + 1    # substitution
                )
    
    # Backtrack to count S, D, I
    i, j = len(ref_tokens), len(hyp_tokens)
    s, dele, ins = 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_tokens[i-1] == hyp_tokens[j-1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and d[i, j] == d[i-1, j-1] + 1:
            s += 1
            i -= 1
            j -= 1
        elif i > 0 and d[i, j] == d[i-1, j] + 1:
            dele += 1
            i -= 1
        elif j > 0 and d[i, j] == d[i, j-1] + 1:
            ins += 1
            j -= 1
        else:
            break
            
    return {"s": s, "d": dele, "i": ins, "dist": d[len(ref_tokens), len(hyp_tokens)]}

def run_held_out_analysis():
    print("Loading test prediction records...")
    b01_data = load_predictions(B01_PREDS_PATH)
    indic_data = load_predictions(INDIC_PREDS_PATH)
    
    common_ids = sorted(list(set(b01_data.keys()) & set(indic_data.keys())))
    N = len(common_ids)
    print(f"Matched {N:,} test samples across both models.")
    
    records = []
    
    # Error accumulators
    b01_words_s, b01_words_d, b01_words_i = 0, 0, 0
    indic_words_s, indic_words_d, indic_words_i = 0, 0, 0
    
    b01_chars_s, b01_chars_d, b01_chars_i = 0, 0, 0
    indic_chars_s, indic_chars_d, indic_chars_i = 0, 0, 0
    
    b01_exact_matches = 0
    indic_exact_matches = 0
    
    b01_empty = 0
    indic_empty = 0
    
    total_audio_sec = 0.0
    
    for sid in common_ids:
        b = b01_data[sid]
        ind = indic_data[sid]
        
        ref_norm = b["ref_norm"]
        dur = b.get("duration", 0.0)
        total_audio_sec += dur
        
        b_hyp_norm = b["hyp_norm"]
        ind_hyp_norm = ind["hyp_norm"]
        
        if len(b["hyp_raw"].strip()) == 0:
            b01_empty += 1
        if len(ind["hyp_raw"].strip()) == 0:
            indic_empty += 1
            
        ref_words = ref_norm.strip().split()
        b_words = b_hyp_norm.strip().split()
        ind_words = ind_hyp_norm.strip().split()
        
        ref_chars = list(ref_norm.replace(" ", ""))
        b_chars = list(b_hyp_norm.replace(" ", ""))
        ind_chars = list(ind_hyp_norm.replace(" ", ""))
        
        w_b = detailed_levenshtein(ref_words, b_words)
        w_ind = detailed_levenshtein(ref_words, ind_words)
        
        c_b = detailed_levenshtein(ref_chars, b_chars)
        c_ind = detailed_levenshtein(ref_chars, ind_chars)
        
        b01_words_s += w_b["s"]
        b01_words_d += w_b["d"]
        b01_words_i += w_b["i"]
        
        indic_words_s += w_ind["s"]
        indic_words_d += w_ind["d"]
        indic_words_i += w_ind["i"]
        
        b01_chars_s += c_b["s"]
        b01_chars_d += c_b["d"]
        b01_chars_i += c_b["i"]
        
        indic_chars_s += c_ind["s"]
        indic_chars_d += c_ind["d"]
        indic_chars_i += c_ind["i"]
        
        if w_b["dist"] == 0:
            b01_exact_matches += 1
        if w_ind["dist"] == 0:
            indic_exact_matches += 1
            
        records.append({
            "sample_id": sid,
            "source": b.get("source", "unknown"),
            "speaker_id": b.get("speaker_id", "unknown"),
            "duration": dur,
            "ref_words": len(ref_words),
            "ref_chars": len(ref_chars),
            "b01_word_dist": w_b["dist"],
            "indic_word_dist": w_ind["dist"],
            "b01_char_dist": c_b["dist"],
            "indic_char_dist": c_ind["dist"]
        })
        
    rw_arr = np.array([r["ref_words"] for r in records], dtype=np.int64)
    b01_wd_arr = np.array([r["b01_word_dist"] for r in records], dtype=np.int64)
    indic_wd_arr = np.array([r["indic_word_dist"] for r in records], dtype=np.int64)
    
    rc_arr = np.array([r["ref_chars"] for r in records], dtype=np.int64)
    b01_cd_arr = np.array([r["b01_char_dist"] for r in records], dtype=np.int64)
    indic_cd_arr = np.array([r["indic_char_dist"] for r in records], dtype=np.int64)
    
    total_ref_words = np.sum(rw_arr)
    total_ref_chars = np.sum(rc_arr)
    
    b01_norm_wer = np.sum(b01_wd_arr) / total_ref_words
    indic_norm_wer = np.sum(indic_wd_arr) / total_ref_words
    delta_wer = b01_norm_wer - indic_norm_wer
    
    b01_norm_cer = np.sum(b01_cd_arr) / total_ref_chars
    indic_norm_cer = np.sum(indic_cd_arr) / total_ref_chars
    delta_cer = b01_norm_cer - indic_norm_cer
    
    print("\n=== Corpus-Level Held-Out Test Results ===")
    print(f"Total Eligible Samples: {N:,} ({total_audio_sec/3600:.2f} hours)")
    print(f"Total Reference Words: {total_ref_words:,} | Total Reference Characters: {total_ref_chars:,}")
    print(f"B-01 (XLS-R 300M):         Norm WER = {b01_norm_wer*100:.2f}%, Norm CER = {b01_norm_cer*100:.2f}%")
    print(f"IndicConformer Pretrained: Norm WER = {indic_norm_wer*100:.2f}%, Norm CER = {indic_norm_cer*100:.2f}%")
    print(f"Difference (B01 - Indic):  ΔWER = {delta_wer*100:+.2f}%, ΔCER = {delta_cer*100:+.2f}%")
    
    # 10,000 Paired Bootstrap Iterations
    num_bootstrap = 10000
    print(f"\nExecuting {num_bootstrap:,} paired bootstrap iterations (seed=42)...")
    np.random.seed(42)
    
    diff_wers = np.zeros(num_bootstrap)
    diff_cers = np.zeros(num_bootstrap)
    b01_wers = np.zeros(num_bootstrap)
    indic_wers = np.zeros(num_bootstrap)
    b01_cers = np.zeros(num_bootstrap)
    indic_cers = np.zeros(num_bootstrap)
    
    for i in range(num_bootstrap):
        idx = np.random.randint(0, N, size=N)
        rw_sum = np.sum(rw_arr[idx])
        rc_sum = np.sum(rc_arr[idx])
        
        bw = np.sum(b01_wd_arr[idx]) / rw_sum
        iw = np.sum(indic_wd_arr[idx]) / rw_sum
        
        bc = np.sum(b01_cd_arr[idx]) / rc_sum
        ic = np.sum(indic_cd_arr[idx]) / rc_sum
        
        b01_wers[i] = bw
        indic_wers[i] = iw
        diff_wers[i] = bw - iw
        
        b01_cers[i] = bc
        indic_cers[i] = ic
        diff_cers[i] = bc - ic
        
    p_wer = 2 * min(np.mean(diff_wers <= 0), np.mean(diff_wers >= 0))
    p_cer = 2 * min(np.mean(diff_cers <= 0), np.mean(diff_cers >= 0))
    
    ci_b01_wer = [float(np.percentile(b01_wers, 2.5)), float(np.percentile(b01_wers, 97.5))]
    ci_indic_wer = [float(np.percentile(indic_wers, 2.5)), float(np.percentile(indic_wers, 97.5))]
    ci_diff_wer = [float(np.percentile(diff_wers, 2.5)), float(np.percentile(diff_wers, 97.5))]
    
    ci_b01_cer = [float(np.percentile(b01_cers, 2.5)), float(np.percentile(b01_cers, 97.5))]
    ci_indic_cer = [float(np.percentile(indic_cers, 2.5)), float(np.percentile(indic_cers, 97.5))]
    ci_diff_cer = [float(np.percentile(diff_cers, 2.5)), float(np.percentile(diff_cers, 97.5))]
    
    print(f"ΔWER 95% CI: [{ci_diff_wer[0]*100:+.2f}%, {ci_diff_wer[1]*100:+.2f}%], p-value = {p_wer:.5f}")
    print(f"ΔCER 95% CI: [{ci_diff_cer[0]*100:+.2f}%, {ci_diff_cer[1]*100:+.2f}%], p-value = {p_cer:.5f}")
    
    # Speaker-clustered Bootstrap Analysis
    speaker_groups = defaultdict(list)
    for idx, r in enumerate(records):
        spk = r["speaker_id"]
        speaker_groups[spk].append(idx)
        
    spk_keys = list(speaker_groups.keys())
    print(f"Found {len(spk_keys):,} unique speaker clusters.")
    
    spk_diff_wers = np.zeros(num_bootstrap)
    for i in range(num_bootstrap):
        sampled_spks = np.random.choice(len(spk_keys), size=len(spk_keys), replace=True)
        idx_list = []
        for s_idx in sampled_spks:
            idx_list.extend(speaker_groups[spk_keys[s_idx]])
        idx_arr = np.array(idx_list)
        
        rw_sum = np.sum(rw_arr[idx_arr])
        bw = np.sum(b01_wd_arr[idx_arr]) / rw_sum
        iw = np.sum(indic_wd_arr[idx_arr]) / rw_sum
        spk_diff_wers[i] = bw - iw
        
    ci_spk_diff_wer = [float(np.percentile(spk_diff_wers, 2.5)), float(np.percentile(spk_diff_wers, 97.5))]
    p_spk_wer = 2 * min(np.mean(spk_diff_wers <= 0), np.mean(spk_diff_wers >= 0))
    print(f"Speaker-Clustered ΔWER 95% CI: [{ci_spk_diff_wer[0]*100:+.2f}%, {ci_spk_diff_wer[1]*100:+.2f}%], p = {p_spk_wer:.5f}")
    
    # Per-source analysis
    sources = sorted(list(set(r["source"] for r in records)))
    source_rows = []
    source_dict = {}
    
    for src in sources:
        s_recs = [r for r in records if r["source"] == src]
        s_rw = sum(r["ref_words"] for r in s_recs)
        s_rc = sum(r["ref_chars"] for r in s_recs)
        s_dur = sum(r["duration"] for r in s_recs)
        
        s_b_wd = sum(r["b01_word_dist"] for r in s_recs)
        s_i_wd = sum(r["indic_word_dist"] for r in s_recs)
        
        s_b_cd = sum(r["b01_char_dist"] for r in s_recs)
        s_i_cd = sum(r["indic_char_dist"] for r in s_recs)
        
        s_b_wer = s_b_wd / max(s_rw, 1)
        s_i_wer = s_i_wd / max(s_rw, 1)
        s_b_cer = s_b_cd / max(s_rc, 1)
        s_i_cer = s_i_cd / max(s_rc, 1)
        
        source_dict[src] = {
            "samples": len(s_recs),
            "audio_hours": round(s_dur / 3600, 3),
            "ref_words": s_rw,
            "ref_chars": s_rc,
            "b01_wer": round(s_b_wer, 4),
            "indic_wer": round(s_i_wer, 4),
            "delta_wer": round(s_b_wer - s_i_wer, 4),
            "b01_cer": round(s_b_cer, 4),
            "indic_cer": round(s_i_cer, 4),
            "delta_cer": round(s_b_cer - s_i_cer, 4)
        }
        
        source_rows.append({
            "source": src,
            "sample_count": len(s_recs),
            "audio_hours": round(s_dur / 3600, 3),
            "ref_words": s_rw,
            "b01_normalized_wer": f"{s_b_wer*100:.2f}%",
            "indic_normalized_wer": f"{s_i_wer*100:.2f}%",
            "delta_wer": f"{(s_b_wer - s_i_wer)*100:+.2f}%",
            "b01_normalized_cer": f"{s_b_cer*100:.2f}%",
            "indic_normalized_cer": f"{s_i_cer*100:.2f}%",
            "delta_cer": f"{(s_b_cer - s_i_cer)*100:+.2f}%"
        })
        
    # Write source_breakdown.csv
    src_csv_path = STAGE6_DIR / "source_breakdown.csv"
    with open(src_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["source", "sample_count", "audio_hours", "ref_words", "b01_normalized_wer", "indic_normalized_wer", "delta_wer", "b01_normalized_cer", "indic_normalized_cer", "delta_cer"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in source_rows:
            w.writerow(row)
            
    test_metrics = {
        "evaluation_split": "Held-Out Test Set (Untouched)",
        "num_samples": N,
        "total_audio_hours": round(total_audio_sec / 3600, 4),
        "total_ref_words": int(total_ref_words),
        "total_ref_chars": int(total_ref_chars),
        "systems": {
            "b01_xlsr300m": {
                "name": "B-01: XLS-R 300M + Kashmiri Character CTC (Epoch 13 Fine-Tuned)",
                "normalized_wer": round(float(b01_norm_wer), 4),
                "wer_95_ci": [round(x, 4) for x in ci_b01_wer],
                "normalized_cer": round(float(b01_norm_cer), 4),
                "cer_95_ci": [round(x, 4) for x in ci_b01_cer],
                "words_substitutions": int(b01_words_s),
                "words_deletions": int(b01_words_d),
                "words_insertions": int(b01_words_i),
                "chars_substitutions": int(b01_chars_s),
                "chars_deletions": int(b01_chars_d),
                "chars_insertions": int(b01_chars_i),
                "exact_match_utterances": int(b01_exact_matches),
                "exact_match_pct": round(b01_exact_matches / N * 100, 2),
                "empty_hypotheses": int(b01_empty),
                "empty_hypotheses_pct": round(b01_empty / N * 100, 2)
            },
            "indicconformer_baseline": {
                "name": "A-01: IndicConformer (AI4Bharat Official Pretrained Baseline)",
                "normalized_wer": round(float(indic_norm_wer), 4),
                "wer_95_ci": [round(x, 4) for x in ci_indic_wer],
                "normalized_cer": round(float(indic_norm_cer), 4),
                "cer_95_ci": [round(x, 4) for x in ci_indic_cer],
                "words_substitutions": int(indic_words_s),
                "words_deletions": int(indic_words_d),
                "words_insertions": int(indic_words_i),
                "chars_substitutions": int(indic_chars_s),
                "chars_deletions": int(indic_chars_d),
                "chars_insertions": int(indic_chars_i),
                "exact_match_utterances": int(indic_exact_matches),
                "exact_match_pct": round(indic_exact_matches / N * 100, 2),
                "empty_hypotheses": int(indic_empty),
                "empty_hypotheses_pct": round(indic_empty / N * 100, 2)
            }
        },
        "statistical_comparison": {
            "delta_wer": round(float(delta_wer), 4),
            "delta_wer_95_ci": [round(x, 4) for x in ci_diff_wer],
            "p_value_wer": round(float(p_wer), 5),
            "speaker_clustered_delta_wer_95_ci": [round(x, 4) for x in ci_spk_diff_wer],
            "speaker_clustered_p_value_wer": round(float(p_spk_wer), 5),
            "delta_cer": round(float(delta_cer), 4),
            "delta_cer_95_ci": [round(x, 4) for x in ci_diff_cer],
            "p_value_cer": round(float(p_cer), 5)
        },
        "source_breakdown": source_dict
    }
    
    with open(STAGE6_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
        
    with open(STAGE6_DIR / "paired_bootstrap_test.json", "w", encoding="utf-8") as f:
        json.dump({
            "num_iterations": num_bootstrap,
            "random_seed": 42,
            "sample_count": N,
            "delta_wer": round(float(delta_wer), 4),
            "delta_wer_95_ci": [round(x, 4) for x in ci_diff_wer],
            "p_value_wer": round(float(p_wer), 5),
            "speaker_clustered_delta_wer_95_ci": [round(x, 4) for x in ci_spk_diff_wer],
            "speaker_clustered_p_value_wer": round(float(p_spk_wer), 5),
            "delta_cer": round(float(delta_cer), 4),
            "delta_cer_95_ci": [round(x, 4) for x in ci_diff_cer],
            "p_value_cer": round(float(p_cer), 5)
        }, f, indent=2)
        
    print(f"\nSaved test_metrics.json, paired_bootstrap_test.json, and source_breakdown.csv to {STAGE6_DIR}")

if __name__ == "__main__":
    run_held_out_analysis()
