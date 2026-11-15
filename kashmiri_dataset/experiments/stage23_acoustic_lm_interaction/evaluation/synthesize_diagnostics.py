#!/usr/bin/env python3
"""
Stage 23 Diagnostic Synthesis Script
Computes Phase B (Posterior Analysis), Phase C (Matched N-best Analysis),
and Phase D (Error Transition Analysis) from the generated prediction files.
"""

import os, sys, json, math, csv
import numpy as np
from scipy import stats
import jiwer

sys.path.insert(0, "/home/musa/Kashmiri Project")
from kashmiri_dataset.scripts.metrics_evaluator import normalize_perso_arabic_kashmiri as norm_func

WORKSPACE = "/home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage23_acoustic_lm_interaction"
EVAL_DIR = os.path.join(WORKSPACE, "evaluation")
POST_DIR = os.path.join(WORKSPACE, "posterior_analysis")
NBEST_DIR = os.path.join(WORKSPACE, "nbest")
TRANS_DIR = os.path.join(WORKSPACE, "error_transitions")
REP_DIR = os.path.join(WORKSPACE, "reports")

def load_preds(path):
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records[r["sample_id"]] = r
    return records

def compute_corpus_wer_cer(preds, key_hyp, key_ref="ref_norm"):
    refs = [r[key_ref] for r in preds]
    hyps = [r[key_hyp] for r in preds]
    total_ref_words = sum(len(r.split()) for r in refs)
    total_ref_chars = sum(len(r) for r in refs)
    
    # Using jiwer corpus-level aggregation
    wer = jiwer.wer(refs, hyps)
    cer = jiwer.cer(refs, hyps)
    return round(wer * 100, 2), round(cer * 100, 2), total_ref_words, total_ref_chars

def run_phase_b_posteriors(e03_preds, e2201_preds):
    print("\n--- Running Phase B: Acoustic Posterior Analysis ---")
    common_ids = sorted(set(e03_preds.keys()) & set(e2201_preds.keys()))
    print(f"Analyzing {len(common_ids)} matched calibration utterances...")
    
    metrics = ["frame_entropy", "top1_prob", "top2_margin", "blank_p_mean", "blank_p_std", "blank_p_median", "spike_density"]
    
    e03_vals = {m: [] for m in metrics}
    e2201_vals = {m: [] for m in metrics}
    
    ref_scoring_rows = []
    
    for sid in common_ids:
        r03 = e03_preds[sid]
        r22 = e2201_preds[sid]
        
        p03 = r03["posterior_stats"]
        p22 = r22["posterior_stats"]
        
        for m in metrics:
            e03_vals[m].append(p03[m])
            e2201_vals[m].append(p22[m])
            
        c03 = r03["ctc_forward_scores"]
        c22 = r22["ctc_forward_scores"]
        
        ref_logp_03 = c03["ref_ctc_logp"]
        ref_logp_22 = c22["ref_ctc_logp"]
        greedy_logp_03 = c03["greedy_ctc_logp"]
        greedy_logp_22 = c22["greedy_ctc_logp"]
        m03 = c03["ref_vs_greedy_margin"]
        m22 = c22["ref_vs_greedy_margin"]
        
        delta_ref = (ref_logp_22 - ref_logp_03) if (ref_logp_22 is not None and ref_logp_03 is not None) else None
        delta_margin = (m22 - m03) if (m22 is not None and m03 is not None) else None
        
        ref_scoring_rows.append({
            "sample_id": sid,
            "duration": r03["duration"],
            "e03_ref_logp": ref_logp_03,
            "e2201_ref_logp": ref_logp_22,
            "delta_ref_logp": round(delta_ref, 4) if delta_ref is not None else None,
            "e03_greedy_logp": greedy_logp_03,
            "e2201_greedy_logp": greedy_logp_22,
            "e03_margin": m03,
            "e2201_margin": m22,
            "delta_margin": round(delta_margin, 4) if delta_margin is not None else None
        })
        
    # Write stage23_ctc_reference_scoring.csv
    ref_csv_path = os.path.join(POST_DIR, "stage23_ctc_reference_scoring.csv")
    with open(ref_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ref_scoring_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ref_scoring_rows)
    print(f"Saved {ref_csv_path}")
    
    # Summary of posterior metrics
    post_summary_rows = []
    for m in metrics:
        v03 = np.array(e03_vals[m])
        v22 = np.array(e2201_vals[m])
        diff = v22 - v03
        t_stat, p_val = stats.ttest_rel(v22, v03)
        post_summary_rows.append({
            "metric": m,
            "e03_mean": round(float(np.mean(v03)), 4),
            "e03_std": round(float(np.std(v03)), 4),
            "e2201_mean": round(float(np.mean(v22)), 4),
            "e2201_std": round(float(np.std(v22)), 4),
            "delta": round(float(np.mean(diff)), 4),
            "pct_change": round(float(np.mean(diff) / np.mean(v03) * 100), 2),
            "p_value": f"{p_val:.4e}" if p_val < 0.0001 else round(float(p_val), 4)
        })
        
    post_csv_path = os.path.join(POST_DIR, "stage23_posterior_comparison.csv")
    with open(post_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(post_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(post_summary_rows)
    print(f"Saved {post_csv_path}")
    
    # Valid ref score delta analysis
    valid_deltas = [r["delta_ref_logp"] for r in ref_scoring_rows if r["delta_ref_logp"] is not None]
    valid_margin_deltas = [r["delta_margin"] for r in ref_scoring_rows if r["delta_margin"] is not None]
    pct_ref_improved = sum(1 for d in valid_deltas if d > 0) / len(valid_deltas) * 100 if valid_deltas else 0.0
    pct_margin_improved = sum(1 for d in valid_margin_deltas if d > 0) / len(valid_margin_deltas) * 100 if valid_margin_deltas else 0.0
    
    print(f"Reference CTC Log-Likelihood improved in {pct_ref_improved:.1f}% of utterances (mean delta: {np.mean(valid_deltas):.2f}).")
    print(f"Reference-vs-Greedy Margin improved in {pct_margin_improved:.1f}% of utterances (mean delta: {np.mean(valid_margin_deltas):.2f}).")
    
    return post_summary_rows, pct_ref_improved, pct_margin_improved, np.mean(valid_deltas)

def run_phase_c_nbest(e03_preds, e2201_preds):
    print("\n--- Running Phase C: Matched N-Best Analysis ---")
    common_ids = sorted(set(e03_preds.keys()) & set(e2201_preds.keys()))
    
    models = ["E-03", "E22-01"]
    preds_map = {"E-03": e03_preds, "E22-01": e2201_preds}
    
    nbest_summary = []
    
    for m in models:
        p_dict = preds_map[m]
        items = [p_dict[sid] for sid in common_ids]
        
        top1_wers = [it["kenlm_wer"] for it in items]
        o4_wers = [it["oracle_4_wer"] for it in items]
        o8_wers = [it["oracle_8_wer"] for it in items]
        o16_wers = [it["oracle_16_wer"] for it in items]
        o32_wers = [it["oracle_32_wer"] for it in items]
        o64_wers = [it["oracle_64_wer"] for it in items]
        
        exact_ref_count = sum(1 for it in items if it["has_exact_ref"])
        better_cand_count = sum(1 for it in items if it["better_candidate_available"])
        
        # Candidate diversity (average distinct candidates per beam)
        diversities = []
        for it in items:
            distinct = len(set(b["norm_text"] for b in it.get("nbest", [])))
            diversities.append(distinct)
            
        nbest_summary.append({
            "model": m,
            "top1_wer": round(float(np.mean(top1_wers)) * 100, 2),
            "oracle_4_wer": round(float(np.mean(o4_wers)) * 100, 2),
            "oracle_8_wer": round(float(np.mean(o8_wers)) * 100, 2),
            "oracle_16_wer": round(float(np.mean(o16_wers)) * 100, 2),
            "oracle_32_wer": round(float(np.mean(o32_wers)) * 100, 2),
            "oracle_64_wer": round(float(np.mean(o64_wers)) * 100, 2),
            "exact_ref_inclusion_pct": round(exact_ref_count / len(items) * 100, 2),
            "better_candidate_avail_pct": round(better_cand_count / len(items) * 100, 2),
            "avg_candidate_diversity": round(float(np.mean(diversities)), 2)
        })
        
    nbest_csv = os.path.join(NBEST_DIR, "stage23_nbest_candidate_comparison.csv")
    with open(nbest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(nbest_summary[0].keys()))
        writer.writeheader()
        writer.writerows(nbest_summary)
    print(f"Saved {nbest_csv}")
    
    # Overlap and intersection analysis
    overlap_records = []
    jaccards = []
    shared_ranks = []
    
    cat_counts = {
        "correct_only_in_e03": 0,
        "correct_only_in_e2201": 0,
        "correct_in_both": 0,
        "correct_in_neither": 0,
        "top_identical": 0,
        "top_diff_pools_overlap": 0
    }
    
    for sid in common_ids:
        r03 = e03_preds[sid]
        r22 = e2201_preds[sid]
        ref = r03["ref_norm"]
        
        cands03 = [b["norm_text"] for b in r03.get("nbest", [])]
        cands22 = [b["norm_text"] for b in r22.get("nbest", [])]
        set03 = set(cands03)
        set22 = set(cands22)
        
        inter = set03 & set22
        union = set03 | set22
        jaccard = len(inter) / max(len(union), 1)
        jaccards.append(jaccard)
        
        has_ref_03 = ref in set03 and len(ref) > 0
        has_ref_22 = ref in set22 and len(ref) > 0
        
        if has_ref_03 and not has_ref_22:
            cat = "correct_only_in_e03"
        elif has_ref_22 and not has_ref_03:
            cat = "correct_only_in_e2201"
        elif has_ref_03 and has_ref_22:
            cat = "correct_in_both"
        else:
            cat = "correct_in_neither"
        cat_counts[cat] += 1
        
        top03 = r03["kenlm_norm"]
        top22 = r22["kenlm_norm"]
        if top03 == top22:
            cat_counts["top_identical"] += 1
        else:
            cat_counts["top_diff_pools_overlap"] += 1
            
        overlap_records.append({
            "sample_id": sid,
            "ref_norm": ref,
            "jaccard_overlap": round(jaccard, 4),
            "common_candidate_count": len(inter),
            "e03_candidate_count": len(set03),
            "e2201_candidate_count": len(set22),
            "category": cat,
            "top_identical": (top03 == top22)
        })
        
    overlap_jsonl = os.path.join(NBEST_DIR, "stage23_candidate_overlap.jsonl")
    with open(overlap_jsonl, "w", encoding="utf-8") as f:
        for r in overlap_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {overlap_jsonl}")
    print(f"Average Candidate Pool Jaccard Overlap: {np.mean(jaccards)*100:.2f}%")
    print(f"Candidate Categories: {cat_counts}")
    
    return nbest_summary, cat_counts, np.mean(jaccards)

def run_phase_d_transitions(e03_preds, e2201_preds):
    print("\n--- Running Phase D: Error Transition Analysis ---")
    common_ids = sorted(set(e03_preds.keys()) & set(e2201_preds.keys()))
    
    # 7 Transition classes
    # 1. both_greedy_wrong_e2201_kenlm_correct
    # 2. e03_greedy_correct_e2201_greedy_wrong
    # 3. e2201_greedy_correct_e03_greedy_wrong
    # 4. both_greedy_identical_kenlm_differs
    # 5. both_have_correct_cand_select_different
    # 6. e2201_discovers_new_better_candidate
    # 7. e2201_loses_better_candidate
    
    class_stats = {
        "1_both_greedy_wrong_e2201_kenlm_correct": {"utts": 0, "ref_words": 0, "delta_errs": 0, "examples": []},
        "2_e03_greedy_correct_e2201_greedy_wrong": {"utts": 0, "ref_words": 0, "delta_errs": 0, "examples": []},
        "3_e2201_greedy_correct_e03_greedy_wrong": {"utts": 0, "ref_words": 0, "delta_errs": 0, "examples": []},
        "4_both_greedy_identical_kenlm_differs": {"utts": 0, "ref_words": 0, "delta_errs": 0, "examples": []},
        "5_both_have_correct_cand_select_diff": {"utts": 0, "ref_words": 0, "delta_errs": 0, "examples": []},
        "6_e2201_discovers_new_better_cand": {"utts": 0, "ref_words": 0, "delta_errs": 0, "examples": []},
        "7_e2201_loses_better_cand": {"utts": 0, "ref_words": 0, "delta_errs": 0, "examples": []}
    }
    
    win_loss_records = []
    
    for sid in common_ids:
        r03 = e03_preds[sid]
        r22 = e2201_preds[sid]
        ref = r03["ref_norm"]
        n_words = max(len(ref.split()), 1)
        
        g03 = r03["greedy_norm"]
        g22 = r22["greedy_norm"]
        k03 = r03["kenlm_norm"]
        k22 = r22["kenlm_norm"]
        
        # Word errors
        err_g03 = round(r03["greedy_wer"] * n_words)
        err_g22 = round(r22["greedy_wer"] * n_words)
        err_k03 = round(r03["kenlm_wer"] * n_words)
        err_k22 = round(r22["kenlm_wer"] * n_words)
        
        # Delta errors for KenLM: negative means E22-01 made FEWER errors (win)
        net_delta = err_k22 - err_k03
        
        cands03 = set(b["norm_text"] for b in r03.get("nbest", []))
        cands22 = set(b["norm_text"] for b in r22.get("nbest", []))
        has_ref_03 = (ref in cands03 and len(ref) > 0)
        has_ref_22 = (ref in cands22 and len(ref) > 0)
        
        assigned_classes = []
        
        # 1. both greedy wrong, e2201 kenlm correct
        if g03 != ref and g22 != ref and k22 == ref:
            assigned_classes.append("1_both_greedy_wrong_e2201_kenlm_correct")
            
        # 2. e03 greedy correct, e2201 greedy wrong
        if g03 == ref and g22 != ref:
            assigned_classes.append("2_e03_greedy_correct_e2201_greedy_wrong")
            
        # 3. e2201 greedy correct, e03 greedy wrong
        if g22 == ref and g03 != ref:
            assigned_classes.append("3_e2201_greedy_correct_e03_greedy_wrong")
            
        # 4. both greedy identical but kenlm outputs differ
        if g03 == g22 and k03 != k22:
            assigned_classes.append("4_both_greedy_identical_kenlm_differs")
            
        # 5. both have correct candidate, but select different outputs
        if has_ref_03 and has_ref_22 and k03 != k22:
            assigned_classes.append("5_both_have_correct_cand_select_diff")
            
        # 6. e2201 discovers new better candidate
        min_err_03 = min(b["wer"] for b in r03.get("nbest", [])) if r03.get("nbest") else 1.0
        min_err_22 = min(b["wer"] for b in r22.get("nbest", [])) if r22.get("nbest") else 1.0
        if min_err_22 < min_err_03:
            assigned_classes.append("6_e2201_discovers_new_better_cand")
            
        # 7. e2201 loses better candidate
        if min_err_22 > min_err_03:
            assigned_classes.append("7_e2201_loses_better_cand")
            
        for c in assigned_classes:
            class_stats[c]["utts"] += 1
            class_stats[c]["ref_words"] += n_words
            class_stats[c]["delta_errs"] += net_delta
            if len(class_stats[c]["examples"]) < 3:
                class_stats[c]["examples"].append({
                    "sample_id": sid,
                    "ref": ref,
                    "g03": g03, "g22": g22,
                    "k03": k03, "k22": k22,
                    "net_delta": net_delta
                })
                
        win_loss_records.append({
            "sample_id": sid,
            "duration": r03["duration"],
            "source": r03["source"],
            "ref": ref,
            "g03": g03, "g22": g22,
            "k03": k03, "k22": k22,
            "err_k03": err_k03, "err_k22": err_k22,
            "net_delta_errs": net_delta,
            "assigned_classes": assigned_classes
        })
        
    # Write stage23_decoder_win_loss_atlas.jsonl
    atlas_path = os.path.join(TRANS_DIR, "stage23_decoder_win_loss_atlas.jsonl")
    with open(atlas_path, "w", encoding="utf-8") as f:
        for r in win_loss_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {atlas_path}")
    
    # Write stage23_error_transition_matrix.csv
    trans_rows = []
    for c, st in class_stats.items():
        trans_rows.append({
            "transition_class": c,
            "utterance_count": st["utts"],
            "ref_words": st["ref_words"],
            "net_word_error_delta": st["delta_errs"],
            "avg_wer_delta_pp": round(st["delta_errs"] / max(st["ref_words"], 1) * 100, 3)
        })
    trans_csv = os.path.join(TRANS_DIR, "stage23_error_transition_matrix.csv")
    with open(trans_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(trans_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trans_rows)
    print(f"Saved {trans_csv}")
    
    for r in trans_rows:
        print(f"  {r['transition_class']}: {r['utterance_count']} utts, net error delta: {r['net_word_error_delta']}")
        
    return trans_rows, class_stats

def main():
    e03_path = os.path.join(EVAL_DIR, "dev_calibration_e03_predictions.jsonl")
    e2201_path = os.path.join(EVAL_DIR, "dev_calibration_e2201_predictions.jsonl")
    
    if not (os.path.exists(e03_path) and os.path.exists(e2201_path)):
        print("Prediction files not ready yet.")
        return
        
    e03_preds = load_preds(e03_path)
    e2201_preds = load_preds(e2201_path)
    
    print(f"Loaded {len(e03_preds)} E-03 records and {len(e2201_preds)} E22-01 records.")
    
    run_phase_b_posteriors(e03_preds, e2201_preds)
    run_phase_c_nbest(e03_preds, e2201_preds)
    run_phase_d_transitions(e03_preds, e2201_preds)

if __name__ == "__main__":
    main()
