#!/usr/bin/env python3
"""
Stage 22C Master Benchmark Reproduction & Statistical Validation Engine
Executes complete technical and scientific verification of T0, T1, T2, T3 on test.jsonl
under certified exact invariants and compares against Stage 22B historical predictions.
"""

import os, sys, json, time, csv, multiprocessing, concurrent.futures
import numpy as np
import soundfile as sf
import scipy.signal
import scipy.stats
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pyctcdecode import build_ctcdecoder
import jiwer

sys.path.insert(0, "/home/musa/Kashmiri Project")
from kashmiri_dataset.scripts.metrics_evaluator import normalize_perso_arabic_kashmiri as norm_func

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage22c_exact_output_lengths import compute_exact_ctc_length

# Global worker state for multiprocessing decoder pool
worker_decoder = None

def init_decoder_worker(labels, lm_path, unigrams, alpha, beta):
    global worker_decoder
    worker_decoder = build_ctcdecoder(
        labels=labels,
        kenlm_model_path=lm_path,
        unigrams=unigrams,
        alpha=alpha,
        beta=beta
    )

def decode_task(task):
    global worker_decoder
    lps, beam_width = task
    return worker_decoder.decode(lps, beam_width=beam_width)

def load_audio_waveform(path):
    if not os.path.isabs(path):
        path = os.path.join("/home/musa/Kashmiri Project", path)
    wav, sr = sf.read(path)
    if sr != 16000:
        num_samples = int(len(wav) * 16000 / sr)
        wav = scipy.signal.resample(wav, num_samples)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32)

def extract_long_audio_c2(model, proc, wav, device, core_len=18*16000, ctx_len=2*16000):
    total_samples = len(wav)
    all_central_lps = []
    
    for s_core in range(0, total_samples, core_len):
        e_core = min(s_core + core_len, total_samples)
        s_ctx = max(0, s_core - ctx_len)
        e_ctx = min(total_samples, e_core + ctx_len)
        
        chunk = wav[s_ctx:e_ctx]
        inp = proc(chunk, sampling_rate=16000, return_tensors="pt").input_values.to(device)
        with torch.no_grad():
            logits = model(inp).logits
            lps = torch.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            
        f_left = compute_exact_ctc_length(s_core - s_ctx) if (s_core - s_ctx) >= 400 else 0
        f_total = compute_exact_ctc_length(e_core - s_ctx)
        all_central_lps.append(lps[f_left:f_total, :])
        
    return np.concatenate(all_central_lps, axis=0)

def evaluate_certified_model_incremental(model_name, model_path, test_items, pool, device, out_jsonl_path,
                                         batch_size=32, max_batch_samples=480000, beam_width=64):
    existing_preds = {}
    if os.path.exists(out_jsonl_path):
        with open(out_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    existing_preds[rec["sample_id"]] = rec
        print(f"[{model_name}] Found {len(existing_preds)} existing cached predictions in {out_jsonl_path}")
        
    if len(existing_preds) == len(test_items):
        print(f"[{model_name}] All {len(test_items)} utterances already completed. Skipping acoustic inference!")
        return [existing_preds[x.get("sample_id") or x.get("id")] for x in test_items]
        
    remaining_indices = [i for i, x in enumerate(test_items) if (x.get("sample_id") or x.get("id")) not in existing_preds]
    print(f"[{model_name}] Processing {len(remaining_indices)}/{len(test_items)} utterances on {device}...")
    
    proc = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path).to(device)
    model.eval()
    
    f_out = open(out_jsonl_path, "a", encoding="utf-8")
    t0 = time.time()
    
    curr_idx = 0
    b_step = 0
    while curr_idx < len(remaining_indices):
        batch_indices = []
        batch_samples = 0
        while curr_idx < len(remaining_indices):
            idx = remaining_indices[curr_idx]
            item = test_items[idx]
            est_samples = int(item.get("duration", 4.0) * 16000)
            
            # Isolate any file > 25s
            if est_samples > 25 * 16000:
                if len(batch_indices) == 0:
                    batch_indices.append(idx)
                    curr_idx += 1
                break
                
            if len(batch_indices) > 0 and (len(batch_indices) >= batch_size or (batch_samples + est_samples) > max_batch_samples):
                break
                
            batch_indices.append(idx)
            batch_samples += est_samples
            curr_idx += 1
            
        batch = [test_items[i] for i in batch_indices]
        
        audio_paths = [x.get("audio") or x.get("audio_path") or x.get("audio_filepath") for x in batch]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            audio_list = list(ex.map(load_audio_waveform, audio_paths))
            
        lengths = [len(w) for w in audio_list]
        max_len = max(lengths)
        
        sliced_lps_list = []
        greedy_preds = []
        
        if max_len > 25 * 16000:
            for w in audio_list:
                if len(w) > 25 * 16000:
                    slps = extract_long_audio_c2(model, proc, w, device)
                else:
                    inp = proc(w, sampling_rate=16000, return_tensors="pt").input_values.to(device)
                    with torch.no_grad():
                        logits = model(inp).logits
                        lps = torch.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                    exact_t = compute_exact_ctc_length(len(w))
                    slps = lps[:exact_t, :]
                sliced_lps_list.append(slps)
                p_ids = np.argmax(slps, axis=-1)
                greedy_preds.append(proc.decode(p_ids))
        else:
            inputs = proc(audio_list, sampling_rate=16000, padding=True, return_tensors="pt")
            with torch.no_grad():
                logits = model(inputs.input_values.to(device)).logits
                log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
                
            del inputs
            del logits
            
            for b_i in range(len(batch)):
                exact_t = compute_exact_ctc_length(lengths[b_i])
                slps = log_probs[b_i, :exact_t, :]
                sliced_lps_list.append(slps)
                p_ids = np.argmax(slps, axis=-1)
                greedy_preds.append(proc.decode(p_ids))
                
            del log_probs
            
        # KenLM beam search
        kenlm_tasks = [(slps, beam_width) for slps in sliced_lps_list]
        kenlm_preds = pool.map(decode_task, kenlm_tasks)
        
        # Flush to disk immediately
        for b_i, x in enumerate(batch):
            sid = x.get("sample_id") or x.get("id")
            ref_txt = (x.get("normalized_text") or x.get("text") or "").strip()
            rec = {
                "sample_id": sid,
                "duration": x.get("duration", lengths[b_i] / 16000.0),
                "source": x.get("source"),
                "speaker_id": x.get("speaker_id", x.get("speaker", "unknown")),
                "reference": ref_txt,
                "greedy_pred": greedy_preds[b_i],
                "kenlm_pred": kenlm_preds[b_i]
            }
            existing_preds[sid] = rec
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f_out.flush()
        
        b_step += 1
        if b_step % 10 == 0:
            torch.cuda.empty_cache()
            
        curr_done = len(existing_preds)
        if b_step % 20 == 0 or curr_done == len(test_items):
            print(f"  [{model_name}] Progress: {curr_done}/{len(test_items)} ({curr_done/len(test_items)*100:.1f}%) in {time.time()-t0:.1f}s")
            
    f_out.close()
    del model
    del proc
    torch.cuda.empty_cache()
    print(f"[{model_name}] Finished all {len(test_items)} utterances!")
    return [existing_preds[x.get("sample_id") or x.get("id")] for x in test_items]

def compute_metrics(refs, preds):
    n_refs = [norm_func(r) for r in refs]
    n_preds = [norm_func(p) for p in preds]
    w = jiwer.process_words(n_refs, n_preds)
    c = jiwer.process_characters(n_refs, n_preds)
    return {
        "wer_pct": round(w.wer * 100.0, 2),
        "cer_pct": round(c.cer * 100.0, 2),
        "substitutions": w.substitutions,
        "deletions": w.deletions,
        "insertions": w.insertions,
        "hits": w.hits,
        "ref_words": w.hits + w.substitutions + w.deletions,
        "ref_chars": c.hits + c.substitutions + c.deletions
    }

def compute_utterance_errors(refs, preds):
    err_counts = []
    ref_lengths = []
    for r, p in zip(refs, preds):
        res = jiwer.process_words(norm_func(r), norm_func(p))
        errs = res.substitutions + res.deletions + res.insertions
        ref_l = res.hits + res.substitutions + res.deletions
        err_counts.append(errs)
        ref_lengths.append(ref_l)
    return np.array(err_counts, dtype=np.float32), np.array(ref_lengths, dtype=np.float32)

def paired_bootstrap(errs_a, errs_b, ref_lengths, n_boot=10000, seed=42):
    np.random.seed(seed)
    n = len(errs_a)
    diffs = []
    indices = np.random.randint(0, n, size=(n_boot, n))
    for i in range(n_boot):
        idx = indices[i]
        sum_ref = np.sum(ref_lengths[idx])
        wer_a = np.sum(errs_a[idx]) / sum_ref * 100.0
        wer_b = np.sum(errs_b[idx]) / sum_ref * 100.0
        diffs.append(wer_b - wer_a)
    diffs = np.array(diffs)
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    mean_diff = np.mean(diffs)
    if mean_diff < 0:
        p_val = np.mean(diffs >= 0)
    else:
        p_val = np.mean(diffs <= 0)
    return round(float(mean_diff), 3), round(float(ci_lower), 3), round(float(ci_upper), 3), float(p_val)

def main():
    root = "/home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage22c_inference_certification"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Stage 22C Master Benchmark Reproduction Engine on {device} ===")
    
    test_manifest_path = "/home/musa/Kashmiri Project/kashmiri_dataset/experiments/native_asr_training/data_splits/test.jsonl"
    with open(test_manifest_path, "r", encoding="utf-8") as f:
        test_items = [json.loads(line) for line in f if line.strip()]
    n_total = len(test_items)
    print(f"Loaded held-out test manifest: {n_total} utterances.")
    
    e03_path = "/home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage15_mwer_verification_and_scaling/checkpoints/E-03/best_calib"
    e22_path = "/home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage22_data_expanded_training/checkpoints/E22-01/best_model"
    
    lm_path = "/home/musa/Kashmiri Project/kashmiri_dataset/experiments/native_asr_training/experiments/stage7_ctc_lm_decoding/language_models/lm_word_4gram_domain_d.arpa"
    unigrams_path = "/home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage22b_decoder_and_independent_test/language_model/unigrams.txt"
    with open(unigrams_path, "r", encoding="utf-8") as f:
        unigrams = [line.strip() for line in f if line.strip()]
        
    proc_e22 = Wav2Vec2Processor.from_pretrained(e22_path)
    sorted_vocab = sorted(proc_e22.tokenizer.get_vocab().items(), key=lambda x: x[1])
    labels = [token if token != "|" else " " for token, idx in sorted_vocab]
    
    num_workers = 4
    print(f"Initializing {num_workers}-worker KenLM decoder pool (alpha=0.30, beta=-1.00, beam=64)...")
    pool = multiprocessing.Pool(
        num_workers,
        initializer=init_decoder_worker,
        initargs=(labels, lm_path, unigrams, 0.30, -1.00)
    )
    
    # 1. Evaluate E-03 under Certified Exact Invariants
    e03_out_path = os.path.join(root, "predictions/e03_certified_predictions.jsonl")
    e03_results = evaluate_certified_model_incremental("E-03", e03_path, test_items, pool, device, e03_out_path)
    
    # 2. Evaluate E22-01 under Certified Exact Invariants
    e22_out_path = os.path.join(root, "predictions/e22_certified_predictions.jsonl")
    e22_results = evaluate_certified_model_incremental("E22-01", e22_path, test_items, pool, device, e22_out_path)
    
    pool.close()
    pool.join()
    
    refs = [x["reference"] for x in e03_results]
    t0_preds = [x["greedy_pred"] for x in e03_results]
    t2_preds = [x["kenlm_pred"] for x in e03_results]
    t1_preds = [x["greedy_pred"] for x in e22_results]
    t3_preds = [x["kenlm_pred"] for x in e22_results]
    
    # Save combined predictions
    combined_out_path = os.path.join(root, "predictions/stage22c_final_predictions.jsonl")
    with open(combined_out_path, "w", encoding="utf-8") as f_out:
        for idx in range(n_total):
            rec = {
                "sample_id": test_items[idx].get("sample_id") or test_items[idx].get("id"),
                "duration": test_items[idx].get("duration"),
                "source": test_items[idx].get("source"),
                "speaker_id": test_items[idx].get("speaker_id", "unknown"),
                "reference": refs[idx],
                "t0_e03_greedy": t0_preds[idx],
                "t1_e22_greedy": t1_preds[idx],
                "t2_e03_kenlm": t2_preds[idx],
                "t3_e22_kenlm": t3_preds[idx]
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nSaved certified four-way predictions to {combined_out_path}")
    
    # Partitions for reporting
    legacy_core_idx = [i for i, x in enumerate(test_items) if x.get("source") is None]
    long_audio_idx = [i for i, x in enumerate(test_items) if x.get("duration", 0.0) > 25.0]
    short_audio_idx = [i for i, x in enumerate(test_items) if x.get("duration", 0.0) <= 25.0]
    
    source_subsets = {}
    for i, x in enumerate(test_items):
        src = x.get("source") or ("indicvoices" if "iv_k" in (x.get("sample_id") or x.get("id")) else ("bolbosh" if "bolb" in (x.get("sample_id") or x.get("id")) else "whisper"))
        source_subsets.setdefault(src, []).append(i)
        
    def get_eval_dict(p_name, idx_list):
        r_sub = [refs[i] for i in idx_list]
        return {
            "partition": p_name,
            "num_utts": len(idx_list),
            "ref_words": sum(len(r.split()) for r in r_sub),
            "t0_m": compute_metrics(r_sub, [t0_preds[i] for i in idx_list]),
            "t1_m": compute_metrics(r_sub, [t1_preds[i] for i in idx_list]),
            "t2_m": compute_metrics(r_sub, [t2_preds[i] for i in idx_list]),
            "t3_m": compute_metrics(r_sub, [t3_preds[i] for i in idx_list])
        }
        
    eval_partitions = [
        ("Full Independent Test (5,923 utts)", list(range(n_total))),
        ("Legacy Core Test (5,251 utts)", legacy_core_idx),
        ("Long-Audio Subset (> 25s, 13 utts)", long_audio_idx),
        ("Short-Audio Subset (<= 25s, 5,910 utts)", short_audio_idx)
    ]
    for src_name, s_idx in sorted(source_subsets.items()):
        eval_partitions.append((f"Source: {src_name}", s_idx))
        
    results_rows = []
    print("\n======================= CERTIFIED TEST BENCHMARKS =======================")
    for p_name, idx_list in eval_partitions:
        d = get_eval_dict(p_name, idx_list)
        print(f"\n--- {p_name} ({d['num_utts']} utts, {d['ref_words']} words) ---")
        for sys_code, sys_name, dec_name, m_data in [
            ("T0", "Stage 15 E-03 Baseline", "Greedy CTC", d["t0_m"]),
            ("T1", "Stage 22 E22-01 Control", "Greedy CTC", d["t1_m"]),
            ("T2", "Stage 15 E-03 Baseline", "KenLM 4-gram", d["t2_m"]),
            ("T3", "Stage 22 E22-01 Champion", "KenLM 4-gram", d["t3_m"])
        ]:
            print(f"  {sys_code:2s} ({sys_name:24s} | {dec_name:12s}): WER: {m_data['wer_pct']:5.2f}% | CER: {m_data['cer_pct']:5.2f}% | Subs: {m_data['substitutions']} | Dels: {m_data['deletions']} | Ins: {m_data['insertions']}")
            results_rows.append({
                "partition": p_name,
                "system_id": sys_code,
                "name": sys_name,
                "decoding": dec_name,
                "wer_pct": m_data["wer_pct"],
                "cer_pct": m_data["cer_pct"],
                "substitutions": m_data["substitutions"],
                "deletions": m_data["deletions"],
                "insertions": m_data["insertions"],
                "hits": m_data["hits"],
                "ref_words": m_data["ref_words"]
            })
            
    csv_results_path = os.path.join(root, "reports/stage22c_reproduced_test_results.csv")
    with open(csv_results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results_rows[0].keys()))
        writer.writeheader()
        writer.writerows(results_rows)
    print(f"\nSaved test results CSV to {csv_results_path}")
    
    # Statistical Analysis
    print("\nRunning Statistical Significance Analysis (10,000 Paired Bootstrap Resamples)...")
    errs_t0, ref_lens = compute_utterance_errors(refs, t0_preds)
    errs_t1, _ = compute_utterance_errors(refs, t1_preds)
    errs_t2, _ = compute_utterance_errors(refs, t2_preds)
    errs_t3, _ = compute_utterance_errors(refs, t3_preds)
    
    safe_ref_lens = np.maximum(ref_lens, 1.0)
    u_wer_t0 = errs_t0 / safe_ref_lens
    u_wer_t1 = errs_t1 / safe_ref_lens
    u_wer_t2 = errs_t2 / safe_ref_lens
    u_wer_t3 = errs_t3 / safe_ref_lens
    
    comparisons = [
        ("Acoustic Fine-Tuning Gain", "T1 vs T0", errs_t0, errs_t1, u_wer_t0, u_wer_t1),
        ("LM Benefit on Baseline", "T2 vs T0", errs_t0, errs_t2, u_wer_t0, u_wer_t2),
        ("LM Benefit on Champion", "T3 vs T1", errs_t1, errs_t3, u_wer_t1, u_wer_t3),
        ("Overall System Advancement", "T3 vs T0", errs_t0, errs_t3, u_wer_t0, u_wer_t3),
        ("Matched Calibrated Advancement", "T3 vs T2", errs_t2, errs_t3, u_wer_t2, u_wer_t3)
    ]
    
    stat_report = {}
    for comp_name, pair_code, e_base, e_cand, u_base, u_cand in comparisons:
        mean_d, ci_l, ci_u, boot_p = paired_bootstrap(e_base, e_cand, ref_lens, n_boot=10000)
        
        diff = u_cand - u_base
        nonzero_diff = diff[diff != 0]
        if len(nonzero_diff) > 0:
            w_stat, wilc_p = scipy.stats.wilcoxon(nonzero_diff, alternative="two-sided")
        else:
            w_stat, wilc_p = 0.0, 1.0
            
        total_ref = np.sum(ref_lens)
        agg_delta = (np.sum(e_cand) - np.sum(e_base)) / total_ref * 100.0
        
        stat_report[pair_code] = {
            "comparison_name": comp_name,
            "pair": pair_code,
            "aggregate_delta_wer_pp": round(float(agg_delta), 2),
            "bootstrap_mean_delta_pp": mean_d,
            "ci_95_lower": ci_l,
            "ci_95_upper": ci_u,
            "bootstrap_p_value": boot_p,
            "wilcoxon_stat": float(w_stat),
            "wilcoxon_p_value": float(wilc_p),
            "statistically_significant": bool(boot_p < 0.001 or wilc_p < 0.001)
        }
        print(f"  {comp_name:32s} ({pair_code}): dWER = {agg_delta:+5.2f} pp | 95% CI: [{ci_l:+5.2f}, {ci_u:+5.2f}] | Boot p={boot_p:.4f} | Wilc p={wilc_p:.2e} | Sig: {stat_report[pair_code]['statistically_significant']}")
        
    stat_out_path = os.path.join(root, "statistics/stage22c_statistical_analysis.json")
    with open(stat_out_path, "w", encoding="utf-8") as f:
        json.dump(stat_report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved statistical analysis to {stat_out_path}")
    
    # 4. Compare with Stage 22B Historical Predictions
    hist_path = "/home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage22b_decoder_and_independent_test/final_test/stage22b_final_test_predictions.jsonl"
    if os.path.exists(hist_path):
        with open(hist_path, "r", encoding="utf-8") as f:
            hist_recs = [json.loads(line) for line in f if line.strip()]
            
        diff_records = []
        for idx in range(n_total):
            h = hist_recs[idx]
            sid = test_items[idx].get("sample_id") or test_items[idx].get("id")
            
            # Check T3 differences
            p_hist_t3 = norm_func(h.get("t3_e22_calibrated_kenlm", ""))
            p_cert_t3 = norm_func(t3_preds[idx])
            
            p_hist_t0 = norm_func(h.get("t0_e03_greedy", ""))
            p_cert_t0 = norm_func(t0_preds[idx])
            
            if p_hist_t3 != p_cert_t3 or p_hist_t0 != p_cert_t0:
                diff_records.append({
                    "sample_id": sid,
                    "duration_s": test_items[idx].get("duration"),
                    "source": test_items[idx].get("source"),
                    "t0_changed": (p_hist_t0 != p_cert_t0),
                    "t3_changed": (p_hist_t3 != p_cert_t3),
                    "hist_t3_pred": p_hist_t3[:60],
                    "cert_t3_pred": p_cert_t3[:60],
                    "diff_cause": "Long-audio C2 context" if (test_items[idx].get("duration", 0.0) > 25.0) else "Exact Conv1d frame length correction"
                })
                
        diff_csv_path = os.path.join(root, "statistics/stage22c_paired_prediction_differences.csv")
        if diff_records:
            with open(diff_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(diff_records[0].keys()))
                writer.writeheader()
                writer.writerows(diff_records)
            print(f"Recorded {len(diff_records)} prediction differences vs Stage 22B to {diff_csv_path}")
        else:
            print("Zero prediction differences vs Stage 22B! Exact bitwise replication achieved.")

if __name__ == "__main__":
    main()
