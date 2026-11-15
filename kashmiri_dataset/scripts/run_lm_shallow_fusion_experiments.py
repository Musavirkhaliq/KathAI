#!/usr/bin/env python3
"""
Workstream 8 Stage 7 Phase F: CTC Prefix Beam Search + Kashmiri Language Model Shallow Fusion
Executes coarse-to-fine hyperparameter search over alpha, beta, and beam width on validation logit cache.
Evaluates the complete Ablation Matrix D-00 through D-09 on all 6,728 validation samples.
Outputs ablation metrics, hyperparameter grid results, best decoder config, and predictions.
"""

import json
import time
import sys
import gc
import csv
from pathlib import Path
import numpy as np
import jiwer
import yaml
from transformers import Wav2Vec2Processor
from pyctcdecode import build_ctcdecoder

sys.path.append("kashmiri_dataset/scripts")
from metrics_evaluator import normalize_perso_arabic_kashmiri

# Directories
STAGE7_DIR = Path("kashmiri_dataset/experiments/native_asr_training/experiments/stage7_ctc_lm_decoding")
CACHE_DIR = STAGE7_DIR / "logit_cache"
MANIFEST_PATH = STAGE7_DIR / "cache_manifest.json"
LM_DIR = STAGE7_DIR / "language_models"
REPORTS_DIR = STAGE7_DIR / "reports"
CONFIGS_DIR = STAGE7_DIR / "configs"
PRED_DIR = STAGE7_DIR / "predictions"
VAL_SPLIT_PATH = Path("kashmiri_dataset/experiments/native_asr_training/data_splits/val.jsonl")
MODEL_DIR = Path("kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc/best_model")

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

print("=== Stage 7 Phase F: CTC Prefix Beam Search + LM Shallow Fusion ===")

# 1. Load Processor and Labels
processor = Wav2Vec2Processor.from_pretrained(str(MODEL_DIR))
vocab = processor.tokenizer.get_vocab()
sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
labels = [tok for tok, idx in sorted_vocab]
labels_for_pyctc = [tok if tok != '|' else ' ' for tok in labels]

# 2. Load Manifest & Ground Truth Transcripts
with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    cache_manifest = json.load(f)

val_metadata = {}
with open(VAL_SPLIT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        val_metadata[item["sample_id"]] = item

samples_list = list(cache_manifest["samples"].keys())
total_samples = len(samples_list)
print(f"Loaded {total_samples} samples from cache manifest.")

# Preload ground truth references
raw_refs = []
norm_refs = []
durations = []
sources = []
for sid in samples_list:
    meta = val_metadata[sid]
    ref = meta.get("text", "").strip()
    norm_ref = normalize_perso_arabic_kashmiri(ref)
    raw_refs.append(ref)
    norm_refs.append(norm_ref)
    durations.append(meta.get("duration", 0.0))
    sources.append(meta.get("source") or meta.get("dataset_source") or "unknown")

total_audio_sec = sum(durations)

# Preload all cached log-probs
print("Preloading cached log-probabilities into RAM...")
t_load_start = time.time()
cached_lps = []
for sid in samples_list:
    npz_file = CACHE_DIR / f"{sid}.npz"
    with np.load(npz_file) as data:
        cached_lps.append(data["log_probs"].astype(np.float32))
print(f"Loaded {len(cached_lps)} arrays in {time.time()-t_load_start:.2f}s")

# 3. Stage 1: Fast Hyperparameter Grid Search on 500-sample Dev Subset
dev_n = 500
dev_lps = cached_lps[:dev_n]
dev_norm_refs = norm_refs[:dev_n]

print(f"\n--- Running Coarse Grid Search on {dev_n} Dev Samples with Conversational LM (LM-B) ---")
lm_b_path = str(LM_DIR / "lm_word_3gram_domain_b.arpa")

alphas = [0.25, 0.50, 0.75, 1.00, 1.25]
betas = [-0.5, 0.0, 0.5, 1.0, 1.5]
beam_widths_search = [8, 16]

grid_search_rows = []
best_dev_wer = float("inf")
best_params = {"alpha": 0.5, "beta": 0.5, "beam_width": 16}

t_grid_start = time.time()
for bw in beam_widths_search:
    for alpha in alphas:
        for beta in betas:
            dec = build_ctcdecoder(
                labels=labels_for_pyctc,
                kenlm_model_path=lm_b_path,
                alpha=alpha,
                beta=beta
            )
            hyps = []
            for lp in dev_lps:
                hyps.append(normalize_perso_arabic_kashmiri(dec.decode(lp, beam_width=bw)))
            
            wer_res = jiwer.process_words(dev_norm_refs, hyps)
            dev_wer = wer_res.wer * 100.0
            
            grid_search_rows.append({
                "lm_model": "lm_word_3gram_domain_b",
                "beam_width": bw,
                "alpha": alpha,
                "beta": beta,
                "dev_wer": round(dev_wer, 2)
            })
            
            if dev_wer < best_dev_wer:
                best_dev_wer = dev_wer
                best_params = {"alpha": alpha, "beta": beta, "beam_width": bw}
                print(f"New Best Grid: BW={bw}, alpha={alpha:.2f}, beta={beta:.2f} -> Dev WER = {dev_wer:.2f}%")

print(f"Grid search complete in {time.time()-t_grid_start:.1f}s. Best parameters: {best_params} (Dev WER: {best_dev_wer:.2f}%)")

# Save Grid Search CSV
grid_csv_path = REPORTS_DIR / "decoder_hyperparameter_search.csv"
with open(grid_csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["lm_model", "beam_width", "alpha", "beta", "dev_wer"])
    writer.writeheader()
    writer.writerows(grid_search_rows)

# 4. Stage 2: Evaluate Complete Ablation Matrix D-00 to D-09 on Full 6,728 Validation Set
print("\n=== Evaluating Complete Ablation Matrix D-00 to D-09 on All 6,728 Utterances ===")

ablations = [
    {
        "id": "D-00",
        "name": "Pure CTC Greedy Baseline",
        "lm_path": None,
        "is_char": False,
        "alpha": 0.0,
        "beta": 0.0,
        "beam_width": 1
    },
    {
        "id": "D-01",
        "name": "Pure CTC Prefix Beam Search (No LM)",
        "lm_path": None,
        "is_char": False,
        "alpha": 0.0,
        "beta": 0.0,
        "beam_width": best_params["beam_width"]
    },
    {
        "id": "D-02",
        "name": "CTC + Character 3-gram (Domain D)",
        "lm_path": str(LM_DIR / "lm_char_3gram_domain_d.arpa"),
        "is_char": True,
        "alpha": 0.25,
        "beta": 0.0,
        "beam_width": best_params["beam_width"]
    },
    {
        "id": "D-03",
        "name": "CTC + Character 5-gram (Domain D)",
        "lm_path": str(LM_DIR / "lm_char_5gram_domain_d.arpa"),
        "is_char": True,
        "alpha": 0.25,
        "beta": 0.0,
        "beam_width": best_params["beam_width"]
    },
    {
        "id": "D-04",
        "name": "CTC + Character 6-gram (Domain D)",
        "lm_path": str(LM_DIR / "lm_char_6gram_domain_d.arpa"),
        "is_char": True,
        "alpha": 0.25,
        "beta": 0.0,
        "beam_width": best_params["beam_width"]
    },
    {
        "id": "D-05",
        "name": "CTC + Word 3-gram (Domain A - General)",
        "lm_path": str(LM_DIR / "lm_word_3gram_domain_a.arpa"),
        "is_char": False,
        "alpha": best_params["alpha"],
        "beta": best_params["beta"],
        "beam_width": best_params["beam_width"]
    },
    {
        "id": "D-06",
        "name": "CTC + Word 3-gram (Domain B - Conversational)",
        "lm_path": str(LM_DIR / "lm_word_3gram_domain_b.arpa"),
        "is_char": False,
        "alpha": best_params["alpha"],
        "beta": best_params["beta"],
        "beam_width": best_params["beam_width"]
    },
    {
        "id": "D-07",
        "name": "CTC + Word 3-gram (Domain C - Read)",
        "lm_path": str(LM_DIR / "lm_word_3gram_domain_c.arpa"),
        "is_char": False,
        "alpha": best_params["alpha"],
        "beta": best_params["beta"],
        "beam_width": best_params["beam_width"]
    },
    {
        "id": "D-08",
        "name": "CTC + Word 3-gram (Domain D - Balanced)",
        "lm_path": str(LM_DIR / "lm_word_3gram_domain_d.arpa"),
        "is_char": False,
        "alpha": best_params["alpha"],
        "beta": best_params["beta"],
        "beam_width": best_params["beam_width"]
    },
    {
        "id": "D-09",
        "name": "CTC + Word 4-gram (Domain D - Balanced)",
        "lm_path": str(LM_DIR / "lm_word_4gram_domain_d.arpa"),
        "is_char": False,
        "alpha": best_params["alpha"],
        "beta": best_params["beta"],
        "beam_width": best_params["beam_width"]
    }
]

ablation_results = []
best_overall_norm_wer = float("inf")
best_overall_hyps_raw = []
best_overall_hyps_norm = []
best_overall_cfg = None

for abl in ablations:
    aid = abl["id"]
    aname = abl["name"]
    lm_p = abl["lm_path"]
    bw = abl["beam_width"]
    alpha = abl["alpha"]
    beta = abl["beta"]
    
    print(f"\n--- Running Ablation {aid}: {aname} ---")
    gc.collect()
    
    dec = build_ctcdecoder(
        labels=labels_for_pyctc,
        kenlm_model_path=lm_p,
        alpha=alpha,
        beta=beta
    )
    
    t_start = time.time()
    hyps_raw = []
    hyps_norm = []
    
    for lp in cached_lps:
        h = dec.decode(lp, beam_width=bw).strip()
        hyps_raw.append(h)
        hyps_norm.append(normalize_perso_arabic_kashmiri(h))
        
    eval_elapsed = time.time() - t_start
    rtf = eval_elapsed / total_audio_sec
    lat_ms = (eval_elapsed / total_samples) * 1000.0
    
    # Calculate metrics
    norm_wer = jiwer.process_words(norm_refs, hyps_norm).wer * 100.0
    raw_wer = jiwer.process_words(raw_refs, hyps_raw).wer * 100.0
    norm_cer = jiwer.process_characters(norm_refs, hyps_norm).cer * 100.0
    raw_cer = jiwer.process_characters(raw_refs, hyps_raw).cer * 100.0
    
    # Delta from D-00 baseline
    if aid == "D-00":
        baseline_wer = norm_wer
        delta_wer = 0.0
    else:
        delta_wer = norm_wer - baseline_wer
        
    print(f"{aid} ({aname}): Norm WER = {norm_wer:.2f}% (Δ = {delta_wer:+.2f}%), Norm CER = {norm_cer:.2f}%")
    print(f"Decode Time: {eval_elapsed:.1f}s | RTF: {rtf:.5f} | Latency: {lat_ms:.2f} ms/sample")
    
    ablation_results.append({
        "ablation_id": aid,
        "configuration": aname,
        "lm_model": Path(lm_p).name if lm_p else "None",
        "beam_width": bw,
        "alpha": alpha,
        "beta": beta,
        "norm_wer": round(norm_wer, 2),
        "norm_cer": round(norm_cer, 2),
        "raw_wer": round(raw_wer, 2),
        "raw_cer": round(raw_cer, 2),
        "delta_wer": round(delta_wer, 2),
        "rtf": round(rtf, 5),
        "latency_ms": round(lat_ms, 2)
    })
    
    if norm_wer < best_overall_norm_wer:
        best_overall_norm_wer = norm_wer
        best_overall_hyps_raw = hyps_raw
        best_overall_hyps_norm = hyps_norm
        best_overall_cfg = {
            "ablation_id": aid,
            "configuration": aname,
            "lm_model": lm_p,
            "beam_width": bw,
            "alpha": alpha,
            "beta": beta,
            "norm_wer": norm_wer,
            "norm_cer": norm_cer,
            "rtf": rtf,
            "latency_ms": lat_ms
        }

# Save Ablation Results CSV
ablation_csv_path = REPORTS_DIR / "decoder_ablation_results.csv"
with open(ablation_csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "ablation_id", "configuration", "lm_model", "beam_width", "alpha", "beta",
        "norm_wer", "norm_cer", "raw_wer", "raw_cer", "delta_wer", "rtf", "latency_ms"
    ])
    writer.writeheader()
    writer.writerows(ablation_results)

# Save Best Decoder YAML Config
best_config_path = CONFIGS_DIR / "best_decoder_config.yaml"
with open(best_config_path, "w", encoding="utf-8") as f:
    yaml.dump({
        "stage": "Stage 7 LM Shallow Fusion",
        "best_ablation": best_overall_cfg["ablation_id"],
        "configuration_name": best_overall_cfg["configuration"],
        "lm_checkpoint": best_overall_cfg["lm_model"],
        "beam_width": best_overall_cfg["beam_width"],
        "alpha": best_overall_cfg["alpha"],
        "beta": best_overall_cfg["beta"],
        "best_norm_wer": round(best_overall_cfg["norm_wer"], 2),
        "best_norm_cer": round(best_overall_cfg["norm_cer"], 2),
        "real_time_factor": round(best_overall_cfg["rtf"], 5),
        "latency_ms_per_utterance": round(best_overall_cfg["latency_ms"], 2)
    }, f, default_flow_style=False)

# Save Best Predictions JSONL
best_pred_path = PRED_DIR / "best_validation_predictions.jsonl"
with open(best_pred_path, "w", encoding="utf-8") as f:
    for sid, h_raw, h_norm, r_raw, r_norm, src in zip(samples_list, best_overall_hyps_raw, best_overall_hyps_norm, raw_refs, norm_refs, sources):
        f.write(json.dumps({
            "sample_id": sid,
            "source": src,
            "hypothesis_raw": h_raw,
            "hypothesis_norm": h_norm,
            "reference_raw": r_raw,
            "reference_norm": r_norm
        }, ensure_ascii=False) + "\n")

print(f"\nPhase F Complete! Best system: {best_overall_cfg['ablation_id']} ({best_overall_cfg['configuration']}) with Norm WER = {best_overall_norm_wer:.2f}%")
print(f"Saved ablation table to {ablation_csv_path}")
print(f"Saved best config to {best_config_path}")
print(f"Saved best predictions to {best_pred_path}")
