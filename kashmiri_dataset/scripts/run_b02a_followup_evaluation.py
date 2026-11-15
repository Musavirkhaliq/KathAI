#!/usr/bin/env python3
"""
EduCasheer Kashmiri ASR: B-02A Follow-Up Held-Out Test Evaluation Engine
========================================================================
Executes a single controlled follow-up evaluation of frozen B-02A (Epoch 19)
versus frozen B-01 (Epoch 13) on the 5,172 eligible held-out test utterances
under identical preprocessing, masking, decoders, and scoring.
"""

import os
import sys
import json
import time
import math
import hashlib
import unicodedata
from pathlib import Path
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pyctcdecode import build_ctcdecoder
import jiwer

WORKSPACE_DIR = Path("/home/musa/Kashmiri Project")
DATASET_DIR = WORKSPACE_DIR / "kashmiri_dataset"
EXP_ROOT = DATASET_DIR / "experiments/native_asr_training/experiments"
TEST_MANIFEST = DATASET_DIR / "experiments/native_asr_training/data_splits/test.jsonl"
STAGE6_DIR = EXP_ROOT / "stage6_held_out_benchmark"
STAGE7_4GRAM_LM = EXP_ROOT / "stage7_ctc_lm_decoding/language_models/lm_word_4gram_domain_d.arpa"

B01_DIR = EXP_ROOT / "B-01-xlsr300m-character-ctc/best_model"
B02A_DIR = EXP_ROOT / "stage9_b02_acoustic_comparison/B-02A/checkpoints/epoch_19"

OUT_DIR = WORKSPACE_DIR
ARCHIVE_DIR = EXP_ROOT / "stage9_b02_acoustic_comparison/b02a_followup_heldout_evaluation"
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(DATASET_DIR / "scripts"))
from metrics_evaluator import calculate_wer_and_cer, normalize_perso_arabic_kashmiri


def get_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_alignment_counts(refs, hyps):
    """Computes exact substitutions, deletions, insertions, hits across corpus."""
    out = jiwer.process_words(refs, hyps)
    return {
        "substitutions": int(out.substitutions),
        "deletions": int(out.deletions),
        "insertions": int(out.insertions),
        "hits": int(out.hits),
        "ref_words": int(out.substitutions + out.deletions + out.hits),
        "wer": float(out.wer)
    }


def compute_char_alignment_counts(refs, hyps):
    out = jiwer.process_characters(refs, hyps)
    return {
        "substitutions": int(out.substitutions),
        "deletions": int(out.deletions),
        "insertions": int(out.insertions),
        "hits": int(out.hits),
        "ref_chars": int(out.substitutions + out.deletions + out.hits),
        "cer": float(out.cer)
    }


def evaluate_model_on_test(model_name, model_dir, test_samples, dec_no_lm, dec_kenlm, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}\nEVALUATING SYSTEM: {model_name}\nPath: {model_dir}\nDevice: {device}\n{'='*70}")

    processor = Wav2Vec2Processor.from_pretrained(str(model_dir))
    model = Wav2Vec2ForCTC.from_pretrained(str(model_dir)).to(device)
    model.eval()

    predictions = []
    total_samples = len(test_samples)
    total_audio_sec = sum(s["duration"] for s in test_samples)

    total_gpu_time = 0.0
    total_greedy_lat = 0.0
    total_beam_no_lm_lat = 0.0
    total_beam_kenlm_lat = 0.0

    t_eval_start = time.time()

    with torch.no_grad():
        for b_idx in range(0, total_samples, batch_size):
            batch_items = test_samples[b_idx:b_idx+batch_size]
            wavs = []
            audio_lens = []

            for item in batch_items:
                audio_path = WORKSPACE_DIR / item["audio_path"]
                try:
                    wav, sr = sf.read(str(audio_path), dtype="float32")
                    if audio_path.suffix == ".wav" and wav.ndim > 1:
                        wav = np.mean(wav, axis=1)
                    if sr != 16000:
                        import torchaudio
                        t_wav = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
                        wav = torchaudio.transforms.Resample(sr, 16000)(t_wav).squeeze(0).numpy()
                    wavs.append(wav)
                    audio_lens.append(len(wav))
                except Exception as e:
                    print(f"Error reading {audio_path}: {e}")
                    dummy = np.zeros(16000, dtype=np.float32)
                    wavs.append(dummy)
                    audio_lens.append(16000)

            t_gpu_0 = time.time()
            inputs = processor(wavs, sampling_rate=16000, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(device)
            attention_mask = inputs.attention_mask.to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_values=input_values, attention_mask=attention_mask)
                logits = outputs.logits  # [B, T, V]
                log_probs = torch.log_softmax(logits, dim=-1).cpu().float().numpy()

            batch_gpu_time = time.time() - t_gpu_0
            total_gpu_time += batch_gpu_time

            for i_in_b, item in enumerate(batch_items):
                sid = item["sample_id"]
                dur = item["duration"]
                raw_ref = item["text"].strip()
                norm_ref = normalize_perso_arabic_kashmiri(raw_ref)
                src = item.get("source") or item.get("dataset_source") or "unknown"
                spk = item.get("speaker_id", "unknown")

                # Exact unpadded acoustic frames
                a_len = audio_lens[i_in_b]
                t_frames = int(model._get_feat_extract_output_lengths(torch.tensor([a_len])).item())
                lp_slice = log_probs[i_in_b, :t_frames, :]
                logit_slice = logits[i_in_b, :t_frames, :]

                # 1. Corrected Masked Greedy CTC
                t0_dec = time.time()
                pred_ids = torch.argmax(logit_slice, dim=-1)
                greedy_raw = processor.decode(pred_ids)
                greedy_norm = normalize_perso_arabic_kashmiri(greedy_raw)
                greedy_lat = time.time() - t0_dec
                total_greedy_lat += greedy_lat

                # 2. Beam Search No LM (width 16)
                t0_dec = time.time()
                beam_no_lm_raw = dec_no_lm.decode(lp_slice, beam_width=16)
                beam_no_lm_norm = normalize_perso_arabic_kashmiri(beam_no_lm_raw)
                beam_no_lm_lat = time.time() - t0_dec
                total_beam_no_lm_lat += beam_no_lm_lat

                # 3. Beam Search + KenLM (width 16, alpha=0.25, beta=0.00)
                t0_dec = time.time()
                beam_kenlm_raw = dec_kenlm.decode(lp_slice, beam_width=16)
                beam_kenlm_norm = normalize_perso_arabic_kashmiri(beam_kenlm_raw)
                beam_kenlm_lat = time.time() - t0_dec
                total_beam_kenlm_lat += beam_kenlm_lat

                # Metrics for greedy
                m_g_norm = calculate_wer_and_cer(norm_ref, greedy_norm)
                m_g_raw = calculate_wer_and_cer(raw_ref, greedy_raw)

                # Metrics for beam no lm
                m_b_norm = calculate_wer_and_cer(norm_ref, beam_no_lm_norm)
                m_b_raw = calculate_wer_and_cer(raw_ref, beam_no_lm_raw)

                # Metrics for beam kenlm
                m_k_norm = calculate_wer_and_cer(norm_ref, beam_kenlm_norm)
                m_k_raw = calculate_wer_and_cer(raw_ref, beam_kenlm_raw)

                predictions.append({
                    "sample_id": sid,
                    "source": src,
                    "speaker_id": spk,
                    "duration": dur,
                    "ref_raw": raw_ref,
                    "ref_norm": norm_ref,
                    "greedy": {
                        "hyp_raw": greedy_raw,
                        "hyp_norm": greedy_norm,
                        "raw_wer": round(m_g_raw["wer"], 4),
                        "raw_cer": round(m_g_raw["cer"], 4),
                        "norm_wer": round(m_g_norm["wer"], 4),
                        "norm_cer": round(m_g_norm["cer"], 4),
                        "exact_match": m_g_norm["exact_match"],
                        "empty": len(greedy_norm.strip()) == 0,
                        "latency_sec": round(greedy_lat, 5)
                    },
                    "beam_no_lm": {
                        "hyp_raw": beam_no_lm_raw,
                        "hyp_norm": beam_no_lm_norm,
                        "raw_wer": round(m_b_raw["wer"], 4),
                        "raw_cer": round(m_b_raw["cer"], 4),
                        "norm_wer": round(m_b_norm["wer"], 4),
                        "norm_cer": round(m_b_norm["cer"], 4),
                        "exact_match": m_b_norm["exact_match"],
                        "empty": len(beam_no_lm_norm.strip()) == 0,
                        "latency_sec": round(beam_no_lm_lat, 5)
                    },
                    "beam_kenlm": {
                        "hyp_raw": beam_kenlm_raw,
                        "hyp_norm": beam_kenlm_norm,
                        "raw_wer": round(m_k_raw["wer"], 4),
                        "raw_cer": round(m_k_raw["cer"], 4),
                        "norm_wer": round(m_k_norm["wer"], 4),
                        "norm_cer": round(m_k_norm["cer"], 4),
                        "exact_match": m_k_norm["exact_match"],
                        "empty": len(beam_kenlm_norm.strip()) == 0,
                        "latency_sec": round(beam_kenlm_lat, 5)
                    }
                })

            done_count = len(predictions)
            if (done_count // batch_size) % 25 == 0 or done_count == total_samples:
                elapsed = time.time() - t_eval_start
                print(f"[{done_count}/{total_samples}] Processed in {elapsed:.1f}s...", flush=True)

    total_eval_time = time.time() - t_eval_start

    # Compute corpus-level and breakdown statistics for each decoder
    decoders_results = {}
    for dec_key in ["greedy", "beam_no_lm", "beam_kenlm"]:
        refs_norm = [p["ref_norm"] for p in predictions]
        hyps_norm = [p[dec_key]["hyp_norm"] for p in predictions]
        refs_raw = [p["ref_raw"] for p in predictions]
        hyps_raw = [p[dec_key]["hyp_raw"] for p in predictions]

        w_norm = compute_alignment_counts(refs_norm, hyps_norm)
        c_norm = compute_char_alignment_counts(refs_norm, hyps_norm)
        w_raw = compute_alignment_counts(refs_raw, hyps_raw)
        c_raw = compute_char_alignment_counts(refs_raw, hyps_raw)

        exact_matches = sum(1 for p in predictions if p[dec_key]["exact_match"])
        empty_preds = sum(1 for p in predictions if p[dec_key]["empty"])

        # Decoder latency
        if dec_key == "greedy":
            dec_lat = total_greedy_lat
        elif dec_key == "beam_no_lm":
            dec_lat = total_beam_no_lm_lat
        else:
            dec_lat = total_beam_kenlm_lat

        e2e_time = total_gpu_time + dec_lat
        rtf = e2e_time / max(total_audio_sec, 0.001)
        throughput_x = total_audio_sec / max(e2e_time, 0.001)

        # Source breakdown
        src_breakdown = {}
        unique_sources = sorted(list(set(p["source"] for p in predictions)))
        for s in unique_sources:
            s_preds = [p for p in predictions if p["source"] == s]
            s_refs_n = [p["ref_norm"] for p in s_preds]
            s_hyps_n = [p[dec_key]["hyp_norm"] for p in s_preds]
            s_refs_r = [p["ref_raw"] for p in s_preds]
            s_hyps_r = [p[dec_key]["hyp_raw"] for p in s_preds]

            sw_n = compute_alignment_counts(s_refs_n, s_hyps_n)
            sc_n = compute_char_alignment_counts(s_refs_n, s_hyps_n)
            sw_r = compute_alignment_counts(s_refs_r, s_hyps_r)
            sc_r = compute_char_alignment_counts(s_refs_r, s_hyps_r)

            src_breakdown[s] = {
                "sample_count": len(s_preds),
                "audio_hours": round(sum(p["duration"] for p in s_preds) / 3600.0, 4),
                "norm_wer": round(sw_n["wer"], 4),
                "norm_cer": round(sc_n["cer"], 4),
                "raw_wer": round(sw_r["wer"], 4),
                "raw_cer": round(sc_r["cer"], 4),
                "exact_match_count": sum(1 for p in s_preds if p[dec_key]["exact_match"]),
                "exact_match_pct": round(sum(1 for p in s_preds if p[dec_key]["exact_match"]) / len(s_preds) * 100.0, 2),
                "empty_count": sum(1 for p in s_preds if p[dec_key]["empty"]),
                "substitutions": sw_n["substitutions"],
                "deletions": sw_n["deletions"],
                "insertions": sw_n["insertions"],
                "ref_words": sw_n["ref_words"]
            }

        decoders_results[dec_key] = {
            "norm_wer": round(w_norm["wer"], 4),
            "norm_cer": round(c_norm["cer"], 4),
            "raw_wer": round(w_raw["wer"], 4),
            "raw_cer": round(c_raw["cer"], 4),
            "substitutions": w_norm["substitutions"],
            "deletions": w_norm["deletions"],
            "insertions": w_norm["insertions"],
            "hits": w_norm["hits"],
            "total_ref_words": w_norm["ref_words"],
            "char_substitutions": c_norm["substitutions"],
            "char_deletions": c_norm["deletions"],
            "char_insertions": c_norm["insertions"],
            "total_ref_chars": c_norm["ref_chars"],
            "exact_matches": exact_matches,
            "exact_match_pct": round(exact_matches / total_samples * 100.0, 2),
            "empty_predictions": empty_preds,
            "empty_pct": round(empty_preds / total_samples * 100.0, 2),
            "gpu_time_sec": round(total_gpu_time, 2),
            "decoder_latency_sec": round(dec_lat, 2),
            "decoder_ms_per_sample": round(dec_lat / total_samples * 1000.0, 2),
            "end_to_end_sec": round(e2e_time, 2),
            "rtf": round(rtf, 5),
            "throughput_x_realtime": round(throughput_x, 2),
            "source_breakdown": src_breakdown
        }

    return {
        "model_name": model_name,
        "model_dir": str(model_dir),
        "total_samples": total_samples,
        "total_audio_hours": round(total_audio_sec / 3600.0, 4),
        "total_wall_clock_time_sec": round(total_eval_time, 2),
        "gpu_inference_time_sec": round(total_gpu_time, 2),
        "predictions": predictions,
        "decoders": decoders_results
    }


def run_paired_bootstrap(b01_preds, b02a_preds, dec_key, num_bootstrap=10000, seed=42):
    print(f"Running {num_bootstrap:,} paired bootstrap iterations for {dec_key}...")
    np.random.seed(seed)

    N = len(b01_preds)
    # Collect per-sample distances
    samples = []
    for p1, p2 in zip(b01_preds, b02a_preds):
        assert p1["sample_id"] == p2["sample_id"]
        ref_n = p1["ref_norm"]
        ref_r = p1["ref_raw"]

        # Norm WER
        w1_n = calculate_wer_and_cer(ref_n, p1[dec_key]["hyp_norm"])
        w2_n = calculate_wer_and_cer(ref_n, p2[dec_key]["hyp_norm"])

        # Raw WER
        w1_r = calculate_wer_and_cer(ref_r, p1[dec_key]["hyp_raw"])
        w2_r = calculate_wer_and_cer(ref_r, p2[dec_key]["hyp_raw"])

        samples.append({
            "rw_norm": w1_n["ref_words"],
            "rc_norm": w1_n["ref_chars"],
            "b01_wd_norm": w1_n["word_dist"],
            "b02a_wd_norm": w2_n["word_dist"],
            "b01_cd_norm": w1_n["char_dist"],
            "b02a_cd_norm": w2_n["char_dist"],
            "rw_raw": w1_r["ref_words"],
            "rc_raw": w1_r["ref_chars"],
            "b01_wd_raw": w1_r["word_dist"],
            "b02a_wd_raw": w2_r["word_dist"],
            "b01_cd_raw": w1_r["char_dist"],
            "b02a_cd_raw": w2_r["char_dist"],
        })

    rw_norm = np.array([s["rw_norm"] for s in samples], dtype=np.int64)
    rc_norm = np.array([s["rc_norm"] for s in samples], dtype=np.int64)
    b01_wd_norm = np.array([s["b01_wd_norm"] for s in samples], dtype=np.int64)
    b02a_wd_norm = np.array([s["b02a_wd_norm"] for s in samples], dtype=np.int64)
    b01_cd_norm = np.array([s["b01_cd_norm"] for s in samples], dtype=np.int64)
    b02a_cd_norm = np.array([s["b02a_cd_norm"] for s in samples], dtype=np.int64)

    rw_raw = np.array([s["rw_raw"] for s in samples], dtype=np.int64)
    rc_raw = np.array([s["rc_raw"] for s in samples], dtype=np.int64)
    b01_wd_raw = np.array([s["b01_wd_raw"] for s in samples], dtype=np.int64)
    b02a_wd_raw = np.array([s["b02a_wd_raw"] for s in samples], dtype=np.int64)
    b01_cd_raw = np.array([s["b01_cd_raw"] for s in samples], dtype=np.int64)
    b02a_cd_raw = np.array([s["b02a_cd_raw"] for s in samples], dtype=np.int64)

    # Point estimates
    tot_rw_norm = max(1, np.sum(rw_norm))
    tot_rc_norm = max(1, np.sum(rc_norm))
    tot_rw_raw = max(1, np.sum(rw_raw))
    tot_rc_raw = max(1, np.sum(rc_raw))

    p_b01_wer_norm = np.sum(b01_wd_norm) / tot_rw_norm
    p_b02a_wer_norm = np.sum(b02a_wd_norm) / tot_rw_norm
    diff_wer_norm = p_b02a_wer_norm - p_b01_wer_norm

    p_b01_cer_norm = np.sum(b01_cd_norm) / tot_rc_norm
    p_b02a_cer_norm = np.sum(b02a_cd_norm) / tot_rc_norm
    diff_cer_norm = p_b02a_cer_norm - p_b01_cer_norm

    p_b01_wer_raw = np.sum(b01_wd_raw) / tot_rw_raw
    p_b02a_wer_raw = np.sum(b02a_wd_raw) / tot_rw_raw
    diff_wer_raw = p_b02a_wer_raw - p_b01_wer_raw

    p_b01_cer_raw = np.sum(b01_cd_raw) / tot_rc_raw
    p_b02a_cer_raw = np.sum(b02a_cd_raw) / tot_rc_raw
    diff_cer_raw = p_b02a_cer_raw - p_b01_cer_raw

    boot_diff_wer_norm = np.zeros(num_bootstrap)
    boot_diff_cer_norm = np.zeros(num_bootstrap)
    boot_b01_wer_norm = np.zeros(num_bootstrap)
    boot_b02a_wer_norm = np.zeros(num_bootstrap)
    boot_b01_cer_norm = np.zeros(num_bootstrap)
    boot_b02a_cer_norm = np.zeros(num_bootstrap)

    boot_diff_wer_raw = np.zeros(num_bootstrap)
    boot_diff_cer_raw = np.zeros(num_bootstrap)

    for b in range(num_bootstrap):
        idx = np.random.randint(0, N, size=N)

        rwn = np.sum(rw_norm[idx])
        rcn = np.sum(rc_norm[idx])
        rwr = np.sum(rw_raw[idx])
        rcr = np.sum(rc_raw[idx])

        w1_n = np.sum(b01_wd_norm[idx]) / max(1, rwn)
        w2_n = np.sum(b02a_wd_norm[idx]) / max(1, rwn)
        c1_n = np.sum(b01_cd_norm[idx]) / max(1, rcn)
        c2_n = np.sum(b02a_cd_norm[idx]) / max(1, rcn)

        w1_r = np.sum(b01_wd_raw[idx]) / max(1, rwr)
        w2_r = np.sum(b02a_wd_raw[idx]) / max(1, rwr)
        c1_r = np.sum(b01_cd_raw[idx]) / max(1, rcr)
        c2_r = np.sum(b02a_cd_raw[idx]) / max(1, rcr)

        boot_b01_wer_norm[b] = w1_n
        boot_b02a_wer_norm[b] = w2_n
        boot_diff_wer_norm[b] = w2_n - w1_n

        boot_b01_cer_norm[b] = c1_n
        boot_b02a_cer_norm[b] = c2_n
        boot_diff_cer_norm[b] = c2_n - c1_n

        boot_diff_wer_raw[b] = w2_r - w1_r
        boot_diff_cer_raw[b] = c2_r - c1_r

    p_val_wer_norm = 2.0 * min(float(np.mean(boot_diff_wer_norm <= 0)), float(np.mean(boot_diff_wer_norm >= 0)))
    p_val_cer_norm = 2.0 * min(float(np.mean(boot_diff_cer_norm <= 0)), float(np.mean(boot_diff_cer_norm >= 0)))
    p_val_wer_raw = 2.0 * min(float(np.mean(boot_diff_wer_raw <= 0)), float(np.mean(boot_diff_wer_raw >= 0)))
    p_val_cer_raw = 2.0 * min(float(np.mean(boot_diff_cer_raw <= 0)), float(np.mean(boot_diff_cer_raw >= 0)))

    # Relative WER reduction
    rel_werr = ((p_b01_wer_norm - p_b02a_wer_norm) / max(0.0001, p_b01_wer_norm)) * 100.0

    return {
        "decoder": dec_key,
        "num_samples": N,
        "normalized_wer": {
            "b01": round(float(p_b01_wer_norm), 4),
            "b01_95_ci": [round(float(np.percentile(boot_b01_wer_norm, 2.5)), 4), round(float(np.percentile(boot_b01_wer_norm, 97.5)), 4)],
            "b02a": round(float(p_b02a_wer_norm), 4),
            "b02a_95_ci": [round(float(np.percentile(boot_b02a_wer_norm, 2.5)), 4), round(float(np.percentile(boot_b02a_wer_norm, 97.5)), 4)],
            "delta_wer": round(float(diff_wer_norm), 4),
            "delta_wer_pp": round(float(diff_wer_norm * 100.0), 2),
            "delta_wer_95_ci": [round(float(np.percentile(boot_diff_wer_norm, 2.5)), 4), round(float(np.percentile(boot_diff_wer_norm, 97.5)), 4)],
            "relative_reduction_pct": round(float(rel_werr), 2),
            "p_value": round(float(p_val_wer_norm), 5)
        },
        "normalized_cer": {
            "b01": round(float(p_b01_cer_norm), 4),
            "b01_95_ci": [round(float(np.percentile(boot_b01_cer_norm, 2.5)), 4), round(float(np.percentile(boot_b01_cer_norm, 97.5)), 4)],
            "b02a": round(float(p_b02a_cer_norm), 4),
            "b02a_95_ci": [round(float(np.percentile(boot_b02a_cer_norm, 2.5)), 4), round(float(np.percentile(boot_b02a_cer_norm, 97.5)), 4)],
            "delta_cer": round(float(diff_cer_norm), 4),
            "delta_cer_pp": round(float(diff_cer_norm * 100.0), 2),
            "delta_cer_95_ci": [round(float(np.percentile(boot_diff_cer_norm, 2.5)), 4), round(float(np.percentile(boot_diff_cer_norm, 97.5)), 4)],
            "p_value": round(float(p_val_cer_norm), 5)
        },
        "raw_wer": {
            "b01": round(float(p_b01_wer_raw), 4),
            "b02a": round(float(p_b02a_wer_raw), 4),
            "delta_wer": round(float(diff_wer_raw), 4),
            "delta_wer_pp": round(float(diff_wer_raw * 100.0), 2),
            "delta_wer_95_ci": [round(float(np.percentile(boot_diff_wer_raw, 2.5)), 4), round(float(np.percentile(boot_diff_wer_raw, 97.5)), 4)],
            "p_value": round(float(p_val_wer_raw), 5)
        },
        "raw_cer": {
            "b01": round(float(p_b01_cer_raw), 4),
            "b02a": round(float(p_b02a_cer_raw), 4),
            "delta_cer": round(float(diff_cer_raw), 4),
            "delta_cer_pp": round(float(diff_cer_raw * 100.0), 2),
            "delta_cer_95_ci": [round(float(np.percentile(boot_diff_cer_raw, 2.5)), 4), round(float(np.percentile(boot_diff_cer_raw, 97.5)), 4)],
            "p_value": round(float(p_val_cer_raw), 5)
        }
    }


def main():
    print("======================================================================")
    print("EDUCASHEER KASHMIRI ASR: B-02A HELD-OUT TEST EVALUATION")
    print("======================================================================")

    # 1. Mandatory Pre-Evaluation Verification Checks
    print("\n--- Mandatory Pre-Evaluation Checks ---")
    b01_model_hash = get_sha256(B01_DIR / "model.safetensors")
    b02a_model_hash = get_sha256(B02A_DIR / "model.safetensors")
    kenlm_hash = get_sha256(STAGE7_4GRAM_LM)

    print(f"B-01 model.safetensors SHA256:   {b01_model_hash}")
    print(f"B-02A model.safetensors SHA256:  {b02a_model_hash}")
    print(f"KenLM ARPA SHA256:               {kenlm_hash}")

    # Check Stage 6 sample population
    print(f"Loading test manifest from: {TEST_MANIFEST}")
    test_samples = []
    with open(TEST_MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            dur = item.get("duration", 0)
            if 0.5 <= dur <= 16.0:
                t_frames = int(dur * 50)
                target_len = len(unicodedata.normalize("NFKC", item.get("text", "")))
                if 0 < target_len < t_frames:
                    test_samples.append(item)

    print(f"Loaded {len(test_samples)} eligible held-out test samples.")
    assert len(test_samples) == 5172, f"Expected 5,172 samples, found {len(test_samples)}"

    # Build Decoders
    processor_b01 = Wav2Vec2Processor.from_pretrained(str(B01_DIR))
    vocab = processor_b01.tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    labels = [tok if tok != "|" else " " for tok, idx in sorted_vocab]

    print("\nInitializing CTC prefix beam search decoders...")
    dec_no_lm = build_ctcdecoder(labels=labels, kenlm_model_path=None)
    dec_kenlm = build_ctcdecoder(
        labels=labels,
        kenlm_model_path=str(STAGE7_4GRAM_LM),
        alpha=0.25,
        beta=0.00
    )

    # 2. Evaluate B-01 (Baseline)
    print("\n--- Evaluating Frozen B-01 Baseline ---")
    b01_results = evaluate_model_on_test("B-01", B01_DIR, test_samples, dec_no_lm, dec_kenlm, batch_size=32)

    # 3. Evaluate B-02A (Follow-Up Checkpoint Epoch 19)
    print("\n--- Evaluating Frozen B-02A Epoch 19 ---")
    b02a_results = evaluate_model_on_test("B-02A", B02A_DIR, test_samples, dec_no_lm, dec_kenlm, batch_size=32)

    # 4. Save JSONL Prediction Artifacts
    print("\nWriting prediction files...")
    b01_pred_file = OUT_DIR / "b01_corrected_test_predictions.jsonl"
    with open(b01_pred_file, "w", encoding="utf-8") as f:
        for p in b01_results["predictions"]:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote {len(b01_results['predictions'])} lines to {b01_pred_file}")

    b02a_pred_file = OUT_DIR / "b02a_followup_test_predictions.jsonl"
    with open(b02a_pred_file, "w", encoding="utf-8") as f:
        for p in b02a_results["predictions"]:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote {len(b02a_results['predictions'])} lines to {b02a_pred_file}")

    # Copy to archive dir as well
    with open(ARCHIVE_DIR / "b01_corrected_test_predictions.jsonl", "w", encoding="utf-8") as f:
        for p in b01_results["predictions"]:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(ARCHIVE_DIR / "b02a_followup_test_predictions.jsonl", "w", encoding="utf-8") as f:
        for p in b02a_results["predictions"]:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # 5. Run Paired Bootstrap Significance Tests
    print("\n--- Running Paired Bootstrap Significance Analysis ---")
    bootstrap_results = {}
    for dec_key in ["greedy", "beam_no_lm", "beam_kenlm"]:
        bootstrap_results[dec_key] = run_paired_bootstrap(
            b01_results["predictions"],
            b02a_results["predictions"],
            dec_key=dec_key,
            num_bootstrap=10000,
            seed=42
        )

    boot_file = OUT_DIR / "paired_test_bootstrap.json"
    with open(boot_file, "w", encoding="utf-8") as f:
        json.dump(bootstrap_results, f, indent=2)
    with open(ARCHIVE_DIR / "paired_test_bootstrap.json", "w", encoding="utf-8") as f:
        json.dump(bootstrap_results, f, indent=2)
    print(f"Saved paired bootstrap analysis to {boot_file}")

    # 6. Generate matched_decoder_test_comparison.csv
    print("\nGenerating matched decoder comparison CSV...")
    comp_file = OUT_DIR / "matched_decoder_test_comparison.csv"
    with open(comp_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "decoder_configuration",
            "model_id",
            "norm_wer_pct",
            "norm_cer_pct",
            "raw_wer_pct",
            "raw_cer_pct",
            "substitutions",
            "deletions",
            "insertions",
            "hits",
            "total_ref_words",
            "exact_matches_count",
            "exact_matches_pct",
            "empty_predictions_count",
            "empty_predictions_pct",
            "gpu_time_sec",
            "decoder_latency_sec",
            "decoder_ms_per_sample",
            "end_to_end_sec",
            "rtf",
            "throughput_x_realtime",
            "delta_norm_wer_pp",
            "relative_werr_pct",
            "p_value_wer"
        ])

        decoder_names = [
            ("greedy", "Corrected Masked Greedy CTC"),
            ("beam_no_lm", "Prefix Beam Search (W=16, No LM)"),
            ("beam_kenlm", "Prefix Beam Search + KenLM 4-gram (W=16, alpha=0.25, beta=0.00)")
        ]

        for dec_key, dec_label in decoder_names:
            b1_m = b01_results["decoders"][dec_key]
            b2_m = b02a_results["decoders"][dec_key]
            boot_m = bootstrap_results[dec_key]

            # B-01 row
            writer.writerow([
                dec_label,
                "B-01 (Baseline, Epoch 13)",
                f"{b1_m['norm_wer']*100:.2f}",
                f"{b1_m['norm_cer']*100:.2f}",
                f"{b1_m['raw_wer']*100:.2f}",
                f"{b1_m['raw_cer']*100:.2f}",
                b1_m["substitutions"],
                b1_m["deletions"],
                b1_m["insertions"],
                b1_m["hits"],
                b1_m["total_ref_words"],
                b1_m["exact_matches"],
                f"{b1_m['exact_match_pct']:.2f}",
                b1_m["empty_predictions"],
                f"{b1_m['empty_pct']:.2f}",
                f"{b1_m['gpu_time_sec']:.2f}",
                f"{b1_m['decoder_latency_sec']:.2f}",
                f"{b1_m['decoder_ms_per_sample']:.2f}",
                f"{b1_m['end_to_end_sec']:.2f}",
                f"{b1_m['rtf']:.5f}",
                f"{b1_m['throughput_x_realtime']:.2f}",
                "-",
                "-",
                "-"
            ])

            # B-02A row
            writer.writerow([
                dec_label,
                "B-02A (Recovered Speech, Epoch 19)",
                f"{b2_m['norm_wer']*100:.2f}",
                f"{b2_m['norm_cer']*100:.2f}",
                f"{b2_m['raw_wer']*100:.2f}",
                f"{b2_m['raw_cer']*100:.2f}",
                b2_m["substitutions"],
                b2_m["deletions"],
                b2_m["insertions"],
                b2_m["hits"],
                b2_m["total_ref_words"],
                b2_m["exact_matches"],
                f"{b2_m['exact_match_pct']:.2f}",
                b2_m["empty_predictions"],
                f"{b2_m['empty_pct']:.2f}",
                f"{b2_m['gpu_time_sec']:.2f}",
                f"{b2_m['decoder_latency_sec']:.2f}",
                f"{b2_m['decoder_ms_per_sample']:.2f}",
                f"{b2_m['end_to_end_sec']:.2f}",
                f"{b2_m['rtf']:.5f}",
                f"{b2_m['throughput_x_realtime']:.2f}",
                f"{boot_m['normalized_wer']['delta_wer_pp']:+.2f}",
                f"{boot_m['normalized_wer']['relative_reduction_pct']:.2f}%",
                f"{boot_m['normalized_wer']['p_value']:.5f}"
            ])

    with open(ARCHIVE_DIR / "matched_decoder_test_comparison.csv", "w", encoding="utf-8") as f:
        f.write(open(comp_file).read())
    print(f"Saved matched decoder comparison to {comp_file}")

    # 7. Generate sourcewise_test_results.csv
    print("\nGenerating source-wise breakdown CSV...")
    src_file = OUT_DIR / "sourcewise_test_results.csv"
    with open(src_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "source",
            "decoder_configuration",
            "model_id",
            "sample_count",
            "audio_hours",
            "norm_wer_pct",
            "norm_cer_pct",
            "raw_wer_pct",
            "raw_cer_pct",
            "substitutions",
            "deletions",
            "insertions",
            "ref_words",
            "exact_matches_count",
            "exact_matches_pct",
            "empty_count"
        ])

        unique_sources = sorted(list(set(test_samples[0]["source"] for _ in [0]) | set(b01_results["decoders"]["greedy"]["source_breakdown"].keys())))

        for s in unique_sources:
            for dec_key, dec_label in decoder_names:
                b1_s = b01_results["decoders"][dec_key]["source_breakdown"][s]
                b2_s = b02a_results["decoders"][dec_key]["source_breakdown"][s]

                writer.writerow([
                    s,
                    dec_label,
                    "B-01 (Baseline)",
                    b1_s["sample_count"],
                    f"{b1_s['audio_hours']:.4f}",
                    f"{b1_s['norm_wer']*100:.2f}",
                    f"{b1_s['norm_cer']*100:.2f}",
                    f"{b1_s['raw_wer']*100:.2f}",
                    f"{b1_s['raw_cer']*100:.2f}",
                    b1_s["substitutions"],
                    b1_s["deletions"],
                    b1_s["insertions"],
                    b1_s["ref_words"],
                    b1_s["exact_match_count"],
                    f"{b1_s['exact_match_pct']:.2f}",
                    b1_s["empty_count"]
                ])

                writer.writerow([
                    s,
                    dec_label,
                    "B-02A (Epoch 19)",
                    b2_s["sample_count"],
                    f"{b2_s['audio_hours']:.4f}",
                    f"{b2_s['norm_wer']*100:.2f}",
                    f"{b2_s['norm_cer']*100:.2f}",
                    f"{b2_s['raw_wer']*100:.2f}",
                    f"{b2_s['raw_cer']*100:.2f}",
                    b2_s["substitutions"],
                    b2_s["deletions"],
                    b2_s["insertions"],
                    b2_s["ref_words"],
                    b2_s["exact_match_count"],
                    f"{b2_s['exact_match_pct']:.2f}",
                    b2_s["empty_count"]
                ])

    with open(ARCHIVE_DIR / "sourcewise_test_results.csv", "w", encoding="utf-8") as f:
        f.write(open(src_file).read())
    print(f"Saved source-wise breakdown to {src_file}")

    # 8. Print Executive Summary
    print("\n" + "="*80)
    print("                     EVALUATION EXECUTION COMPLETE")
    print("="*80)
    print(f"Total Evaluated Utterances: {total_samples:,} (9.30 audio hours)")
    print("-" * 80)
    for dec_key, dec_label in decoder_names:
        b1_w = b01_results["decoders"][dec_key]["norm_wer"] * 100
        b2_w = b02a_results["decoders"][dec_key]["norm_wer"] * 100
        delta = bootstrap_results[dec_key]["normalized_wer"]["delta_wer_pp"]
        werr = bootstrap_results[dec_key]["normalized_wer"]["relative_reduction_pct"]
        pval = bootstrap_results[dec_key]["normalized_wer"]["p_value"]
        print(f"[{dec_label}]")
        print(f"  B-01:  {b1_w:.2f}% Norm WER")
        print(f"  B-02A: {b2_w:.2f}% Norm WER")
        print(f"  Diff:  {delta:+.2f} pp (WERR: {werr:.2f}%, p = {pval:.5f})")
    print("="*80)

if __name__ == "__main__":
    import csv
    main()
