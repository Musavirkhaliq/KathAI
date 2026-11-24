#!/usr/bin/env python3
"""
Official V3.0 Benchmark Evaluation & Comprehensive Leaderboard Engine
====================================================================
Evaluates all historical, baseline, and candidate ASR acoustic models on the
newly established canonical V3 benchmark splits:
  1. test_standard_v3.jsonl (1,500 utts, 3.19 hrs) - Standard Multi-Source Test
  2. test_community_waaz_v3.jsonl (114 utts, 0.19 hrs) - Real-World WAAZ Community Test
  3. val_standard_v3.jsonl (1,000 utts, 2.06 hrs) - Standard Multi-Source Val
  4. val_community_waaz_v3.jsonl (309 utts, 0.83 hrs) - Real-World WAAZ Community Val

Models Evaluated:
  1. V3_Champion (Step 6000) - State-of-the-Art Model on 200 hrs Unified Reclaimed Data
  2. E22_AllData_Run1 (Step 2250) - Initial Expanded Fine-tune on 160 hrs Clean Data
  3. E22-01 (Stage 22 Champion) - Previous Master Champion (Multi-Source Balanced)
  4. E22-02 (Stage 22 Full WAAZ) - Trained on Full WAAZ Community Set
  5. E22-03 (Stage 22 MultiSource) - Trained on Multi-Source Data
  6. E22-04 (Stage 22 WAAZ Prioritized) - Trained with WAAZ Oversampling
  7. E-03 (Stage 15 MWER Best Calib) - Stage 15 MWER Calibrated Baseline
  8. R-02 (Stage 12 Robustness) - Noise-augmented Robustness Model
  9. B-02B (Stage 9 Acoustic) - Architectural Comparison Acoustic Variant
  10. B-01 (Original Baseline) - Original XLSR-300M Native Baseline
"""

import os
import sys
import json
import time
import gc
import hashlib
from pathlib import Path
from collections import defaultdict
import multiprocessing

import numpy as np
import soundfile as sf
import torch
import jiwer
from pyctcdecode import build_ctcdecoder
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

WORKSPACE = Path("/home/musa/Kashmiri Project")
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "kashmiri_dataset/scripts"))

from metrics_evaluator import normalize_perso_arabic_kashmiri as norm_strict

OUTPUT_DIR = WORKSPACE / "kashmiri_dataset/experiments/v3_benchmark_evaluations"
PRED_DIR = OUTPUT_DIR / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

LM_PATH = WORKSPACE / "kashmiri_dataset/experiments/native_asr_training/experiments/stage7_ctc_lm_decoding/language_models/lm_word_4gram_domain_d.arpa"
TOK_DIR = WORKSPACE / "kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc/tokenizer"

BENCHMARK_SPLITS = [
    ("test_standard_v3", "Test Standard (1,500 utts)", WORKSPACE / "kashmiri_dataset/releases/asr_v3_unified_splits/test_standard_v3.jsonl"),
    ("test_community_waaz_v3", "Test Community WAAZ (114 utts)", WORKSPACE / "kashmiri_dataset/releases/asr_v3_unified_splits/test_community_waaz_v3.jsonl"),
    ("val_standard_v3", "Val Standard (1,000 utts)", WORKSPACE / "kashmiri_dataset/releases/asr_v3_unified_splits/val_standard_v3.jsonl"),
    ("val_community_waaz_v3", "Val Community WAAZ (309 utts)", WORKSPACE / "kashmiri_dataset/releases/asr_v3_unified_splits/val_community_waaz_v3.jsonl"),
]

CANDIDATE_MODELS = [
    {
        "id": "V3_Champion",
        "name": "EduCasheer V3.0 Champion (Step 6000)",
        "era": "V3.0 Unified All-Data (200 hrs)",
        "path": WORKSPACE / "kashmiri_dataset/experiments/v3_unified_all_data_training/checkpoints/best_model"
    },
    {
        "id": "E22_AllData_Run1",
        "name": "EduCasheer All-Data Run 1 (Step 2250)",
        "era": "Phase 1 Expanded (160 hrs)",
        "path": WORKSPACE / "kashmiri_dataset/experiments/all_data_expanded_training/checkpoints/best_model"
    },
    {
        "id": "E22-01",
        "name": "EduCasheer E22-01 Champion",
        "era": "Stage 22 (Multi-Source Balanced)",
        "path": WORKSPACE / "kashmiri_dataset/experiments/stage22_data_expanded_training/checkpoints/E22-01/best_model"
    },
    {
        "id": "E22-02",
        "name": "EduCasheer E22-02 Full WAAZ",
        "era": "Stage 22 (Full WAAZ)",
        "path": WORKSPACE / "kashmiri_dataset/experiments/stage22_data_expanded_training/checkpoints/E22-02/best_model"
    },
    {
        "id": "E22-03",
        "name": "EduCasheer E22-03 MultiSource",
        "era": "Stage 22 (Multi-Source)",
        "path": WORKSPACE / "kashmiri_dataset/experiments/stage22_data_expanded_training/checkpoints/E22-03/best_model"
    },
    {
        "id": "E22-04",
        "name": "EduCasheer E22-04 WAAZ Prioritized",
        "era": "Stage 22 (WAAZ Oversampled)",
        "path": WORKSPACE / "kashmiri_dataset/experiments/stage22_data_expanded_training/checkpoints/E22-04/best_model"
    },
    {
        "id": "E-03",
        "name": "EduCasheer E-03 Calibrated",
        "era": "Stage 15 (MWER Calibrated)",
        "path": WORKSPACE / "kashmiri_dataset/experiments/stage15_mwer_verification_and_scaling/checkpoints/E-03/best_calib"
    },
    {
        "id": "R-02",
        "name": "EduCasheer R-02 Robustness",
        "era": "Stage 12 (Noise Augmentation)",
        "path": WORKSPACE / "kashmiri_dataset/experiments/stage12_robustness_training/R-02_combined_full/best_checkpoint"
    },
    {
        "id": "B-02B",
        "name": "EduCasheer B-02B Acoustic",
        "era": "Stage 9 (Acoustic Exploration)",
        "path": WORKSPACE / "kashmiri_dataset/experiments/native_asr_training/experiments/stage9_b02_acoustic_comparison/B-02B/pilot_checkpoints/best_model"
    },
    {
        "id": "B-01",
        "name": "EduCasheer B-01 Native Baseline",
        "era": "Original Baseline (XLSR-300M)",
        "path": WORKSPACE / "kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc/best_model"
    }
]

def ensure_free_memory(target_gb=20):
    try:
        import gc
        buf = bytearray(target_gb * 1024 * 1024 * 1024)
        del buf
        gc.collect()
    except Exception:
        pass

def get_file_sha256(filepath):
    if not filepath.exists():
        return "missing"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def compute_word_metrics(refs, hyps):
    subs, dels, ins = 0, 0, 0
    ref_words = 0
    per_utt_errs = []
    per_utt_ref_lens = []
    for r, h in zip(refs, hyps):
        res = jiwer.process_words(r, h)
        err = res.substitutions + res.deletions + res.insertions
        subs += res.substitutions
        dels += res.deletions
        ins += res.insertions
        rw = len(r.split())
        ref_words += rw
        per_utt_errs.append(err)
        per_utt_ref_lens.append(rw)
    tot_err = subs + dels + ins
    wer = (tot_err / ref_words) * 100.0 if ref_words > 0 else 0.0
    return {
        "wer": round(float(wer), 2),
        "substitutions": int(subs),
        "deletions": int(dels),
        "insertions": int(ins),
        "total_errors": int(tot_err),
        "ref_words": int(ref_words),
        "per_utt_errs": per_utt_errs,
        "per_utt_ref_lens": per_utt_ref_lens
    }

def compute_cer_metrics(refs, hyps):
    res_sp = jiwer.process_characters(refs, hyps)
    errs_sp = res_sp.substitutions + res_sp.deletions + res_sp.insertions
    denom_sp = sum(len(r) for r in refs)
    cer_with_space = (errs_sp / denom_sp) * 100.0 if denom_sp > 0 else 0.0

    refs_nosp = [r.replace(" ", "") for r in refs]
    hyps_nosp = [h.replace(" ", "") for h in hyps]
    res_nosp = jiwer.process_characters(refs_nosp, hyps_nosp)
    errs_nosp = res_nosp.substitutions + res_nosp.deletions + res_nosp.insertions
    denom_nosp = sum(len(r) for r in refs_nosp)
    cer_no_space = (errs_nosp / denom_nosp) * 100.0 if denom_nosp > 0 else 0.0

    return {
        "cer_with_space": round(float(cer_with_space), 2),
        "cer_no_space": round(float(cer_no_space), 2)
    }

def evaluate_model_on_manifest(manifest_path, model, processor, decoder, pool, batch_size=32):
    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    num_records = len(records)
    greedy_hyps, kenlm_hyps, refs = [], [], []
    sample_ids, sources, durations, speaker_ids = [], [], [], []

    total_infer_time = 0.0
    total_audio_time = 0.0

    for i in range(0, num_records, batch_size):
        batch_records = records[i:i + batch_size]
        waveforms = []
        for rec in batch_records:
            ap = rec.get("audio_path")
            if not os.path.isabs(ap):
                ap = str(WORKSPACE / ap)
            try:
                wav, sr = sf.read(ap, dtype="float32")
                if wav.ndim > 1:
                    wav = np.mean(wav, axis=1)
                if sr != 16000:
                    import torchaudio.functional as F_audio
                    w_tensor = torch.from_numpy(wav).float()
                    w_tensor = F_audio.resample(w_tensor, orig_freq=sr, new_freq=16000)
                    wav = w_tensor.numpy()
                waveforms.append(wav)
                dur = len(wav) / 16000.0
                total_audio_time += dur
            except Exception as e:
                waveforms.append(np.zeros(16000, dtype=np.float32))
                total_audio_time += 1.0

        t0 = time.time()
        inputs = processor(waveforms, sampling_rate=16000, return_tensors="pt", padding=True).input_values.to("cuda")
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(inputs).logits
        t1 = time.time()
        total_infer_time += (t1 - t0)

        p_ids = torch.argmax(logits, dim=-1)
        b_greedy = [norm_strict(h) for h in processor.batch_decode(p_ids)]

        lps = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
        b_kenlm = [norm_strict(h) for h in decoder.decode_batch(pool, lps, beam_width=16)]

        for b_idx, rec in enumerate(batch_records):
            ref = norm_strict(rec.get("text") or rec.get("raw_text") or "")
            refs.append(ref)
            greedy_hyps.append(b_greedy[b_idx])
            kenlm_hyps.append(b_kenlm[b_idx])
            sample_ids.append(rec.get("sample_id", f"utt_{i+b_idx}"))
            sources.append(rec.get("source", "unknown"))
            durations.append(float(rec.get("duration", 0.0)))
            speaker_ids.append(rec.get("speaker_id", "unknown"))

    g_word = compute_word_metrics(refs, greedy_hyps)
    k_word = compute_word_metrics(refs, kenlm_hyps)
    g_cer = compute_cer_metrics(refs, greedy_hyps)
    k_cer = compute_cer_metrics(refs, kenlm_hyps)

    predictions = []
    for idx in range(num_records):
        predictions.append({
            "sample_id": sample_ids[idx],
            "speaker_id": speaker_ids[idx],
            "source": sources[idx],
            "duration": durations[idx],
            "reference": refs[idx],
            "greedy_hypothesis": greedy_hyps[idx],
            "kenlm_hypothesis": kenlm_hyps[idx],
            "greedy_err": g_word["per_utt_errs"][idx],
            "kenlm_err": k_word["per_utt_errs"][idx],
            "ref_len": g_word["per_utt_ref_lens"][idx]
        })

    rtf = total_infer_time / max(1e-4, total_audio_time)

    return {
        "num_records": num_records,
        "total_ref_words": g_word["ref_words"],
        "total_audio_hours": round(total_audio_time / 3600.0, 3),
        "rtf": round(float(rtf), 4),
        "greedy": {
            "wer": g_word["wer"],
            "cer_with_space": g_cer["cer_with_space"],
            "cer_no_space": g_cer["cer_no_space"],
            "substitutions": g_word["substitutions"],
            "deletions": g_word["deletions"],
            "insertions": g_word["insertions"],
            "total_errors": g_word["total_errors"]
        },
        "kenlm": {
            "wer": k_word["wer"],
            "cer_with_space": k_cer["cer_with_space"],
            "cer_no_space": k_cer["cer_no_space"],
            "substitutions": k_word["substitutions"],
            "deletions": k_word["deletions"],
            "insertions": k_word["insertions"],
            "total_errors": k_word["total_errors"]
        },
        "predictions": predictions
    }

def main():
    print("=" * 80)
    print("EduCasheer Kashmiri ASR V3.0 Canonical Benchmark Evaluation")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 0. Ensure free physical memory and initialize CUDA cleanly
    print("Reclaiming system page cache to ensure available physical memory...")
    ensure_free_memory(20)
    torch.cuda.init()
    print(f"CUDA Initialized: {torch.cuda.get_device_name(0)}")

    # 1. Initialize tokenizer and decoder
    print(f"Loading Base Tokenizer from: {TOK_DIR}")
    base_proc = Wav2Vec2Processor.from_pretrained(str(TOK_DIR))
    vocab = base_proc.tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    labels = [tok if tok != "|" else " " for tok, _ in sorted_vocab]

    print(f"Building CTC KenLM Decoder (LM: {LM_PATH.name}, alpha=0.20, beta=-0.50, beam=16)...")
    decoder = build_ctcdecoder(
        labels=labels,
        kenlm_model_path=str(LM_PATH),
        alpha=0.20,
        beta=-0.50,
        unk_score_offset=-10.0
    )

    all_leaderboard_results = {}
    intermediate_path = OUTPUT_DIR / "v3_leaderboard_intermediate.json"
    if intermediate_path.exists():
        try:
            with open(intermediate_path, "r", encoding="utf-8") as inf:
                all_leaderboard_results = json.load(inf)
                print(f"Loaded {len(all_leaderboard_results)} cached model evaluations from {intermediate_path.name}")
        except Exception:
            pass

    num_threads = min(12, multiprocessing.cpu_count())
    print(f"Starting decoding pool with {num_threads} worker threads...")

    with multiprocessing.get_context("fork").Pool(num_threads) as pool:
        for m_idx, m_info in enumerate(CANDIDATE_MODELS, 1):
            mid = m_info["id"]
            mname = m_info["name"]
            mera = m_info["era"]
            mpath = m_info["path"]

            print("\n" + "#" * 80)
            print(f"[{m_idx}/{len(CANDIDATE_MODELS)}] EVALUATING: {mname} ({mid})")
            print(f"Era: {mera} | Path: {mpath}")
            print("#" * 80)

            if not mpath.exists():
                print(f"[ERROR] Path does not exist: {mpath}")
                continue

            weights_file = mpath / "model.safetensors"
            if not weights_file.exists():
                weights_file = mpath / "pytorch_model.bin"
            sha256 = get_file_sha256(weights_file)
            print(f"Model Weights: {weights_file.name} | SHA-256: {sha256}")

            # Check if all splits already evaluated for this model
            if mid in all_leaderboard_results and all(sk in all_leaderboard_results[mid]["splits"] for sk, _, _ in BENCHMARK_SPLITS):
                print(f"[CACHE HIT] All splits already evaluated for {mid}. Skipping model reload.")
                continue

            # Load model onto GPU
            t_load0 = time.time()
            processor = Wav2Vec2Processor.from_pretrained(str(mpath if (mpath / "vocab.json").exists() else TOK_DIR))
            model = Wav2Vec2ForCTC.from_pretrained(str(mpath)).to("cuda")
            model.eval()
            print(f"Model loaded in {time.time() - t_load0:.2f}s")

            model_split_results = {}
            if mid in all_leaderboard_results:
                model_split_results = all_leaderboard_results[mid].get("splits", {})

            for split_key, split_title, split_path in BENCHMARK_SPLITS:
                if split_key in model_split_results:
                    print(f"  [CACHE HIT] {split_title} already computed. Skipping.")
                    continue

                print(f"  --> Running Evaluation on {split_title} ({split_path.name})...")
                res = evaluate_model_on_manifest(
                    split_path,
                    model,
                    processor,
                    decoder,
                    pool,
                    batch_size=32
                )

                # Save predictions
                pred_file = PRED_DIR / f"{mid}_{split_key}_predictions.jsonl"
                with open(pred_file, "w", encoding="utf-8") as pf:
                    for pred in res["predictions"]:
                        pf.write(json.dumps(pred, ensure_ascii=False) + "\n")

                del res["predictions"]
                model_split_results[split_key] = res

                print(f"      [RESULT] Greedy WER: {res['greedy']['wer']:5.2f}% | KenLM WER: {res['kenlm']['wer']:5.2f}% | CER: {res['kenlm']['cer_no_space']:5.2f}% | RTF: {res['rtf']:.4f}")

            all_leaderboard_results[mid] = {
                "id": mid,
                "name": mname,
                "era": mera,
                "path": str(mpath),
                "sha256": sha256,
                "splits": model_split_results
            }

            # Save intermediate results
            with open(intermediate_path, "w", encoding="utf-8") as outf:
                json.dump(all_leaderboard_results, outf, indent=2, ensure_ascii=False)

            # Cleanup GPU memory
            del model
            del processor
            gc.collect()
            torch.cuda.empty_cache()
            ensure_free_memory(5)

    # Save final comprehensive summary
    final_summary_path = OUTPUT_DIR / "v3_leaderboard_summary.json"
    with open(final_summary_path, "w", encoding="utf-8") as outf:
        json.dump(all_leaderboard_results, outf, indent=2, ensure_ascii=False)
    print("\n" + "=" * 80)
    print(f"EVALUATION COMPLETE! Leaderboard saved to: {final_summary_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
