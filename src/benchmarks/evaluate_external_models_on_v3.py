#!/usr/bin/env python3
"""
Official V3.0 Canonical Benchmark: External Open-Source Model Evaluation
========================================================================
Evaluates public and open-source Kashmiri ASR models on the newly established
canonical V3 benchmark splits:
  1. test_standard_v3.jsonl (1,500 utts, 3.19 hrs) - Standard Multi-Source Test
  2. test_community_waaz_v3.jsonl (114 utts, 0.19 hrs) - Real-World WAAZ Community Test
  3. val_community_waaz_v3.jsonl (309 utts, 0.83 hrs) - Real-World WAAZ Community Val

External Models Evaluated:
  1. AI4Bharat IndicConformer 600M RNN-T (Conformer Hybrid 600M)
  2. AI4Bharat IndicConformer 600M CTC (Conformer Hybrid 600M)
  3. Aazu34 Kashmiri ASR CTC (Conformer 115M INT8 ONNX)
  4. Khushi wav2vec2-large-xlsr-53 (317M)
  5. Bodhan-AI IndicTranscribe Canary FastConformer (1.2B)
"""

import os
import sys
import json
import time
import gc
import unicodedata
from pathlib import Path
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch
import torchaudio
import jiwer

WORKSPACE = Path("/home/musa/Kashmiri Project")
DATASET_DIR = WORKSPACE / "kashmiri_dataset"
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(DATASET_DIR / "scripts"))
sys.path.insert(0, str(DATASET_DIR / "experiments/external_asr_benchmark/inference_adapters"))

from metrics_evaluator import normalize_perso_arabic_kashmiri as norm_strict

OUTPUT_DIR = DATASET_DIR / "experiments/v3_benchmark_evaluations"
PRED_DIR = OUTPUT_DIR / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = DATASET_DIR / "experiments/external_asr_benchmark/models"

BENCHMARK_SPLITS = [
    ("test_standard_v3", "Test Standard (1,500 utts)", DATASET_DIR / "releases/asr_v3_unified_splits/test_standard_v3.jsonl"),
    ("test_community_waaz_v3", "Test Community WAAZ (114 utts)", DATASET_DIR / "releases/asr_v3_unified_splits/test_community_waaz_v3.jsonl"),
    ("val_community_waaz_v3", "Val Community WAAZ (309 utts)", DATASET_DIR / "releases/asr_v3_unified_splits/val_community_waaz_v3.jsonl"),
]

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

def load_audio_16k(audio_path):
    p = Path(audio_path)
    if not p.is_absolute():
        p = WORKSPACE / p
    wav, sr = sf.read(str(p), dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    if sr != 16000:
        t_wav = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
        wav = torchaudio.transforms.Resample(sr, 16000)(t_wav).squeeze(0).numpy()
    return wav

def run_eval_for_model(mid, mname, predict_fn, cleanup_fn=None):
    results = {}
    summary_file = OUTPUT_DIR / "external_models_summary.json"
    existing_all = {}
    if summary_file.exists():
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                existing_all = json.load(f)
        except Exception:
            pass

    for split_key, split_title, split_path in BENCHMARK_SPLITS:
        pred_file = PRED_DIR / f"{mid}_{split_key}_predictions.jsonl"

        # Check if already cached
        if pred_file.exists() and mid in existing_all and split_key in existing_all[mid].get("splits", {}):
            print(f"  [CACHE HIT] {mid} on {split_title} already computed. Loading...", flush=True)
            results[split_key] = existing_all[mid]["splits"][split_key]
            continue

        records = []
        with open(split_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line.strip()))

        print(f"  --> Running {mid} on {split_title} ({len(records)} utts)...", flush=True)

        refs, hyps, sample_ids, sources, durations, speaker_ids = [], [], [], [], [], []
        total_audio_sec = 0.0
        total_infer_sec = 0.0

        t_start = time.time()
        for idx, rec in enumerate(records):
            sid = rec.get("sample_id", f"utt_{idx}")
            ap = rec.get("audio_path")
            if not os.path.isabs(ap):
                ap = str(WORKSPACE / ap)
            ref_raw = (rec.get("text") or rec.get("raw_text") or "").strip()
            ref_norm = norm_strict(ref_raw)
            dur = float(rec.get("duration", 0.0))
            if dur <= 0.0:
                try:
                    wav = load_audio_16k(ap)
                    dur = len(wav) / 16000.0
                except Exception:
                    dur = 1.0

            total_audio_sec += dur

            t0 = time.time()
            try:
                raw_pred = predict_fn(ap)
            except Exception as e:
                print(f"      [WARN] Prediction failed for {sid}: {e}", flush=True)
                raw_pred = ""
            t1 = time.time()
            infer_sec = t1 - t0
            total_infer_sec += infer_sec

            hyp_norm = norm_strict(raw_pred)

            refs.append(ref_norm)
            hyps.append(hyp_norm)
            sample_ids.append(sid)
            sources.append(rec.get("source", "unknown"))
            durations.append(dur)
            speaker_ids.append(rec.get("speaker_id", "unknown"))

            if (idx + 1) % 250 == 0 or (idx + 1) == len(records):
                print(f"      [{idx+1}/{len(records)}] elapsed: {time.time()-t_start:.1f}s | RTF: {total_infer_sec / max(1e-4, total_audio_sec):.4f}", flush=True)

        word_metrics = compute_word_metrics(refs, hyps)
        cer_metrics = compute_cer_metrics(refs, hyps)
        rtf = total_infer_sec / max(1e-4, total_audio_sec)

        predictions = []
        for idx in range(len(records)):
            predictions.append({
                "sample_id": sample_ids[idx],
                "speaker_id": speaker_ids[idx],
                "source": sources[idx],
                "duration": durations[idx],
                "reference": refs[idx],
                "hypothesis": hyps[idx],
                "error": word_metrics["per_utt_errs"][idx],
                "ref_len": word_metrics["per_utt_ref_lens"][idx]
            })

        with open(pred_file, "w", encoding="utf-8") as pf:
            for pred in predictions:
                pf.write(json.dumps(pred, ensure_ascii=False) + "\n")

        split_summary = {
            "num_records": len(records),
            "total_ref_words": word_metrics["ref_words"],
            "total_audio_hours": round(total_audio_sec / 3600.0, 3),
            "rtf": round(float(rtf), 4),
            "wer": word_metrics["wer"],
            "cer_with_space": cer_metrics["cer_with_space"],
            "cer_no_space": cer_metrics["cer_no_space"],
            "substitutions": word_metrics["substitutions"],
            "deletions": word_metrics["deletions"],
            "insertions": word_metrics["insertions"],
            "total_errors": word_metrics["total_errors"]
        }
        results[split_key] = split_summary
        print(f"      [RESULT] {mid} on {split_title} -> WER: {word_metrics['wer']:5.2f}% | CER: {cer_metrics['cer_no_space']:5.2f}% | RTF: {rtf:.4f}", flush=True)

    if cleanup_fn:
        cleanup_fn()

    return results

def main():
    print("=" * 80, flush=True)
    print("EduCasheer Kashmiri ASR V3.0 Canonical Benchmark: External Model Runner", flush=True)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 80, flush=True)

    summary_file = OUTPUT_DIR / "external_models_summary.json"
    all_external_results = {}
    if summary_file.exists():
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                all_external_results = json.load(f)
                print(f"Loaded existing results for {list(all_external_results.keys())}", flush=True)
        except Exception:
            pass

    # ---------------------------------------------------------
    # 1. Aazu34 Conformer CTC (115M INT8 ONNX)
    # ---------------------------------------------------------
    mid = "aazu34_conformer_ctc"
    if mid not in all_external_results or len(all_external_results[mid].get("splits", {})) < len(BENCHMARK_SPLITS):
        print("\n" + "#" * 80, flush=True)
        print("MODEL [1/5]: Aazu34/kashmiri-asr-ctc (115M INT8 ONNX)", flush=True)
        print("#" * 80, flush=True)
        from adapter_aazu34 import Aazu34Model
        aazu_dir = MODELS_DIR / "Aazu34_kashmiri-asr-ctc"
        aazu_model = Aazu34Model(aazu_dir)

        def predict_aazu(ap):
            wav = load_audio_16k(ap)
            return aazu_model.transcribe(wav)

        splits_res = run_eval_for_model(mid, "Aazu34 Conformer CTC (115M)", predict_aazu)
        all_external_results[mid] = {
            "id": mid,
            "name": "Aazu34 Conformer CTC",
            "architecture": "Conformer CTC (INT8 ONNX)",
            "parameters": "115M",
            "origin": "Aazu34 (HuggingFace)",
            "splits": splits_res
        }
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_external_results, f, indent=2, ensure_ascii=False)
        del aazu_model
        gc.collect()

    # ---------------------------------------------------------
    # 2. Khushi wav2vec2-large-xlsr-53 (317M)
    # ---------------------------------------------------------
    mid = "khushi_wav2vec2_xlsr53"
    if mid not in all_external_results or len(all_external_results[mid].get("splits", {})) < len(BENCHMARK_SPLITS):
        print("\n" + "#" * 80, flush=True)
        print("MODEL [2/5]: khushi1234455687/wav2vec2-large-xlsr-53-Kashmiri-Version6 (317M)", flush=True)
        print("#" * 80, flush=True)
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
        khushi_dir = MODELS_DIR / "khushi1234455687_wav2vec2-large-xlsr-53-Kashmiri-Version6"
        processor = Wav2Vec2Processor.from_pretrained(str(khushi_dir))
        model = Wav2Vec2ForCTC.from_pretrained(str(khushi_dir)).to("cuda")
        model.eval()

        def predict_khushi(ap):
            wav = load_audio_16k(ap)
            inputs = processor(wav, sampling_rate=16000, return_tensors="pt").input_values.to("cuda")
            with torch.no_grad():
                logits = model(inputs).logits
            p_ids = torch.argmax(logits, dim=-1)
            return processor.batch_decode(p_ids)[0]

        def cleanup_khushi():
            nonlocal model, processor
            del model
            del processor
            gc.collect()
            torch.cuda.empty_cache()

        splits_res = run_eval_for_model(mid, "Khushi wav2vec2 XLSR-53", predict_khushi, cleanup_khushi)
        all_external_results[mid] = {
            "id": mid,
            "name": "Khushi wav2vec2 XLSR-53",
            "architecture": "wav2vec2-large-xlsr-53",
            "parameters": "317M",
            "origin": "Khushi (HuggingFace)",
            "splits": splits_res
        }
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_external_results, f, indent=2, ensure_ascii=False)

    # ---------------------------------------------------------
    # 3. AI4Bharat IndicConformer 600M CTC
    # ---------------------------------------------------------
    mid = "ai4bharat_indicconformer_600m_ctc"
    if mid not in all_external_results or len(all_external_results[mid].get("splits", {})) < len(BENCHMARK_SPLITS):
        print("\n" + "#" * 80, flush=True)
        print("MODEL [3/5]: ai4bharat/indic-conformer-600m-multilingual (CTC Mode, 600M)", flush=True)
        print("#" * 80, flush=True)
        from adapter_indicconformer_600m import IndicConformer600M
        indic_dir = MODELS_DIR / "ai4bharat_indic-conformer-600m-multilingual"
        m_ctc = IndicConformer600M(indic_dir, device="cpu", decoding="ctc")

        def predict_indic_ctc(ap):
            wav_tensor = m_ctc.load_audio(ap)
            enc, enc_len = m_ctc.encode(wav_tensor)
            return m_ctc.ctc_decode(enc, enc_len, lang="ks")

        splits_res = run_eval_for_model(mid, "AI4Bharat IndicConformer 600M CTC", predict_indic_ctc)
        all_external_results[mid] = {
            "id": mid,
            "name": "AI4Bharat IndicConformer 600M CTC",
            "architecture": "Conformer Hybrid (CTC)",
            "parameters": "600M",
            "origin": "AI4Bharat",
            "splits": splits_res
        }
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_external_results, f, indent=2, ensure_ascii=False)
        del m_ctc
        gc.collect()

    # ---------------------------------------------------------
    # 4. AI4Bharat IndicConformer 600M RNN-T
    # ---------------------------------------------------------
    mid = "ai4bharat_indicconformer_600m_rnnt"
    if mid not in all_external_results or len(all_external_results[mid].get("splits", {})) < len(BENCHMARK_SPLITS):
        print("\n" + "#" * 80, flush=True)
        print("MODEL [4/5]: ai4bharat/indic-conformer-600m-multilingual (RNN-T Mode, 600M)", flush=True)
        print("#" * 80, flush=True)
        from adapter_indicconformer_600m import IndicConformer600M
        indic_dir = MODELS_DIR / "ai4bharat_indic-conformer-600m-multilingual"
        m_rnnt = IndicConformer600M(indic_dir, device="cpu", decoding="rnnt")

        def predict_indic_rnnt(ap):
            wav_tensor = m_rnnt.load_audio(ap)
            enc, enc_len = m_rnnt.encode(wav_tensor)
            return m_rnnt.rnnt_decode(enc, enc_len, lang="ks")

        splits_res = run_eval_for_model(mid, "AI4Bharat IndicConformer 600M RNN-T", predict_indic_rnnt)
        all_external_results[mid] = {
            "id": mid,
            "name": "AI4Bharat IndicConformer 600M RNN-T",
            "architecture": "Conformer Hybrid (RNN-T Joint Network)",
            "parameters": "600M",
            "origin": "AI4Bharat",
            "splits": splits_res
        }
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_external_results, f, indent=2, ensure_ascii=False)
        del m_rnnt
        gc.collect()

    # ---------------------------------------------------------
    # 5. Bodhan-AI IndicTranscribe Canary FastConformer (1.2B)
    # ---------------------------------------------------------
    mid = "bodhan_ai_indic_transcribe_core"
    if mid not in all_external_results or len(all_external_results[mid].get("splits", {})) < len(BENCHMARK_SPLITS):
        print("\n" + "#" * 80, flush=True)
        print("MODEL [5/5]: bodhan-ai/indic-transcribe-core (Canary FastConformer 1.2B on CUDA)", flush=True)
        print("#" * 80, flush=True)
        bodhan_dir = MODELS_DIR / "bodhan-ai_indic-transcribe-core"
        sys.path.insert(0, str(bodhan_dir))
        from indic_transcribe import IndicTranscribe
        asr_bodhan = IndicTranscribe.from_pretrained(str(bodhan_dir), device="cuda")

        def predict_bodhan(ap):
            return asr_bodhan.transcribe(ap, lang="ks", mode="native")

        def cleanup_bodhan():
            nonlocal asr_bodhan
            del asr_bodhan
            gc.collect()
            torch.cuda.empty_cache()

        splits_res = run_eval_for_model(mid, "Bodhan-AI IndicTranscribe (1.2B)", predict_bodhan, cleanup_bodhan)
        all_external_results[mid] = {
            "id": mid,
            "name": "Bodhan-AI IndicTranscribe",
            "architecture": "Canary FastConformer + Transformer Decoder",
            "parameters": "1.2B",
            "origin": "Bodhan-AI",
            "splits": splits_res
        }
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_external_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80, flush=True)
    print("ALL EXTERNAL MODEL EVALUATIONS COMPLETE!", flush=True)
    print(f"Summary saved to: {summary_file}", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
