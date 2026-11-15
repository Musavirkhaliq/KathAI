#!/usr/bin/env python3
"""
EduCasheer Kashmiri ASR Benchmark Evaluation Engine
===================================================
Evaluates acoustic and baseline models across benchmark manifests (Pilot or Full Benchmark):
- Models supported:
    1. EduCasheer B-02C (Greedy CTC and CTC Beam Search + KenLM)
    2. EduCasheer B-02B (Greedy CTC and CTC Beam Search + KenLM)
    3. EduCasheer B-02A (Greedy CTC and CTC Beam Search + KenLM)
    4. EduCasheer B-01 (Greedy CTC and CTC Beam Search + KenLM)
    5. AI4Bharat IndicConformer 600M (CTC and RNN-T)
    6. Aazu34 Conformer CTC
    7. Khushi wav2vec2 CTC

Produces:
- Tier-wise (Tier A, Tier B, Tier C) metrics
- Aggregate benchmark metrics
- S / D / I breakdown
- RTF and latency measurements
- Per-utterance prediction files
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

WORKSPACE_DIR = Path("/home/musa/Kashmiri Project")
DATASET_DIR = WORKSPACE_DIR / "kashmiri_dataset"
EXP_ROOT = DATASET_DIR / "experiments"
STAGE9_DIR = EXP_ROOT / "native_asr_training/experiments/stage9_b02_acoustic_comparison"
TOK_DIR = EXP_ROOT / "native_asr_training/experiments/B-01-xlsr300m-character-ctc/tokenizer"
STAGE7_LM = EXP_ROOT / "native_asr_training/experiments/stage7_ctc_lm_decoding/language_models/lm_word_4gram_domain_d.arpa"

sys.path.insert(0, str(DATASET_DIR / "scripts"))
from metrics_evaluator import calculate_wer_and_cer, normalize_perso_arabic_kashmiri
import jiwer

def load_audio_16k(audio_path):
    p = Path(audio_path)
    if not p.is_absolute():
        p = WORKSPACE_DIR / p
    wav, sr = sf.read(str(p), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import torchaudio
        t_wav = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
        wav = torchaudio.transforms.Resample(sr, 16000)(t_wav).squeeze(0).numpy()
    return wav

def evaluate_educasheer_model(model_name, checkpoint_path, manifest_path, output_dir, device="cuda", run_kenlm=True):
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    from pyctcdecode import build_ctcdecoder

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[EduCasheer Evaluator] Loading {model_name} from: {checkpoint_path}")
    processor = Wav2Vec2Processor.from_pretrained(str(TOK_DIR))
    model = Wav2Vec2ForCTC.from_pretrained(str(checkpoint_path)).to(device)
    model.eval()

    decoder_kenlm = None
    if run_kenlm:
        vocab = processor.tokenizer.get_vocab()
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        labels = [tok if tok != '|' else ' ' for tok, _ in sorted_vocab]
        decoder_kenlm = build_ctcdecoder(
            labels=labels,
            kenlm_model_path=str(STAGE7_LM),
            alpha=0.25,
            beta=0.00
        )

    samples = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"Evaluating {len(samples)} samples on {device}...")

    results_greedy = []
    results_kenlm = []

    total_audio_sec = 0.0
    total_infer_sec = 0.0

    for idx, s in enumerate(samples):
        wav = load_audio_16k(s["audio_path"])
        dur = len(wav) / 16000.0
        total_audio_sec += dur

        t0 = time.time()
        input_values = processor(wav, sampling_rate=16000, return_tensors="pt").input_values.to(device)
        with torch.no_grad():
            logits = model(input_values).logits[0] # [T, V]
        t1 = time.time()
        total_infer_sec += (t1 - t0)

        # Greedy CTC
        pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
        pred_raw = processor.decode(pred_ids)
        pred_norm = normalize_perso_arabic_kashmiri(pred_raw)

        ref_raw = s.get("raw_text", s.get("text", "")).strip()
        ref_norm = normalize_perso_arabic_kashmiri(ref_raw)

        tier = s.get("tier", "unknown")
        spk = s.get("speaker_id", "unknown")

        res_g = {
            "sample_id": s["sample_id"],
            "tier": tier,
            "speaker_id": spk,
            "duration": dur,
            "ref_raw": ref_raw,
            "ref_norm": ref_norm,
            "hyp_raw": pred_raw,
            "hyp_norm": pred_norm,
            "infer_time_sec": t1 - t0
        }
        results_greedy.append(res_g)

        if decoder_kenlm is not None:
            logits_np = logits.cpu().numpy()
            hyp_kenlm_raw = decoder_kenlm.decode(logits_np)
            hyp_kenlm_norm = normalize_perso_arabic_kashmiri(hyp_kenlm_raw)
            res_k = dict(res_g)
            res_k["hyp_raw"] = hyp_kenlm_raw
            res_k["hyp_norm"] = hyp_kenlm_norm
            results_kenlm.append(res_k)

        if (idx + 1) % 50 == 0 or idx + 1 == len(samples):
            print(f"  Processed {idx + 1}/{len(samples)} clips ({total_audio_sec:.1f}s audio)...")

    # Compute metrics function
    def compute_metrics(res_list, name):
        refs = [r["ref_norm"] for r in res_list]
        hyps = [r["hyp_norm"] for r in res_list]
        out_w = jiwer.process_words(refs, hyps)
        out_c = jiwer.process_characters(refs, hyps)
        
        # Tier-wise breakdown
        tiers = sorted(set(r["tier"] for r in res_list))
        tier_metrics = {}
        for t in tiers:
            t_refs = [r["ref_norm"] for r in res_list if r["tier"] == t]
            t_hyps = [r["hyp_norm"] for r in res_list if r["tier"] == t]
            t_w = jiwer.process_words(t_refs, t_hyps)
            t_c = jiwer.process_characters(t_refs, t_hyps)
            tier_metrics[t] = {
                "samples": len(t_refs),
                "norm_wer": round(t_w.wer * 100, 2),
                "norm_cer": round(t_c.cer * 100, 2),
                "sub_rate": round(t_w.substitutions / max(1, t_w.substitutions + t_w.deletions + t_w.hits) * 100, 2),
                "del_rate": round(t_w.deletions / max(1, t_w.substitutions + t_w.deletions + t_w.hits) * 100, 2),
                "ins_rate": round(t_w.insertions / max(1, t_w.substitutions + t_w.deletions + t_w.hits) * 100, 2)
            }

        return {
            "model_name": name,
            "total_samples": len(refs),
            "aggregate_norm_wer": round(out_w.wer * 100, 2),
            "aggregate_norm_cer": round(out_c.cer * 100, 2),
            "word_hits": out_w.hits,
            "word_sub": out_w.substitutions,
            "word_del": out_w.deletions,
            "word_ins": out_w.insertions,
            "sub_rate_pct": round(out_w.substitutions / max(1, out_w.substitutions + out_w.deletions + out_w.hits) * 100, 2),
            "del_rate_pct": round(out_w.deletions / max(1, out_w.substitutions + out_w.deletions + out_w.hits) * 100, 2),
            "ins_rate_pct": round(out_w.insertions / max(1, out_w.substitutions + out_w.deletions + out_w.hits) * 100, 2),
            "total_audio_sec": round(total_audio_sec, 2),
            "total_infer_sec": round(total_infer_sec, 2),
            "rtf": round(total_infer_sec / max(0.1, total_audio_sec), 4),
            "tier_metrics": tier_metrics
        }

    summary = {}
    summary["greedy"] = compute_metrics(results_greedy, f"{model_name}_greedy")
    
    # Save predictions
    with open(output_dir / f"predictions_{model_name}_greedy.jsonl", "w", encoding="utf-8") as f:
        for r in results_greedy: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if run_kenlm:
        summary["kenlm"] = compute_metrics(results_kenlm, f"{model_name}_kenlm")
        with open(output_dir / f"predictions_{model_name}_kenlm.jsonl", "w", encoding="utf-8") as f:
            for r in results_kenlm: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(output_dir / f"eval_summary_{model_name}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[{model_name}] Greedy WER: {summary['greedy']['aggregate_norm_wer']}% (CER: {summary['greedy']['aggregate_norm_cer']}%) | RTF: {summary['greedy']['rtf']}")
    if run_kenlm:
        print(f"[{model_name}] KenLM  WER: {summary['kenlm']['aggregate_norm_wer']}% (CER: {summary['kenlm']['aggregate_norm_cer']}%)")

    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest JSONL")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save evaluation results")
    parser.add_argument("--models", nargs="+", default=["B-02C", "B-02B", "B-02A", "B-01"], help="Models to benchmark")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_lm", action="store_true", help="Skip KenLM")
    args = parser.parse_args()

    model_ckpts = {
        "B-02C": STAGE9_DIR / "B-02C/checkpoints/best_model",
        "B-02B": STAGE9_DIR / "B-02B/checkpoints/best_model",
        "B-02A": STAGE9_DIR / "B-02A/checkpoints/best_model",
        "B-01": EXP_ROOT / "native_asr_training/experiments/B-01-xlsr300m-character-ctc/best_model"
    }

    all_summaries = {}
    for m in args.models:
        if m in model_ckpts:
            s = evaluate_educasheer_model(
                model_name=m,
                checkpoint_path=model_ckpts[m],
                manifest_path=args.manifest,
                output_dir=args.output_dir,
                device=args.device,
                run_kenlm=not args.no_lm
            )
            all_summaries[m] = s

    # Save comparative summary
    comp_file = Path(args.output_dir) / "benchmark_comparative_summary.json"
    with open(comp_file, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    print(f"\nSaved consolidated summary to: {comp_file}")

if __name__ == "__main__":
    main()
