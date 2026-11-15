#!/usr/bin/env python3
"""
Stage 9: Unified Multi-Decoder Evaluation Engine
================================================
Evaluates any Stage 9 acoustic model checkpoint (or B-01 baseline) on all 6,728
validation samples across:
1. Primary Greedy CTC decoding (unpadded, masked)
2. CTC Beam Search (beam_width=16, no LM)
3. CTC Beam Search + Stage 7 4-gram Kashmiri KenLM (beam_width=16, alpha=0.25, beta=0.00)

Produces:
- Corpus-level Normalized WER & CER
- Corpus-level Raw WER & CER
- Source-wise breakdown
- Complete per-utterance predictions
"""

import os
import sys
import json
import time
import math
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pyctcdecode import build_ctcdecoder

WORKSPACE_DIR = Path("/home/musa/Kashmiri Project")
DATASET_DIR = WORKSPACE_DIR / "kashmiri_dataset"
EXP_ROOT = DATASET_DIR / "experiments/native_asr_training/experiments"
STAGE9_DIR = EXP_ROOT / "stage9_b02_acoustic_comparison"
TOK_DIR = EXP_ROOT / "B-01-xlsr300m-character-ctc/tokenizer"
VAL_MANIFEST = DATASET_DIR / "experiments/native_asr_training/data_splits/val.jsonl"
STAGE7_4GRAM_LM = EXP_ROOT / "stage7_ctc_lm_decoding/language_models/lm_word_4gram_domain_d.arpa"

sys.path.insert(0, str(DATASET_DIR / "scripts"))
from metrics_evaluator import calculate_wer_and_cer, normalize_perso_arabic_kashmiri

def evaluate_checkpoint(model_dir, output_dir, batch_size=32):
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"EVALUATING MODEL CHECKPOINT: {model_dir.name}")
    print(f"Model Path: {model_dir}")
    print(f"Device: {device}")
    print("=" * 70)

    processor = Wav2Vec2Processor.from_pretrained(str(model_dir if (model_dir / "vocab.json").exists() else TOK_DIR))
    model = Wav2Vec2ForCTC.from_pretrained(str(model_dir)).to(device)
    model.eval()

    # Build pyctcdecode decoders
    vocab = processor.tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    labels = [tok for tok, idx in sorted_vocab]
    labels_for_pyctc = [tok if tok != '|' else ' ' for tok in labels]

    print("Building CTC Beam Search Decoders...")
    decoder_no_lm = build_ctcdecoder(labels=labels_for_pyctc, kenlm_model_path=None)
    decoder_kenlm = build_ctcdecoder(
        labels=labels_for_pyctc,
        kenlm_model_path=str(STAGE7_4GRAM_LM),
        alpha=0.25,
        beta=0.00
    )

    # Load validation set
    val_samples = []
    with open(VAL_MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            dur = float(d.get("duration", 0))
            if 0.5 <= dur <= 16.0:
                raw_txt = d.get("text", "").strip()
                norm_txt = normalize_perso_arabic_kashmiri(raw_txt)
                val_samples.append({
                    "sample_id": d["sample_id"],
                    "audio_path": d["audio_path"],
                    "duration": dur,
                    "raw_text": raw_txt,
                    "normalized_text": norm_txt,
                    "source": d.get("source", d.get("dataset_source", "unknown"))
                })

    print(f"Loaded {len(val_samples):,} validation samples.")

    # Storage for predictions
    preds_greedy = []
    preds_beam_no_lm = []
    preds_beam_kenlm = []

    t_start = time.time()
    with torch.no_grad():
        for i in range(0, len(val_samples), batch_size):
            batch = val_samples[i:i+batch_size]
            wavs = []
            valid_items = []
            for item in batch:
                p = WORKSPACE_DIR / item["audio_path"]
                try:
                    wav, sr = sf.read(str(p), dtype="float32")
                    if sr != 16000:
                        import torchaudio
                        t_audio = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
                        wav = torchaudio.transforms.Resample(sr, 16000)(t_audio).squeeze(0).numpy()
                    wavs.append(wav)
                    valid_items.append(item)
                except Exception:
                    continue

            if not wavs:
                continue

            inputs = processor(wavs, sampling_rate=16000, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(device)
            attention_mask = inputs.attention_mask.to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_values=input_values, attention_mask=attention_mask)
                logits = outputs.logits  # [B, T, V]
                log_probs = torch.log_softmax(logits, dim=-1).cpu().float().numpy()

            # Greedy decoding
            pred_ids = torch.argmax(logits, dim=-1)
            for b in range(len(valid_items)):
                item = valid_items[b]
                p_ids = pred_ids[b]
                raw_g = processor.decode(p_ids)
                norm_g = normalize_perso_arabic_kashmiri(raw_g)
                preds_greedy.append({
                    "sample_id": item["sample_id"],
                    "source": item["source"],
                    "raw_ref": item["raw_text"],
                    "norm_ref": item["normalized_text"],
                    "raw_hyp": raw_g,
                    "norm_hyp": norm_g
                })

                # Beam decoding on individual log_prob slice
                # Compute actual acoustic sequence length
                actual_len = int(math.ceil(len(wavs[b]) / 320.0))  # wav2vec2 downsamples 320x (16000 / 50 = 320)
                lp_slice = log_probs[b, :actual_len, :]

                # Beam No LM
                raw_b_no_lm = decoder_no_lm.decode(lp_slice, beam_width=16)
                norm_b_no_lm = normalize_perso_arabic_kashmiri(raw_b_no_lm)
                preds_beam_no_lm.append({
                    "sample_id": item["sample_id"],
                    "source": item["source"],
                    "raw_ref": item["raw_text"],
                    "norm_ref": item["normalized_text"],
                    "raw_hyp": raw_b_no_lm,
                    "norm_hyp": norm_b_no_lm
                })

                # Beam + KenLM
                raw_b_lm = decoder_kenlm.decode(lp_slice, beam_width=16)
                norm_b_lm = normalize_perso_arabic_kashmiri(raw_b_lm)
                preds_beam_kenlm.append({
                    "sample_id": item["sample_id"],
                    "source": item["source"],
                    "raw_ref": item["raw_text"],
                    "norm_ref": item["normalized_text"],
                    "raw_hyp": raw_b_lm,
                    "norm_hyp": norm_b_lm
                })

            if (i // batch_size) % 10 == 0:
                print(f"Evaluated {min(len(val_samples), i + batch_size):,}/{len(val_samples):,} samples ({time.time()-t_start:.1f}s)...")

    # Metrics calculation
    def score_preds(preds, name):
        w_err_norm, total_w_norm = 0, 0
        c_err_norm, total_c_norm = 0, 0
        w_err_raw, total_w_raw = 0, 0
        c_err_raw, total_c_raw = 0, 0
        by_source = {}

        for p in preds:
            src = p["source"]
            if src not in by_source:
                by_source[src] = {"w_err_norm": 0, "total_w_norm": 0, "c_err_norm": 0, "total_c_norm": 0}

            rn = calculate_wer_and_cer(p["norm_ref"], p["norm_hyp"])
            rr = calculate_wer_and_cer(p["raw_ref"], p["raw_hyp"])

            w_err_norm += rn["word_dist"]
            total_w_norm += rn["ref_words"]
            c_err_norm += rn["char_dist"]
            total_c_norm += rn["ref_chars"]

            w_err_raw += rr["word_dist"]
            total_w_raw += rr["ref_words"]
            c_err_raw += rr["char_dist"]
            total_c_raw += rr["ref_chars"]

            by_source[src]["w_err_norm"] += rn["word_dist"]
            by_source[src]["total_w_norm"] += rn["ref_words"]
            by_source[src]["c_err_norm"] += rn["char_dist"]
            by_source[src]["total_c_norm"] += rn["ref_chars"]

        source_metrics = {}
        for src, d in by_source.items():
            source_metrics[src] = {
                "norm_wer": d["w_err_norm"] / max(1, d["total_w_norm"]),
                "norm_cer": d["c_err_norm"] / max(1, d["total_c_norm"]),
                "samples": sum(1 for p in preds if p["source"] == src)
            }

        return {
            "name": name,
            "norm_wer": w_err_norm / max(1, total_w_norm),
            "norm_cer": c_err_norm / max(1, total_c_norm),
            "raw_wer": w_err_raw / max(1, total_w_raw),
            "raw_cer": c_err_raw / max(1, total_c_raw),
            "source_metrics": source_metrics
        }

    res_greedy = score_preds(preds_greedy, "Greedy CTC")
    res_beam_no_lm = score_preds(preds_beam_no_lm, "Beam Search (W=16, No LM)")
    res_beam_kenlm = score_preds(preds_beam_kenlm, "Beam Search (W=16, KenLM 4-gram)")

    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS FOR {model_dir.name}")
    print("=" * 70)
    for res in [res_greedy, res_beam_no_lm, res_beam_kenlm]:
        print(f"{res['name']:<35} | Norm WER: {res['norm_wer']*100:.2f}% | Norm CER: {res['norm_cer']*100:.2f}% | Raw WER: {res['raw_wer']*100:.2f}%")

    # Save summary
    eval_summary = {
        "model_dir": str(model_dir),
        "total_samples": len(preds_greedy),
        "greedy": res_greedy,
        "beam_no_lm": res_beam_no_lm,
        "beam_kenlm": res_beam_kenlm
    }

    with open(output_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2, ensure_ascii=False)

    for preds, fname in [
        (preds_greedy, "predictions_greedy.jsonl"),
        (preds_beam_no_lm, "predictions_beam_no_lm.jsonl"),
        (preds_beam_kenlm, "predictions_beam_kenlm.jsonl")
    ]:
        with open(output_dir / fname, "w", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Results and predictions saved to {output_dir}")
    return eval_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    evaluate_checkpoint(args.model_dir, args.output_dir, args.batch_size)
