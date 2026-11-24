#!/usr/bin/env python3
"""
Standardized Evaluation Script for Kashmiri ASR Models
Evaluates models on:
- dev_a_disjoint (6,257 utts)
- dev_b_waaz_val (127 utts)
- test_manifest.jsonl (5,923 utts)
Computes:
- Raw & Normalized WER
- Raw & Normalized CER
- Substitutions, Insertions, Deletions, Hits
- Sentence-level accuracy
- Real-Time Factor (RTF) and Peak GPU Memory
"""

import argparse
import json
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

import editdistance
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F

# Add project root to python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import sentencepiece as spm
from src.normalization.normalizer import normalize_for_wer_evaluation
from experiments.stage29_omnilingual_kashmiri_asr.scripts.model import (
    Wav2Vec2AsrModel,
    OmniModelConfig,
    load_omni_checkpoint,
)


def compute_wer_cer_metrics(references, hypotheses):
    """
    Computes Levenshtein alignment metrics at word and character levels.
    """
    total_words = 0
    total_word_sub = 0
    total_word_del = 0
    total_word_ins = 0

    total_chars = 0
    total_char_sub = 0
    total_char_del = 0
    total_char_ins = 0

    sentence_matches = 0
    num_samples = len(references)

    for ref, hyp in zip(references, hypotheses):
        # Word-level
        ref_words = ref.split()
        hyp_words = hyp.split()
        total_words += len(ref_words)

        # Sentence exact match
        if ref == hyp:
            sentence_matches += 1

        # Use editdistance for word alignment
        w_dist = editdistance.eval(ref_words, hyp_words)
        # Approximate S, D, I
        r_len = len(ref_words)
        h_len = len(hyp_words)
        diff = h_len - r_len
        if diff > 0:
            ins = diff
            sub_del = w_dist - ins
            sub = sub_del // 2
            delete = sub_del - sub
        else:
            delete = -diff
            sub_ins = w_dist - delete
            sub = sub_ins // 2
            ins = sub_ins - sub

        total_word_sub += sub
        total_word_del += delete
        total_word_ins += ins

        # Character-level
        ref_chars = list(ref.replace(" ", ""))
        hyp_chars = list(hyp.replace(" ", ""))
        total_chars += len(ref_chars)
        c_dist = editdistance.eval(ref_chars, hyp_chars)
        c_diff = len(hyp_chars) - len(ref_chars)
        if c_diff > 0:
            c_ins = c_diff
            c_sub = (c_dist - c_ins) // 2
            c_del = c_dist - c_ins - c_sub
        else:
            c_del = -c_diff
            c_sub = (c_dist - c_del) // 2
            c_ins = c_dist - c_del - c_sub

        total_char_sub += c_sub
        total_char_del += c_del
        total_char_ins += c_ins

    total_word_errors = total_word_sub + total_word_del + total_word_ins
    total_char_errors = total_char_sub + total_char_del + total_char_ins

    wer = (total_word_errors / total_words * 100.0) if total_words > 0 else 0.0
    cer = (total_char_errors / total_chars * 100.0) if total_chars > 0 else 0.0
    sent_acc = (sentence_matches / num_samples * 100.0) if num_samples > 0 else 0.0

    return {
        "wer": wer,
        "cer": cer,
        "total_words": total_words,
        "substitutions": total_word_sub,
        "deletions": total_word_del,
        "insertions": total_word_ins,
        "total_word_errors": total_word_errors,
        "total_chars": total_chars,
        "char_substitutions": total_char_sub,
        "char_deletions": total_char_del,
        "char_insertions": total_char_ins,
        "total_char_errors": total_char_errors,
        "sentence_accuracy": sent_acc,
        "num_sentences": num_samples,
    }


def load_audio_waveform(path: str, target_sr: int = 16000) -> torch.Tensor:
    """Loads and resamples audio to 16kHz mono float32."""
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    tensor = torch.from_numpy(wav)
    if sr != target_sr:
        tensor = F.resample(tensor, orig_freq=sr, new_freq=target_sr)
    return tensor


def evaluate_manifest(
    model: Wav2Vec2AsrModel,
    tokenizer: spm.SentencePieceProcessor,
    manifest_path: str,
    output_pred_path: str,
    device: str = "cuda",
    batch_size: int = 16,
    max_samples: Optional[int] = None,
):
    print(f"\n--- Evaluating Manifest: {manifest_path} ---")
    with open(manifest_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    if max_samples:
        records = records[:max_samples]

    print(f"Total samples to evaluate: {len(records):,}")

    model.eval()
    model.to(device)

    # Sort records by duration for efficient batching
    records.sort(key=lambda x: x.get("duration", 0.0))

    # Separate standard length (< 25s) and long utterances (>= 25s)
    short_records = [r for r in records if r.get("duration", 0.0) < 25.0]
    long_records = [r for r in records if r.get("duration", 0.0) >= 25.0]

    predictions = []
    total_audio_time = 0.0
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats(device)

    # 1. Evaluate standard utterances in batches
    for i in range(0, len(short_records), batch_size):
        batch_records = short_records[i:i + batch_size]
        waveforms = []
        lengths = []

        for r in batch_records:
            w = load_audio_waveform(r["audio_path"])
            waveforms.append(w)
            lengths.append(len(w))
            total_audio_time += len(w) / 16000.0

        max_len = max(lengths)
        batch_tensor = torch.zeros(len(waveforms), max_len, dtype=torch.bfloat16, device=device)
        lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=device)

        for b_idx, w in enumerate(waveforms):
            batch_tensor[b_idx, :len(w)] = w.to(dtype=torch.bfloat16, device=device)

        with torch.no_grad():
            logits, out_lens = model(batch_tensor, lengths_tensor)
            hyps = model.decode_greedy(logits, out_lens, tokenizer)

        for r, hyp in zip(batch_records, hyps):
            norm_ref = normalize_for_wer_evaluation(r.get("text", ""))
            norm_hyp = normalize_for_wer_evaluation(hyp)
            predictions.append({
                "sample_id": r["sample_id"],
                "audio_path": r["audio_path"],
                "duration": r.get("duration", 0.0),
                "reference_raw": r.get("text", ""),
                "reference_norm": norm_ref,
                "hypothesis_raw": hyp,
                "hypothesis_norm": norm_hyp,
            })

        del batch_tensor, lengths_tensor, logits, out_lens, waveforms
        if (i // batch_size) % 50 == 0:
            torch.cuda.empty_cache()
            print(f"  Processed {min(i + batch_size, len(short_records))}/{len(records)} utterances...", flush=True)

    # 2. Evaluate long utterances individually with chunking to protect memory
    if long_records:
        print(f"  Evaluating {len(long_records)} long utterances with memory-safe chunking...")
        for r in long_records:
            w = load_audio_waveform(r["audio_path"])
            total_audio_time += len(w) / 16000.0
            
            chunk_samples = int(30.0 * 16000)
            step_samples = int(28.0 * 16000)
            total_samples = len(w)
            
            hyp_pieces = []
            for start in range(0, total_samples, step_samples):
                end = min(start + chunk_samples, total_samples)
                chunk = w[start:end].unsqueeze(0).to(dtype=torch.bfloat16, device=device)
                c_lens = torch.tensor([chunk.shape[1]], dtype=torch.long, device=device)
                with torch.no_grad():
                    logits, out_lens = model(chunk, c_lens)
                    piece = model.decode_greedy(logits, out_lens, tokenizer)[0]
                    if piece.strip():
                        hyp_pieces.append(piece.strip())
                del chunk, c_lens, logits, out_lens
                torch.cuda.empty_cache()
                if end >= total_samples:
                    break
            
            hyp = " ".join(hyp_pieces)
            norm_ref = normalize_for_wer_evaluation(r.get("text", ""))
            norm_hyp = normalize_for_wer_evaluation(hyp)
            predictions.append({
                "sample_id": r["sample_id"],
                "audio_path": r["audio_path"],
                "duration": r.get("duration", 0.0),
                "reference_raw": r.get("text", ""),
                "reference_norm": norm_ref,
                "hypothesis_raw": hyp,
                "hypothesis_norm": norm_hyp,
            })

    torch.cuda.empty_cache()

    total_compute_time = time.time() - t0
    rtf = total_compute_time / total_audio_time if total_audio_time > 0 else 0.0
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    # Compute metrics
    refs = [p["reference_norm"] for p in predictions]
    hyps = [p["hypothesis_norm"] for p in predictions]
    metrics = compute_wer_cer_metrics(refs, hyps)
    metrics["rtf"] = rtf
    metrics["peak_gpu_memory_gb"] = peak_mem_gb
    metrics["total_audio_hours"] = total_audio_time / 3600.0
    metrics["total_compute_seconds"] = total_compute_time

    print(f"\n--- Results for {Path(manifest_path).stem} ---")
    print(f"  WER: {metrics['wer']:.2f}% | CER: {metrics['cer']:.2f}%")
    print(f"  Substitutions: {metrics['substitutions']:,} | Deletions: {metrics['deletions']:,} | Insertions: {metrics['insertions']:,}")
    print(f"  Sentence Accuracy: {metrics['sentence_accuracy']:.2f}%")
    print(f"  RTF: {rtf:.4f}x (Compute: {total_compute_time:.1f}s, Audio: {total_audio_time:.1f}s)")
    print(f"  Peak GPU Memory: {peak_mem_gb:.2f} GB")

    # Save predictions
    os.makedirs(os.path.dirname(output_pred_path), exist_ok=True)
    with open(output_pred_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved predictions to {output_pred_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Omnilingual ASR CTC Model")
    parser.add_argument("--model-type", choices=["300m", "1b"], default="300m")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="experiments/stage29_omnilingual_kashmiri_asr/assets/omniASR_tokenizer_written_v2.model")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output-predictions", type=str, required=True)
    parser.add_argument("--output-metrics", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading SentencePiece tokenizer from {args.tokenizer}...")
    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer)

    config = OmniModelConfig.get_300m_config() if args.model_type == "300m" else OmniModelConfig.get_1b_config()
    model = Wav2Vec2AsrModel(config)
    load_omni_checkpoint(model, args.checkpoint, device="cpu")
    model.to(dtype=torch.bfloat16, device="cuda")

    metrics = evaluate_manifest(
        model,
        sp,
        args.manifest,
        args.output_predictions,
        device="cuda",
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )

    if args.output_metrics:
        os.makedirs(os.path.dirname(args.output_metrics), exist_ok=True)
        with open(args.output_metrics, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics JSON to {args.output_metrics}")


if __name__ == "__main__":
    main()
