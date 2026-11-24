#!/usr/bin/env python3
"""
Baseline Evaluator for Production Champion E22-01 (XLS-R 300M + Char CTC)
Evaluates E22-01 under matched frozen evaluation conditions across:
- dev_a_disjoint (6,257 utts)
- dev_b_waaz_val (127 utts)
- test_manifest.jsonl (5,923 utts)
Saves predictions to results/comparisons/e22_01_predictions/
"""

import argparse
import json
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
from pathlib import Path

import editdistance
import soundfile as sf
import torch
import torchaudio.functional as AF
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Add project root to python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.normalization.normalizer import normalize_for_wer_evaluation
from experiments.stage29_omnilingual_kashmiri_asr.scripts.evaluate import compute_wer_cer_metrics


def evaluate_e22_manifest(
    model,
    processor,
    manifest_path: str,
    output_pred_path: str,
    device: str = "cuda",
    batch_size: int = 8,
    max_samples: int = None,
):
    print(f"\n--- Evaluating E22-01 on {manifest_path} ---", flush=True)
    with open(manifest_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    if max_samples:
        records = records[:max_samples]

    print(f"Total samples: {len(records):,}", flush=True)
    model.eval()
    model.to(device=device, dtype=torch.bfloat16)

    # Sort records by duration for efficient batching
    records.sort(key=lambda x: x.get("duration", 0.0))

    predictions = []
    total_audio_time = 0.0
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats(device)

    for i in range(0, len(records), batch_size):
        batch_records = records[i:i + batch_size]
        waveforms = []
        lengths = []

        for r in batch_records:
            wav, sr = sf.read(r["audio_path"], dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            t = torch.from_numpy(wav)
            if sr != 16000:
                t = AF.resample(t, orig_freq=sr, new_freq=16000)
            waveforms.append(t)
            lengths.append(len(t))
            total_audio_time += len(t) / 16000.0

        max_len = max(lengths)
        batch_tensor = torch.zeros(len(waveforms), max_len, dtype=torch.bfloat16, device=device)
        attention_mask = torch.zeros(len(waveforms), max_len, dtype=torch.long, device=device)

        for b_idx, w in enumerate(waveforms):
            batch_tensor[b_idx, :len(w)] = w.to(device=device, dtype=torch.bfloat16)
            attention_mask[b_idx, :len(w)] = 1

        with torch.inference_mode():
            outputs = model(batch_tensor, attention_mask=attention_mask)
            logits = outputs.logits
            pred_ids = torch.argmax(logits, dim=-1)
            hyps = processor.batch_decode(pred_ids)

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

        del batch_tensor, attention_mask, outputs, logits, pred_ids, waveforms

        if (i // batch_size) % 50 == 0:
            torch.cuda.empty_cache()
            print(f"  Processed {min(i + batch_size, len(records))}/{len(records)} utterances...", flush=True)

    total_compute_time = time.time() - t0
    rtf = total_compute_time / total_audio_time if total_audio_time > 0 else 0.0
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    refs = [p["reference_norm"] for p in predictions]
    hyps = [p["hypothesis_norm"] for p in predictions]
    metrics = compute_wer_cer_metrics(refs, hyps)
    metrics["rtf"] = rtf
    metrics["peak_gpu_memory_gb"] = peak_mem_gb
    metrics["total_audio_hours"] = total_audio_time / 3600.0
    metrics["total_compute_seconds"] = total_compute_time

    print(f"\n--- Results for E22-01 on {Path(manifest_path).stem} ---")
    print(f"  WER: {metrics['wer']:.2f}% | CER: {metrics['cer']:.2f}%")
    print(f"  Substitutions: {metrics['substitutions']:,} | Deletions: {metrics['deletions']:,} | Insertions: {metrics['insertions']:,}")
    print(f"  Sentence Accuracy: {metrics['sentence_accuracy']:.2f}%")
    print(f"  RTF: {rtf:.4f}x (Compute: {total_compute_time:.1f}s, Audio: {total_audio_time:.1f}s)")
    print(f"  Peak GPU Memory: {peak_mem_gb:.2f} GB")

    os.makedirs(os.path.dirname(output_pred_path), exist_ok=True)
    with open(output_pred_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved predictions to {output_pred_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="kashmiri_dataset/experiments/stage22_data_expanded_training/checkpoints/E22-01/best_model")
    parser.add_argument("--output-dir", type=str, default="experiments/stage29_omnilingual_kashmiri_asr/results/comparisons/e22_01")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading E22-01 Champion from {args.checkpoint}...")
    processor = Wav2Vec2Processor.from_pretrained(args.checkpoint)
    model = Wav2Vec2ForCTC.from_pretrained(args.checkpoint)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    partitions = {
        "dev_a_disjoint": "experiments/stage29_omnilingual_kashmiri_asr/data/validation_manifest_dev_a_disjoint.jsonl",
        "dev_b_waaz_val": "experiments/stage29_omnilingual_kashmiri_asr/data/validation_manifest_dev_b_waaz.jsonl",
        "test": "experiments/stage29_omnilingual_kashmiri_asr/data/test_manifest.jsonl",
    }

    summary = {}
    for part_name, mpath in partitions.items():
        pred_path = out_dir / f"e22_01_{part_name}_predictions.jsonl"
        metrics = evaluate_e22_manifest(
            model=model,
            processor=processor,
            manifest_path=mpath,
            output_pred_path=str(pred_path),
            device="cuda",
            batch_size=16,
            max_samples=args.max_samples,
        )
        summary[part_name] = metrics

    with open(out_dir / "e22_01_summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll E22-01 baseline benchmarks recorded to {out_dir / 'e22_01_summary_metrics.json'}")


if __name__ == "__main__":
    main()
