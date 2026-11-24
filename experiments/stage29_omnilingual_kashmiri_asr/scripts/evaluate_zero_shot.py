#!/usr/bin/env python3
"""
Zero-Shot Evaluation Engine for Meta Omnilingual ASR CTC Models
Evaluates pretrained models before fine-tuning on:
- dev_a_disjoint (speaker-disjoint read & conversational speech)
- dev_b_waaz_val (challenging, spontaneous conversational religious speech)
- test (held-out sealed independent test set)
Computes:
- Raw & Normalized WER / CER
- Substitutions, Insertions, Deletions
- Sentence Accuracy, Real-Time Factor (RTF), Peak GPU Memory
Produces:
- Zero-Shot Leaderboard and JSON metrics
"""

import argparse
import json
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
from pathlib import Path

# Add project root to python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import sentencepiece as spm
import torch
from src.normalization.normalizer import normalize_for_wer_evaluation
from experiments.stage29_omnilingual_kashmiri_asr.scripts.model import (
    Wav2Vec2AsrModel,
    OmniModelConfig,
    load_omni_checkpoint,
)
from experiments.stage29_omnilingual_kashmiri_asr.scripts.evaluate import evaluate_manifest


def run_zero_shot_suite(
    model_type: str,
    checkpoint_path: str,
    tokenizer_path: str,
    results_dir: str = "experiments/stage29_omnilingual_kashmiri_asr/results/zero_shot",
    max_samples: int = None,
):
    print(f"\n==================================================================")
    print(f"=== ESTABLISHING ZERO-SHOT BASELINES FOR omniASR_CTC_{model_type.upper()}_v2 ===")
    print(f"==================================================================")

    res_path = Path(results_dir)
    res_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading SentencePiece tokenizer from {tokenizer_path}...")
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)

    config = OmniModelConfig.get_300m_config() if model_type == "300m" else OmniModelConfig.get_1b_config()
    model = Wav2Vec2AsrModel(config)

    t_load_start = time.time()
    load_omni_checkpoint(model, checkpoint_path, device="cpu")
    model.to(dtype=torch.bfloat16, device="cuda")
    load_duration = time.time() - t_load_start
    print(f"Model loaded and transferred to CUDA in {load_duration:.2f} seconds.")

    partitions = {
        "dev_a_disjoint": "experiments/stage29_omnilingual_kashmiri_asr/data/validation_manifest_dev_a_disjoint.jsonl",
        "dev_b_waaz_val": "experiments/stage29_omnilingual_kashmiri_asr/data/validation_manifest_dev_b_waaz.jsonl",
        "test": "experiments/stage29_omnilingual_kashmiri_asr/data/test_manifest.jsonl",
    }

    all_results = {
        "model_type": model_type,
        "checkpoint": checkpoint_path,
        "load_time_seconds": load_duration,
        "partitions": {},
    }

    for part_name, manifest_file in partitions.items():
        out_preds = res_path / f"zero_shot_{model_type}_{part_name}_predictions.jsonl"
        metrics = evaluate_manifest(
            model=model,
            tokenizer=sp,
            manifest_path=manifest_file,
            output_pred_path=str(out_preds),
            device="cuda",
            batch_size=16 if model_type == "300m" else 8,
            max_samples=max_samples,
        )
        all_results["partitions"][part_name] = metrics

    # Save summary JSON
    summary_file = res_path / f"zero_shot_{model_type}_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved Zero-Shot summary to {summary_file}")

    # Print Zero-Shot Leaderboard Table
    print(f"\n{'='*70}")
    print(f"ZERO-SHOT LEADERBOARD: omniASR_CTC_{model_type.upper()}_v2")
    print(f"{'='*70}")
    print(f"{'Partition':<18} | {'WER (%)':<10} | {'CER (%)':<10} | {'Sent Acc':<10} | {'RTF':<8} | {'Peak GPU Mem':<12}")
    print(f"{'-'*18}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*12}")
    for p_name, m in all_results["partitions"].items():
        print(f"{p_name:<18} | {m['wer']:<10.2f} | {m['cer']:<10.2f} | {m['sentence_accuracy']:<10.2f} | {m['rtf']:<8.4f} | {m['peak_gpu_memory_gb']:<10.2f} GB")
    print(f"{'='*70}\n")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["300m", "1b"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="experiments/stage29_omnilingual_kashmiri_asr/assets/omniASR_tokenizer_written_v2.model")
    parser.add_argument("--results-dir", type=str, default="experiments/stage29_omnilingual_kashmiri_asr/results/zero_shot")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    run_zero_shot_suite(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        results_dir=args.results_dir,
        max_samples=args.max_samples,
    )
