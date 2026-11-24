#!/usr/bin/env python3
"""
Full Evaluation Suite for Stage 29 Kashmiri ASR Models
Evaluates a checkpoint on:
- dev_a_disjoint (6,257 utts)
- dev_b_waaz_val (127 utts)
- test (5,923 utts)
Produces detailed JSON metrics, per-utterance predictions, and summary tables.
"""

import argparse
import json
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import sentencepiece as spm
import torch
from experiments.stage29_omnilingual_kashmiri_asr.scripts.model import (
    Wav2Vec2AsrModel,
    OmniModelConfig,
    load_omni_checkpoint,
)
from experiments.stage29_omnilingual_kashmiri_asr.scripts.evaluate import evaluate_manifest


def run_full_suite(
    model_type: str,
    checkpoint_path: str,
    exp_name: str,
    tokenizer_path: str = "experiments/stage29_omnilingual_kashmiri_asr/assets/omniASR_tokenizer_written_v2.model",
    batch_size: int = 32,
    max_samples: int = None,
):
    print(f"\n==================================================================")
    print(f"=== FULL EVALUATION SUITE FOR {exp_name.upper()} ({model_type.upper()}) ===")
    print(f"==================================================================")

    out_dir = Path(f"experiments/stage29_omnilingual_kashmiri_asr/results/comparisons/{exp_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SentencePiece tokenizer from {tokenizer_path}...")
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)

    config = OmniModelConfig.get_300m_config() if model_type == "300m" else OmniModelConfig.get_1b_config()
    model = Wav2Vec2AsrModel(config)
    load_omni_checkpoint(model, checkpoint_path, device="cpu")
    model.to(dtype=torch.bfloat16, device="cuda")
    model.eval()

    partitions = {
        "dev_a_disjoint": "experiments/stage29_omnilingual_kashmiri_asr/data/validation_manifest_dev_a_disjoint.jsonl",
        "dev_b_waaz_val": "experiments/stage29_omnilingual_kashmiri_asr/data/validation_manifest_dev_b_waaz.jsonl",
        "test": "experiments/stage29_omnilingual_kashmiri_asr/data/test_manifest.jsonl",
    }

    suite_metrics = {
        "experiment_name": exp_name,
        "model_type": model_type,
        "checkpoint": checkpoint_path,
        "timestamp": time.time(),
        "partitions": {},
    }

    for part_name, manifest_file in partitions.items():
        out_preds = out_dir / f"{exp_name}_{part_name}_predictions.jsonl"
        print(f"\nEvaluating {part_name} ({manifest_file})...")
        metrics = evaluate_manifest(
            model=model,
            tokenizer=sp,
            manifest_path=manifest_file,
            output_pred_path=str(out_preds),
            device="cuda",
            batch_size=batch_size,
            max_samples=max_samples,
        )
        suite_metrics["partitions"][part_name] = metrics

    # Save summary JSON
    summary_file = out_dir / f"{exp_name}_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(suite_metrics, f, indent=2)
    print(f"\nSaved suite summary to {summary_file}")

    print("\n======================================================================")
    print(f"BENCHMARK SUMMARY: {exp_name.upper()}")
    print("======================================================================")
    print(f"{'Partition':<18} | {'WER (%)':<10} | {'CER (%)':<10} | {'Sent Acc':<10} | {'RTF':<8}")
    print("-------------------+------------+------------+------------+---------")
    for part, m in suite_metrics["partitions"].items():
        w = m.get('norm_wer', m.get('wer', 0.0))
        c = m.get('norm_cer', m.get('cer', 0.0))
        s = m.get('sent_accuracy', m.get('sentence_accuracy', 0.0))
        r = m.get('rtf', 0.0)
        print(f"{part:<18} | {w:<10.2f} | {c:<10.2f} | {s:<10.2f} | {r:<8.4f}")
    print("======================================================================\n")
    return suite_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["300m", "1b"], default="300m")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    run_full_suite(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint,
        exp_name=args.exp_name,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
