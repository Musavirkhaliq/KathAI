#!/usr/bin/env python3
"""
Kashmiri Speech Recognition CLI & Inference Utility (Production Entry Point)
===========================================================================
Model: fine-tuned facebook/wav2vec2-xls-r-300m + Kashmiri Character CTC
Priority: Champion E22-01 (Stage 22) -> B-02C -> B-02A -> B-01
"""
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.asr.model import DEFAULT_MODEL_DIR, CHECKPOINT_REGISTRY
from src.asr.transcribe import KashmiriASR
from src.normalization.normalizer import normalize_kashmiri_text as normalize_text


def main():
    parser = argparse.ArgumentParser(description="Transcribe Kashmiri speech audio into text.")
    parser.add_argument("audio", type=str, help="Path to audio file (WAV, MP3, FLAC, etc.)")
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR), help="Path to model directory")
    parser.add_argument("--normalize", action="store_true", help="Apply text normalization")
    parser.add_argument("--use-kenlm", action="store_true", help="Enable KenLM beam search decoding")
    args = parser.parse_args()

    asr = KashmiriASR(model_dir=args.model_dir)
    text = asr.transcribe(args.audio, normalize=args.normalize, use_kenlm=args.use_kenlm)
    print("\n" + "="*50)
    print(f"File:       {args.audio}")
    print(f"Transcript: {text}")
    print("="*50)


if __name__ == "__main__":
    main()
