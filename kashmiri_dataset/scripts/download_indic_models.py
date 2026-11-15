#!/usr/bin/env python3
"""
Workstream 9: Targeted Download Engine for IndicConformer-600M and Indic-Transcribe-core
Downloads the verified Kashmiri components without redundant non-Kashmiri language weights.
"""

import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

BASE_MODELS_DIR = Path("kashmiri_dataset/experiments/external_asr_benchmark/models")
BASE_MODELS_DIR.mkdir(parents=True, exist_ok=True)

def download_indic_conformer_600m():
    repo_id = "ai4bharat/indic-conformer-600m-multilingual"
    dest = BASE_MODELS_DIR / "ai4bharat_indic-conformer-600m-multilingual"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}\nDownloading {repo_id} (Kashmiri ONNX assets)...\n{'='*60}")
    
    needed_files = [
        "model_onnx.py",
        "config.json",
        "README.md",
        "assets/encoder.onnx",
        "assets/ctc_decoder.onnx",
        "assets/rnnt_decoder.onnx",
        "assets/joint_enc.onnx",
        "assets/joint_pred.onnx",
        "assets/joint_pre_net.onnx",
        "assets/joint_post_net_ks.onnx",
        "assets/preprocessor.ts",
        "assets/vocab.json",
        "assets/language_masks.json"
    ]
    
    downloaded = {}
    for f in needed_files:
        print(f"Fetching {f}...")
        p = hf_hub_download(repo_id=repo_id, filename=f, local_dir=str(dest))
        downloaded[f] = p
        
    print(f"Successfully downloaded {len(downloaded)} assets to {dest}")
    return dest

def download_indic_transcribe_core():
    repo_id = "bodhan-ai/indic-transcribe-core"
    dest = BASE_MODELS_DIR / "bodhan-ai_indic-transcribe-core"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}\nDownloading {repo_id} (Canary FastConformer 1.2B)...\n{'='*60}")
    
    needed_files = [
        "model.safetensors",
        "feature_extractor.safetensors",
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer_multilingual.model",
        "tokenizer_spl_tokens.model",
        "indic_transcribe.py",
        "feature_extraction_indic_canary.py",
        "modeling_indic_canary.py",
        "tokenization_indic_canary.py",
        "lid.py",
        "README.md"
    ]
    
    downloaded = {}
    for f in needed_files:
        print(f"Fetching {f}...")
        p = hf_hub_download(repo_id=repo_id, filename=f, local_dir=str(dest))
        downloaded[f] = p
        
    print(f"Successfully downloaded {len(downloaded)} assets to {dest}")
    return dest

if __name__ == "__main__":
    download_indic_conformer_600m()
    download_indic_transcribe_core()
