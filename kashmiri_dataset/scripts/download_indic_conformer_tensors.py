#!/usr/bin/env python3
"""
Download external ONNX tensor files for IndicConformer-600M
"""

from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

api = HfApi()
repo_id = "ai4bharat/indic-conformer-600m-multilingual"
dest = Path("kashmiri_dataset/experiments/external_asr_benchmark/models/ai4bharat_indic-conformer-600m-multilingual")
dest_assets = dest / "assets"
dest_assets.mkdir(parents=True, exist_ok=True)

info = api.model_info(repo_id, files_metadata=True)
onnx_tensor_files = [s.rfilename for s in info.siblings if s.rfilename.startswith("assets/onnx__")]

print(f"Total ONNX tensor files to download: {len(onnx_tensor_files)}")

downloaded = 0
for idx, f in enumerate(onnx_tensor_files):
    local_p = dest / f
    if not local_p.exists() or local_p.stat().st_size == 0:
        hf_hub_download(repo_id=repo_id, filename=f, local_dir=str(dest))
    downloaded += 1
    if downloaded % 25 == 0 or downloaded == len(onnx_tensor_files):
        print(f"[{downloaded}/{len(onnx_tensor_files)}] Downloaded...")

print("All ONNX tensor files downloaded successfully!")
