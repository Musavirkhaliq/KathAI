#!/usr/bin/env python3
"""
Workstream 9: External Kashmiri ASR Model Discovery and Inspection
"""

import json
from pathlib import Path
import huggingface_hub
from huggingface_hub import HfApi

api = HfApi()

models_to_check = [
    "bodhan-ai/indic-transcribe-core",
    "ai4bharat/indic-conformer-600m-multilingual",
    "udgamlabs/shruti-kashmiri",
    "ai4bharat/indicconformer_stt_ks_hybrid_ctc_rnnt_large",
    "Muzaffar786/wav2vec2-kashmiri-jhon-data-one",
    "Muzaffar786/wav2vec2-kashmiri-full-data",
    "Aazu34/kashmiri-asr-ctc",
    "khushi1234455687/wav2vec2-large-xlsr-53-Kashmiri-Version6",
    "helpsulaiman/kashmiri-wav2vec2",
    "muneebharoon/whisper-kashmiri"
]

print("=== CHECKING TARGET REPOSITORIES ===")
results = {}
for m in models_to_check:
    try:
        info = api.model_info(m)
        gated = getattr(info, "gated", False)
        private = getattr(info, "private", False)
        tags = getattr(info, "tags", [])
        siblings = [s.rfilename for s in info.siblings] if info.siblings else []
        print(f"FOUND: {m} | Gated: {gated} | Private: {private} | Files: {len(siblings)}")
        key_files = [f for f in siblings if any(f.endswith(ext) for ext in [".pt", ".bin", ".safetensors", ".json", ".onnx", ".cact", ".nemo", ".yaml", ".model"])]
        print(f"  Key files ({len(key_files)}): {key_files[:10]}")
        results[m] = {
            "status": "found",
            "gated": gated,
            "private": private,
            "files_count": len(siblings),
            "key_files": key_files,
            "tags": tags
        }
    except Exception as e:
        print(f"ERROR: {m} -> {e}")
        results[m] = {
            "status": "error",
            "error": str(e)
        }

print("\n=== SEARCHING FOR ALL KASHMIRI SPEECH/ASR MODELS ON HF ===")
try:
    search_res = list(api.list_models(search="kashmiri"))
    asr_models = []
    for r in search_res:
        t_list = getattr(r, "tags", []) or []
        pipeline = getattr(r, "pipeline_tag", None)
        if pipeline in ["automatic-speech-recognition", "audio-to-text"] or any(t in t_list for t in ["asr", "speech", "audio"]):
            dl = getattr(r, "downloads", 0)
            asr_models.append((r.id, dl, pipeline, t_list[:5]))
    
    print(f"Total discovered ASR models with 'kashmiri': {len(asr_models)}")
    for mid, dl, pipe, tags in asr_models:
        print(f"  -> {mid} | pipeline: {pipe} | downloads: {dl}")
except Exception as e:
    print(f"Search error: {e}")

out_path = Path("kashmiri_dataset/experiments/external_asr_benchmark/model_discovery_raw.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved raw discovery data to {out_path}")
