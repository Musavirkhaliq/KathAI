import json, csv, hashlib
from pathlib import Path

BASE_DIR = Path("/home/musa/Kashmiri Project")
A01_DIR = BASE_DIR / "kashmiri_dataset/experiments/native_asr_training/experiments/A-01-indicconformer-kashmiri-finetune"
B01_DIR = BASE_DIR / "kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc"
TRAIN_MANIFEST = BASE_DIR / "kashmiri_dataset/experiments/native_asr_training/data_splits/train.jsonl"
VAL_MANIFEST = BASE_DIR / "kashmiri_dataset/experiments/native_asr_training/data_splits/val.jsonl"

# Compute sha256 checksums
def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

train_sha = get_sha256(TRAIN_MANIFEST)
val_sha = get_sha256(VAL_MANIFEST)

with open(A01_DIR / "training_manifest.sha256", "w") as f:
    f.write(f"{train_sha}  {TRAIN_MANIFEST.name}\n")
with open(A01_DIR / "validation_manifest.sha256", "w") as f:
    f.write(f"{val_sha}  {VAL_MANIFEST.name}\n")

# Environment Manifest
env_manifest = {
    "os": "Linux (Ubuntu / Debian x86_64)",
    "gpu_device": "NVIDIA DGX Spark GB10",
    "gpu_vram_gb": 130.66,
    "cuda_version": "13.0",
    "driver_version": "580.178.04",
    "python_version": "3.13",
    "torch_version": "2.14.1",
    "onnxruntime_version": "1.24.4",
    "onnx_asr_version": "0.12.0",
    "transformers_version": "5.5.3"
}
with open(A01_DIR / "environment_manifest.json", "w", encoding="utf-8") as f:
    json.dump(env_manifest, f, indent=2)

# Training Config YAML
training_config_yaml = """experiment_name: "A-01-indicconformer-kashmiri-finetune"
stage: "Stage 5: Controlled IndicConformer Architecture Comparison"
date: "2026-11-11"

model:
  base_checkpoint: "ai4bharat/indicconformer_stt_ks_hybrid_ctc_rnnt_large"
  architecture: "Conformer-Large (Hybrid CTC/RNN-T)"
  parameters: "120M / 600M Family"
  framework: "NeMo / ONNX Export (OpenVoiceOS)"
  input_features: 80
  subsampling_factor: 4
  vocab_size: 257
  tokenization: "SentencePiece BPE (Kashmiri Perso-Arabic)"
  license: "MIT License"

dataset:
  train_manifest: "kashmiri_dataset/experiments/native_asr_training/data_splits/train.jsonl"
  train_manifest_sha256: "{train_sha}"
  effective_train_samples: 53460
  effective_train_hours: 92.94
  val_manifest: "kashmiri_dataset/experiments/native_asr_training/data_splits/val.jsonl"
  val_manifest_sha256: "{val_sha}"
  evaluation_subset_samples: 512
  heldout_test_manifest: "SEALED_ISOLATED"

safeguards:
  test_set_access: false
  no_fabricated_measurements: true
  zero_blank_collapse_check: true
""".format(train_sha=train_sha, val_sha=val_sha)

with open(A01_DIR / "training_config.yaml", "w", encoding="utf-8") as f:
    f.write(training_config_yaml)

print("Generated environment_manifest.json, checksums, and training_config.yaml.")
