# Code Walkthrough: E22-01 Configuration (`configs/experiments/e22_01.yaml`)

---

## 1. Official E22-01 Configuration Specification

```yaml
# Certified E22-01 Production Configuration
model:
  name: "xlsr_300m_kashmiri_e22_01"
  base_encoder: "facebook/wav2vec2-xls-r-300m"
  vocab_size: 47
  final_dropout: 0.1
  blank_bias: 3.0

data:
  train_manifest: "manifests/train_106k.jsonl"
  val_manifest: "manifests/val_disjoint.jsonl"
  sample_rate: 16000
  max_duration: 20.0
  min_duration: 0.8

training:
  epochs: 20
  batch_size: 8
  gradient_accumulation_steps: 8
  learning_rate: 3.0e-4
  llrd_decay: 0.75
  warmup_ratio: 0.10
  fp16: true
  max_grad_norm: 1.0

decoding:
  temperature: 1.08
  alpha: 0.50
  beta: 1.50
  beam_width: 100
  kenlm_path: "models/lm/kashmiri_5gram.binary"
```
