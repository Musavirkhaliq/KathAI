# EduCasheer Kashmiri Automatic Speech Recognition (ASR) Research Project

[![Acoustic Model: XLS-R 300M](https://img.shields.io/badge/Model-wav2vec2--xls--r--300m-blue.svg)](https://huggingface.co/facebook/wav2vec2-xls-r-300m)
[![Compute: NVIDIA DGX Spark](https://img.shields.io/badge/Compute-NVIDIA%20DGX%20Spark%20GB10-green.svg)](#)
[![Dataset: 214.65 Hours](https://img.shields.io/badge/Speech%20Corpus-214.65%20Hours%20%7C%20106k%20Utterances-orange.svg)](#)
[![SOTA Champion: E22-01](https://img.shields.io/badge/SOTA%20Model-Stage%2022%20E22--01-gold.svg)](#)
[![Calib WER: 24.74%](https://img.shields.io/badge/Calib%20KenLM%20WER-24.74%25-brightgreen.svg)](#)
[![Disjoint Holdout WER: 24.94%](https://img.shields.io/badge/Holdout%20KenLM%20WER-24.94%25-success.svg)](#)

EduCasheer Kashmiri ASR is an advanced, production-grade deep learning research framework dedicated to speech recognition, linguistic normalization, acoustic modeling, language model shallow fusion, and dual romanization for the Kashmiri language (**کٲشُر** / *Koshur*, ISO 639-3: `kas`).

---

## 🧭 Master Project Documentation Hub

| Master Document | Description & Research Role |
| :--- | :--- |
| **[`PROJECT_MASTER_GUIDE.md`](PROJECT_MASTER_GUIDE.md)** | **Definitive 20-Section Master Guide**: Architecture, data lineage, stage evolution, metrics, and onboarding. |
| **[`REPOSITORY_MAP.md`](REPOSITORY_MAP.md)** | Complete structural map of all 243k+ files, directories, and stage artifacts. |
| **[`docs/architecture/SYSTEM_ARCHITECTURE.md`](docs/architecture/SYSTEM_ARCHITECTURE.md)** | End-to-end technical diagrams, receptive field math, and decoding derivations. |
| **[`docs/codebase/CODE_READING_ORDER.md`](docs/codebase/CODE_READING_ORDER.md)** | Recommended 8-step dependency-aware learning sequence for new researchers. |
| **[`docs/codebase/FILE_BY_FILE_AUDIT.md`](docs/codebase/FILE_BY_FILE_AUDIT.md)** | Exhaustive line-by-line AST audit of all 416 first-party code files. |
| **[`docs/research/STAGE_REGISTRY.md`](docs/research/STAGE_REGISTRY.md)** | Chronological forensic registry of all 24 research stages (B-01 through Stage 24). |
| **[`docs/research/EXPERIMENT_RESULTS.md`](docs/research/EXPERIMENT_RESULTS.md)** | Canonical reconciled leaderboard across validation, diagnostic, and disjoint sets. |
| **[`docs/research/MODEL_REGISTRY.md`](docs/research/MODEL_REGISTRY.md)** | Checkpoint paths, model architectures, and SHA-256 cryptographic provenance. |
| **[`docs/research/FAILED_EXPERIMENTS.md`](docs/research/FAILED_EXPERIMENTS.md)** | Exhaustive post-mortems of 7 documented negative results and dead ends. |
| **[`docs/data/DATASET_REGISTRY.md`](docs/data/DATASET_REGISTRY.md)** | Authoritative index of all 12 speech sources (214.65h / 106k samples) and 5 text corpora. |
| **[`docs/data/DATASET_LINEAGE.md`](docs/data/DATASET_LINEAGE.md)** | End-to-end data pipeline from raw Google Drive shards to 16kHz zero-speaker overlap splits. |
| **[`docs/reproducibility/REPRODUCTION_GUIDE.md`](docs/reproducibility/REPRODUCTION_GUIDE.md)** | Step-by-step verified execution commands for smoke tests, benchmarks, and audits. |

---

## 🏆 Current State-of-the-Art (SOTA) Model: Stage 22 Champion `E22-01`

The official project champion and production-certified acoustic model is **`E22-01`**:

- **Model Identifier**: `Stage 22 E22-01` (Certified Production Champion)
- **Architecture**: `facebook/wav2vec2-xls-r-300m` (315,603,745 parameters) + Character CTC (159 tokens).
- **Training Corpus**: **166.39 hours** (97,143 clean utterances) incorporating audited community conversational speech from WAAZ.
- **Checkpoint Location**:
  [`kashmiri_dataset/experiments/stage22_data_expanded_training/checkpoints/E22-01/best_model`](kashmiri_dataset/experiments/stage22_data_expanded_training/checkpoints/E22-01/best_model)
- **Cryptographic Checksum**: SHA-256 `0c18d60b985966da6f97db20f79211ce669deb6db19861bdedded1569f5560a2`
- **Certified Empirical Accuracy**:
  - **`dev_calibration.jsonl` (1,001 utts, 16,632 words)**: **24.74% KenLM WER** (4,114 errors)
  - **`dev_a_disjoint.jsonl` (6,257 utts, 106,530 words)**: **24.94% KenLM WER** (26,566 errors)
  - **`dev_b_waaz_val.jsonl` (127 utts, 1,262 words)**: **46.12% KenLM WER** (Held-out spontaneous community speech)
  - **Real-Time Factor (RTF)**: **0.005** on NVIDIA DGX Spark GB10 GPU.

---

## ⚡ Quickstart: 30-Second Verification

Transcribe a sample audio file or verify GPU runtime:
```bash
# 1. Fast smoke test on validation sample
python3 test_batch_transcribe.py

# 2. CLI transcription of any arbitrary audio file
python3 transcribe_kashmiri.py path/to/recording.wav --normalize

# 3. Launch automated repository audit suite
python3 scripts/audit/run_audit_suite.py
```

---

## 🔬 Key Scientific Findings

1. **Acoustic Scaling Dominates**: Expanding training data from 92.9h to 165.3h provided the single largest performance gain in project history (-6.02% absolute WER, $p < 0.0001$).
2. **Forensic Audio Resampling Breakthrough (Stage 20R/20S)**: Discovered that 431 Bolbosh files at 22,050 Hz were passed into 16 kHz feature extractors unresampled in Stages 11–19, generating 493 phantom word errors. Kaiser windowed sinc resampling eliminated all 493 errors on identical weights and proved that foundation B-02C and sequence-continuation E-03 differ by only 1 word error ($p = 0.963$).
3. **Exact Receptive Field Slicing (Stage 22C)**: Proved CNN receptive field recurrence ($R=400$ samples, stride 320), eliminating 100% of trailing token hallucinations in pyctcdecode.
4. **Resolution of Neural Candidate Reranking (Stage 24)**: Proved that a 329k-parameter neural ranker trained over 20,000 utterances yielded zero gain over 4-gram KenLM shallow fusion ($p = 0.738$). Officially halted candidate reranking.

---

## 📄 License & Research Citation

This research repository is developed under educational and research open-access licenses. Individual datasets retain their respective upstream licenses (e.g., CC-BY 4.0 for Bolbosh and IndicVoices).
