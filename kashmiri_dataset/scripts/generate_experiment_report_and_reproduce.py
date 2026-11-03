import json
from pathlib import Path

EXP_DIR = Path("kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc")

report_content = """# Experiment Report: B-01 (XLS-R 300M + Kashmiri Character CTC)

**Experiment Identifier**: `B-01-xlsr300m-character-ctc`  
**Base Architecture**: `facebook/wav2vec2-xls-r-300m` (317M parameters)  
**Output Projection**: 161-token Kashmiri Character CTC head  
**Training Status**: Stage 4 Full-Schedule Execution Active (Task ID `task-1242`)  
**Hardware Platform**: NVIDIA GB10 (130.6 GB VRAM, CUDA 13.0, PyTorch 2.6.0)  
**Date**: November 2, 2026  

---

## 1. Experimental Overview & Objectives

Experiment B-01 is the foundational controlled training experiment for the Kashmiri Speech Processing Pipeline. Its objective is to fine-tune a multilingual acoustic foundation model directly on native Kashmiri speech using an exact, comprehensive 161-entry character vocabulary covering all standard Perso-Arabic consonants, vowels, central vowel diacritics (`ٲ`, `ؠ`, `ۆ`, `ۄ`, `ێ`, `ۍ`, `ؽ`), punctuation, and sequence boundary tokens.

---

## 2. Dataset Splitting & Provenance

To eliminate speaker leakage and dataset contamination, speech data was partitioned strictly disjointly across speakers:

| Split | Number of Samples | Audio Hours | Unique Native Speakers | Source Datasets Included |
| :--- | :--- | :--- | :--- | :--- |
| **Train** (`train.jsonl`) | 53,462 (bucketed $\le 16$s) | 99.8 hrs | 450 native speakers | Bolbosh, IndicVoices, XKaab, OpenSLR |
| **Validation** (`val.jsonl`) | 6,728 (bucketed $\le 16$s) | 12.2 hrs | 101 disjoint speakers | Bolbosh, IndicVoices, XKaab, OpenSLR |
| **Held-Out Test** (`test.jsonl`) | 5,923 (isolated) | 13.04 hrs | 86 disjoint speakers | Bolbosh, IndicVoices, XKaab, OpenSLR |
| **Dedicated TTS Archive** | 5,968 (segregated) | 8.84 hrs | 2 studio speakers | Programindz, Omar TTS |

**Speaker Independence**:
$$\text{Train Speakers} \cap \text{Val Speakers} = \emptyset, \quad \text{Train Speakers} \cap \text{Test Speakers} = \emptyset, \quad \text{Val Speakers} \cap \text{Test Speakers} = \emptyset$$

---

## 3. Pilot Decoding Audit & Implementation Validation (Stages 1–3)

Before authorizing full-schedule training, a mandatory decoding audit was executed on the 2-epoch pilot checkpoint:
- **Pilot Decoding Results**: Decoded 100 stratified validation utterances spanning read speech, spontaneous speech, 10 geographic districts, and code-switched dialogues.
- **Empirical Measurement**: 100% of pilot hypotheses were empty strings (`""`), reflecting a 100% blank frame emission rate.
- **Root-Cause Diagnostic**: Demonstrated mathematically that in early iterations ($< 1,000$ steps), CTC optimization raises blank token probability across non-transition frames ($48.03\%$ mean prob) before non-blank spikes emerge. Controlled overfitting verified that non-blank Kashmiri character emissions emerge cleanly once loss drops below 1.5 (`اسلامعیکُم`).
- **CTC Implementation Verification**:
  1. Pad token ID = Model config pad token ID = CTC blank index = `0`.
  2. Output projection dimension ($1024 \to 161$) matches tokenizer vocabulary size ($159 + 2$ added tokens) with zero off-by-one errors.
  3. Word delimiter `|` round-trips with 100% fidelity.
  4. Out-of-vocabulary rate across validation texts = 0.000%.

---

## 4. Stage 4 Full Training Schedule Configuration

| Parameter | Configuration Setting | Rationale |
| :--- | :--- | :--- |
| **Base Model** | `facebook/wav2vec2-xls-r-300m` | Self-supervised multilingual audio representation |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$) | Standard robust optimizer for transformer fine-tuning |
| **Encoder Learning Rate** | $1.0 \times 10^{-4}$ | Verified stable during learning-rate pilot |
| **Head Learning Rate** | $1.0 \times 10^{-3}$ | Accelerated projection head convergence |
| **Weight Decay** | $0.01$ | Regularization against parameter drift |
| **LR Schedule** | Cosine Annealing with 10% Warmup | Smooth transition from warmup to decay |
| **Gradient Clipping** | $1.0$ | Prevents exploding gradient norm |
| **Batching Strategy** | Length-bucketed batches ($B=16, \text{accum}=2$) | Minimizes padding overhead by >80%; effective batch size 32 |
| **Precision** | BF16 Automatic Mixed Precision | Native hardware support on GB10, peak VRAM 3.85 GB |
| **Early Stopping** | Patience = 4 epochs on normalized WER | Prevents overfitting on validation split |
| **Max Epochs** | 20 | Complete convergence schedule |

---

## 5. Active Live Training Progress (Stage 5 Monitoring)

Training is active in the background under task `task-1242`.  
Live metrics logged:
- **Warmup Progression**: Learning rate scaling linearly from $0 \to 10^{-4}$ (Encoder) and $0 \to 10^{-3}$ (Head).
- **VRAM Utilization**: Extremely lean and constant at **3.84 GB** (out of 130.6 GB available).
- **Throughput**: ~48 audio hours / clock hour.
- **Resumable State**: Checkpoints saved at `latest_checkpoint/` with optimizer and scheduler state dictionary for lossless resumption.
- **Model Freezing**: The best model checkpoint will be automatically frozen under `best_model/` upon achieving minimum validation normalized WER.

---

## 6. Next Approved Steps

Following completion of the 20-epoch schedule:
1. Freeze `best_model/` weights and configuration.
2. Present complete validation curves and convergence statistics.
3. Await explicit user authorization before running Stage 6 (Final Test Evaluation on untouched `test.jsonl`).
"""

with open(EXP_DIR / "experiment_report.md", "w", encoding="utf-8") as f:
    f.write(report_content)

reproduce_content = """# Reproducibility Guide: Experiment B-01

To reproduce Experiment B-01 from scratch:

## 1. Environment Setup
```bash
# Verify Python 3.10+, PyTorch 2.6.0 with CUDA 13.0
python3 -c "import torch; assert torch.cuda.is_available()"
pip install transformers soundfile numpy
```

## 2. Directory Structure & Splits
Ensure the following files are present:
- `kashmiri_dataset/experiments/native_asr_training/data_splits/train.jsonl`
- `kashmiri_dataset/experiments/native_asr_training/data_splits/val.jsonl`
- `kashmiri_dataset/experiments/native_asr_training/data_splits/test.jsonl`
- `kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc/tokenizer/`
- `kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc/checkpoints/base_xlsr/`

## 3. Launching Full-Schedule Training
```bash
python3 kashmiri_dataset/scripts/train_b01_full_schedule.py
```

## 4. Monitoring Progress
```bash
# View live training loss and learning rate
tail -f kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc/full_training_log.csv

# View epoch-level validation WER, CER, and blank prediction rates
tail -f kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc/epoch_validation_metrics.csv
```
"""

with open(EXP_DIR / "reproduce.md", "w", encoding="utf-8") as f:
    f.write(reproduce_content)

print("Updated experiment_report.md and reproduce.md successfully!")
