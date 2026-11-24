# Stage 29: Meta Omnilingual ASR CTC 1B v2 for Kashmiri Speech Recognition

**EduCasheer Kashmiri AI — Advanced Acoustic Modeling Research Initiative**  
**Lead Research Team**: Principal ASR Research Scientist, Deep Learning Architect, ML Systems Engineer, Reproducibility Auditor  
**Hardware Environment**: NVIDIA DGX Spark (NVIDIA GB10 GPU, 128GB Unified Memory, ARM64)  
**Date Initiated**: November 17, 2026  

---

## 1. Mission Overview & Scientific Hypotheses

The Kashmiri speech recognition initiative previously established **E22-01** (XLS-R 300M + Character CTC, trained on 166.39h) as the certified production champion with:
- **24.94% WER** on `dev_a_disjoint` (KenLM calibrated).
- **29.06% WER** on the Stage 22B sealed held-out test set (5,923 utterances, 120,816 words).
- **46.12% WER** on spontaneous, conversational WAAZ validation (`dev_b_waaz_val`).

### Research Objective:
Investigate whether Meta's second-generation multilingual speech foundation models (**Omnilingual ASR v2**) can substantially advance Kashmiri recognition accuracy toward parity with high-resource ASR benchmarks.

### Model Hierarchy:
- **Primary Candidate**: `omniASR_CTC_1B_v2` (1B parameter wav2vec2 acoustic encoder + 10,288-token written CTC head).
- **Secondary Control**: `omniASR_CTC_300M_v2` (300M parameter wav2vec2 acoustic encoder + 10,288-token written CTC head).
- **Production Baseline**: EduCasheer `E22-01` (XLS-R 300M + 159-token Character CTC).

### Core Hypotheses:
1. **$H_1$ (Pretraining Superiority)**: Omnilingual v2's 1,600+ language multilingual pretraining provides superior phonetic representations for Kashmiri central vowels and retroflex consonants compared to XLS-R 53/128.
2. **$H_2$ (Capacity Scaling)**: Scaling from 300M to 1B acoustic parameters will substantially reduce word error rates, particularly on ambiguous acoustic boundaries and complex Perso-Arabic morphology.
3. **$H_3$ (Data vs. Architecture)**: Controlled comparison between Dataset A (matched 166.39h) and Dataset B (expanded 207.24h) isolates whether gains originate from model scale or speech supervision.

---

## 2. Directory Structure

```text
stage29_omnilingual_kashmiri_asr/
├── README.md                           # This document: mission overview and status
├── RESEARCH_PLAN.md                    # Formal pre-registered scientific protocol
├── ENVIRONMENT.md                     # Hardware, OS, and software environment audit
├── configs/
│   ├── omni_ctc_1b_v2.yaml            # Standard 1B v2 fine-tuning configuration
│   ├── omni_ctc_300m_v2.yaml          # Standard 300M v2 control configuration
│   ├── pilot_1b.yaml                  # Rapid preflight overfitting & pilot config
│   ├── full_training_1b.yaml          # Full production training specification
│   └── evaluation.yaml                # Frozen multi-set evaluation protocol
├── data/
│   ├── train_manifest_dataset_a.jsonl # Matched historical training data (166.39h)
│   ├── train_manifest_dataset_b.jsonl # Expanded verified Kashmiri corpus (207.24h)
│   ├── validation_manifest_dev_a_disjoint.jsonl # Primary dev benchmark (11.30h)
│   ├── validation_manifest_dev_b_waaz.jsonl     # Conversational WAAZ benchmark (0.20h)
│   ├── test_manifest.jsonl            # Sealed independent test set (13.04h)
│   ├── exclusion_manifest.jsonl       # 788 misaligned meta_omnilingual samples
│   ├── data_audit.json                # Cryptographic manifest hashes and metrics
│   └── tokenizer_audit.json           # SentencePiece vocabulary coverage audit
├── scripts/
│   ├── prepare_data.py                # Manifest generation and data audit
│   ├── verify_tokenizer.py            # SentencePiece Kashmiri compatibility test
│   ├── evaluate_zero_shot.py          # Zero-shot baseline evaluation
│   ├── profile_training.py            # GB10 memory and throughput profiling
│   ├── train.py                       # Unified CTC training and checkpointing loop
│   ├── evaluate.py                    # Multi-partition evaluation script
│   ├── compare_models.py              # Statistical significance & comparison table
│   └── verify_checkpoints.py          # Model weight and reproducibility auditor
├── checkpoints/
│   ├── omni_300m_v2/                  # 300M base and fine-tuned checkpoints
│   └── omni_1b_v2/                    # 1B base and fine-tuned checkpoints
├── results/
│   ├── zero_shot/                     # Pretrained baseline metrics and hypotheses
│   ├── pilot/                         # One-batch and pilot validation results
│   ├── full_training/                 # Production fine-tuning metrics
│   └── comparisons/                   # Head-to-head comparisons against E22-01
└── reports/
    ├── DATA_AUDIT.md                  # Comprehensive dataset provenance audit
    ├── TOKENIZER_COMPATIBILITY.md     # Orthography and SentencePiece coverage audit
    ├── HARDWARE_FEASIBILITY.md        # NVIDIA GB10 unified memory feasibility study
    ├── TRAINING_REPORT.md             # Empirical training curves and loss convergence
    ├── EVALUATION_REPORT.md           # Benchmark evaluation across all partitions
    └── FINAL_RESEARCH_REPORT.md       # Final scientific verdict and findings
```
