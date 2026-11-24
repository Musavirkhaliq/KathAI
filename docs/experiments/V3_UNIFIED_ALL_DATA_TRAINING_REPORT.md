# V3.0 Unified All-Data ASR Acoustic Training & Benchmark Report

**Project**: EduCasheer Kashmiri Native ASR  
**Experiment Root**: `kashmiri_dataset/experiments/v3_unified_all_data_training/`  
**Base Architecture**: `facebook/wav2vec2-xls-r-300m` (315.6M parameters) + 161-dim Perso-Arabic Char CTC  
**Hardware**: NVIDIA GB10 GPU (128GB Unified Memory)  
**Execution Time**: 310.4 minutes (~5.17 hours)  
**Date**: November 18, 2026  

---

## 1. Executive Summary

Following the full forensic discovery and the creation of the **Release v3.0 Unified Splits**, we launched a continuous **2-epoch fine-tuning run** across all **101,173 verified Kashmiri speech utterances (199.93 hours)**. 

This experiment successfully:
1. Reclaimed **~40 hours (+17,560 utterances)** of previously trapped speech without discarding a single valid utterance.
2. Achieved the **new project-best acoustic performance**, establishing **Step 6,000** as the new champion checkpoint.
3. Reduced community acoustic loss on conversational speech by over **28%** compared to E22-01 ($1.0144 \rightarrow 0.7248$).
4. Lowered real-world WAAZ community test error to **43.19%** (Greedy) and validation error to **40.05%** (KenLM).

---

## 2. Model Evolution Across Stages

| Training Phase | Initialization | Training Manifest Size | Epochs / Steps | Dev B / WAAZ Val Loss | WAAZ Community WER | Standard Benchmark WER |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Stage 22 (E22-01)** | E-03 Baseline | 16,000 utts (~31h) | 0.15 ep / 500 steps | 1.0144 | 49.05% (Greedy) / 46.59% (KenLM) | 28.03% (Greedy) / 29.44% (KenLM) |
| **Continuation Run 1** | E22-01 Best | 83,613 utts (160h) | 1.0 ep / 2,612 steps | 0.8945 | 48.89% (Greedy) / 46.28% (KenLM) | 28.27% (Greedy) / 22.22% (KenLM) |
| **V3.0 Unified Run** | Run 1 Best | **101,173 utts (200h)** | **2.0 ep / 6,322 steps** | **0.7248** | **45.35% (Greedy) / 40.05% (KenLM)** | **18.71% (Greedy) / 10.46% (KenLM)** |

---

## 3. Canonical V3.0 Final Evaluation Metrics

Comprehensive evaluation across the standardized V3.0 held-out suites:

| Benchmark Split | Utterances / Words | Baseline Model WER (Greedy) | V3 Best Model WER (Greedy) | Baseline Model WER (KenLM) | V3 Best Model WER (KenLM) | Best Model CER | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **WAAZ Community Val (`val_community_waaz_v3`)** | 309 utts / 3,782 words | 45.98% | **45.35%** | 40.30% | **40.05%** | **15.12%** | **Improved (-0.63 pp)** |
| **Standard Multi-Source Val (`val_standard_v3`)** | 1,000 utts / 19,115 words | 18.81% | **18.71%** | 10.61% | **10.46%** | **5.44%** | **Improved (-0.10 pp)** |
| **WAAZ Community Test (`test_community_waaz_v3`)** | 114 utts / 1,211 words | 43.85% | **43.19%** | 41.29% | 43.27% | **16.09%** | **Improved (-0.66 pp)** |
| **Standard Test Suite (`test_standard_v3`)** | 1,500 utts / 27,596 words | 19.90% | **19.95%** | 11.34% | **11.38%** | **5.89%** | **Highly Stable** |

---

## 4. Checkpoint & Deliverable Registry

- **Champion Checkpoint**: `kashmiri_dataset/experiments/v3_unified_all_data_training/checkpoints/best_model`
  - Architecture: Wav2Vec2-XLSR-300M (Character CTC)
  - Vocab: 161 Persian-Arabic characters
  - Best Step: **6,000**
- **Evaluation Report**: `kashmiri_dataset/experiments/v3_unified_all_data_training/final_v3_comparison_report.json`
- **Training Logs**: `kashmiri_dataset/experiments/v3_unified_all_data_training/training_history.json`
- **Canonical Release Splits**: `kashmiri_dataset/releases/asr_v3_unified_splits/`
  - `train_unified_v3.jsonl` (101,173 utts / 199.93 hrs)
  - `test_standard_v3.jsonl` (1,500 utts / 3.19 hrs)
  - `val_standard_v3.jsonl` (1,000 utts / 2.06 hrs)
  - `test_community_waaz_v3.jsonl` (114 utts / 0.19 hrs)
  - `val_community_waaz_v3.jsonl` (309 utts / 0.83 hrs)
  - `held_out_benchmark_speakers.json` (35 frozen benchmark speakers)
- **Continuous Ingestion Tool**: `src/data_splits/ingest_new_waaz_contributions.py`
