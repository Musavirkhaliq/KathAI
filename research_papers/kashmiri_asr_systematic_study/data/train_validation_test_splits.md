# Training, Validation, and Test Split Definitions

This document details the exact partitioning methodology, sample counts, speaker constraints, and hours allocated to all development and evaluation splits in the Kashmiri ASR project.

---

## 1. Master Acoustic Modeling Partitions

| Partition Identifier | File Path | Utterance Count | Audio Duration | Unique Speakers | Primary Acoustic Sources | Split Constraint |
|:---|:---|:---:|:---:|:---:|:---|:---|
| **B-01 Train** | `.../native_asr_training/experiments/B-01.../train.jsonl` | 53,460 | 92.94 hrs | 340+ | Bolbosh, OpenSLR, XKaab, IndicVoices | Speaker-disjoint from val/test |
| **B-02A Train** | `.../stage9_b02_acoustic_comparison/B-02A/train.jsonl` | 64,522 | 104.93 hrs | 410+ | B-01 + recovered speech | Speaker-disjoint from val/test |
| **B-02B Train** | `.../stage9_b02_acoustic_comparison/B-02B/train.jsonl` | 80,106 | 123.45 hrs | 415+ | B-02A + Mirwaiz Quran | Speaker-disjoint from val/test |
| **B-02C Train** | `.../stage9_b02_acoustic_comparison/B-02C/train.jsonl` | 96,664 | 165.32 hrs | 580+ | B-02B + IndicVoices-R | Speaker-disjoint from val/test |
| **Tier A Train** | `kashmiri_dataset/manifests/training/stage17b_tier_a_train.jsonl` | 63,078 | 114.20 hrs | 420+ | High-SNR verified speech ($\ge 20$ dB) | Speaker-disjoint from val/test |
| **Tier B Train** | `kashmiri_dataset/manifests/training/stage17b_tier_b_train.jsonl` | 31,617 | 51.12 hrs | 160+ | Conversational speech ($12-20$ dB) | Speaker-disjoint from val/test |
| **Validation (Dev Canonical)** | `kashmiri_dataset/experiments/native_asr_training/data_splits/val.jsonl` | 6,728 | 13.80 hrs | 112 | Bolbosh (2,830), IndicVoices (3,164), XKaab (582), OpenSLR (152) | **0.00% speaker overlap with train** |
| **Diagnostic Assessment** | `.../stage11_robustness_optimization/data_splits/dev_assessment.jsonl` | 1,001 | 1.95 hrs | 48 | Native 16k (570), Bolbosh 22k (431) | Frozen diagnostic manifest |
| **Diagnostic Calibration** | `.../stage11_robustness_optimization/data_splits/dev_calibration.jsonl` | 1,000 | 1.92 hrs | 45 | Native 16k (583), Bolbosh 22k (417) | Disjoint from dev_assessment |
| **Benchmark v1.1 Primary** | `.../benchmark_construction/releases/v1_1_corrected/manifests/benchmark_v1_1_primary.jsonl` | 6,315 | 12.40 hrs | 95 | Tier A, B, C balanced | Quarantined benchmark |

---

## 2. Validation Split Composition Breakdown (`val.jsonl`)

The 6,728-sample canonical validation split represents a balanced multi-domain evaluation pool:

| Sub-Corpus Source | Utterance Count | Reference Words | Mean Duration (s) | Domain / Genre |
|:---|:---:|:---:|:---:|:---|
| **AI4Bharat IndicVoices** | 3,164 | 49,210 | 7.42 | Spontaneous dialogue & field interviews |
| **GAASH-Lab Bolbosh** | 2,830 | 44,120 | 6.85 | Read sentences & conversational passages |
| **XKaab ASR** | 582 | 8,915 | 6.12 | Regional dialectal speech |
| **OpenSLR SLR122** | 152 | 2,440 | 6.98 | Clean studio read speech |
| **Total** | **6,728** | **104,685** | **7.08** | **Comprehensive Multi-Domain Kashmiri** |

---

## 3. Diagnostic Split Composition Breakdown (`dev_assessment.jsonl`)

The 1,001-sample diagnostic assessment split contains exactly 16,845 reference words:

| Cohort | Utterance Count | Reference Words | Native Sample Rate | Resampling Requirement |
|:---|:---:|:---:|:---:|:---|
| **Native 16 kHz Cohort** | 570 | 8,775 | 16,000 Hz | Direct bypass (0% distortion) |
| *— X-Kaab Studio* | 86 | 1,252 | 16,000 Hz | Direct bypass |
| *— OpenSLR SLR122* | 22 | 275 | 16,000 Hz | Direct bypass |
| *— Field / Conversational 16k* | 462 | 7,248 | 16,000 Hz | Direct bypass |
| **Bolbosh 22.05 kHz Cohort** | 431 | 8,070 | 22,050 Hz | **Mandatory Kaiser windowed sinc to 16 kHz** |
| **Total** | **1,001** | **16,845** | **Mixed** | **Canonical Evaluation Protocol v2** |
