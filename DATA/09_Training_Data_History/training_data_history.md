# Training Data History & Model-Wise Exposure Audit

## 1. Executive Summary

Over the course of the project, five major model training generations have been conducted. Across these experiments, training volume grew from an initial 11.08 hours in B-01 to ~180 hours in Stage 22.

- **Total Physical Human Speech in Possession**: **542.28 Hours**
- **Maximum Historical Training Volume (Stage 22)**: **178.60 Hours**
- **Immediately Training-Ready Unused Gold Data**: **2.86 Hours** (new Omarrran Conversational, clean Xkaab, verified WAAZ community audio)
- **Total Human Speech Never Used in Any Model**: **363.68 Hours** (including Silver and Bronze tiers)

## 2. Model-Wise Training Data Matrix

| Model ID | Model Architecture | Manifest Used | Clips Used | Training Hours | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **B-01** | Conformer CTC (Baseline v1) | `kashmiri_dataset/manifests/train.csv` | 7,307 | 10.4h | Initial 4-source baseline (OpenSLR, Programindz, Xkaab, KTTS) |
| **B-02A** | Conformer CTC + KenLM 3-gram | `kashmiri_dataset/manifests/train.csv` | 7,307 | 10.4h | Same acoustic split as B-01 with language model rescoring |
| **Stage 12 (R-01/R-02)** | Robust Conformer (Acoustic Augmentation) | `kashmiri_dataset/experiments/stage12_robustness_training/manifests/stage12_train_approved_96k.jsonl` | 93,126 | 151.37h | Expanded with Bolbosh and augmented acoustic variations |
| **Stage 22 (E-22)** | Conformer Large + CTC Beam Search | `asr_training_eligible_manifest.jsonl` | 106,487 | 214.65h | Full expanded training manifest (106k clips) |
| **Stage 29 (O29)** | Meta Omnilingual / MMS Kashmiri Fine-Tuning | `experiments/stage29_omnilingual_kashmiri_asr/configs/train_manifest.jsonl` | 0 | 0.0h | Multilingual foundation model adapted to Kashmiri |
