# Gated & External Kashmiri Datasets: Comprehensive Audit Report

**Date**: November 8, 2026  
**Auditor**: Principal Speech & NLP Research Engineer  
**Hugging Face Account**: `Musavir123`  
**Token Authorization Status**: Validated (`Bearer hf_FRUimccd...`)  

---

## 1. Executive Summary

A comprehensive, file-by-file audit was executed against all known Kashmiri repositories on Hugging Face to verify:
1. Whether repositories requiring gated approval have been approved by their authors.
2. What datasets were already downloaded locally vs. truly missing.
3. Strict enforcement: **Never re-download existing data or overwrite verified files.**

---

## 2. Gated Datasets Approval Audit

The Hugging Face REST API differentiates between **public metadata access** (README.md) and **authenticated raw data file access** (requiring author sign-off). Each repository was directly tested against raw payload URLs:

| Dataset Name | Hugging Face Repository | Gate Type | HTTP Status | Actual Permission Status | Action Required |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **600K-KS-OCR** | `Omarrran/600k_KS_OCR_Word_Segmented_Dataset` | Manual | 403 Forbidden | **Pending Author Approval** | Awaiting review by `@Omarrran` |
| **Koshur Pixel** | `Omarrran/Koshur_Pixel` | Manual | 403 Forbidden | **Restricted / Pending** | Awaiting review by `@Omarrran` |
| **KS-PRET-5M** | `Omarrran/KS-PRET-5M_..._12M_tokens_2026` | Manual | 403 Forbidden | **Pending Author Approval** | Awaiting review by `@Omarrran` |
| **KS-LIT-3M** | `Omarrran/3.1Million_KASHMIRI_text_..._by_HNM` | Manual | 403 Forbidden | **Pending Author Approval** | Awaiting review by `@Omarrran` |
| **SMUQamar 270K**| `SMUQamar/Kashmiri-English-Dataset-270K` | Manual | 403 Forbidden | **Pending Author Approval** | Awaiting review by `@SMUQamar` |
| **Faizaniqbal Audio** | `Faizaniqbal/kashmiri-audio-corpus` | Manual | 403 Forbidden | **Restricted / Pending** | Awaiting review by `@Faizaniqbal` |

> [!IMPORTANT]
> Although user agreements have been submitted on Hugging Face, **the repository authors have not yet clicked "Approve" on their dashboards**. Until manual author approval is granted, Hugging Face returns `403 Forbidden: Your request to access dataset is awaiting a review from the repo authors`.

---

## 3. Strict Deduplication Audit & New Data Downloaded

To comply strictly with the instruction: **"MAKE SURE NOT TO DOWNLOAD WHAT WAS DOWNLOADED ALREADY OR IS THERE"**, we scanned the entire filesystem (`Data/`, `kashmiri_dataset/raw/`, `kashmiri_dataset/external/`, `kashmiri_dataset/processed/`).

### 3.1 Datasets Verified as Already Present Locally (100% Skipped)
- **`yashwardhan20417/indicvoices-r-kashmiri-codemixing`** (33.05 GB) — *Preserved*
- **`yashwardhan20417/rasa-kashmiri-codemixing-merged`** (11.89 GB) — *Preserved*
- **`humair025/KashmiriSpeech-IndicVoices`** (4.28 GB extracted WAVs) — *Preserved*
- **`muneebharoon/whisper-kashmiri`** (3.07 GB) — *Preserved*
- **`Omarrran/kashmiri_single_speaker_ttd_hf_dataset_2024`** (2.84 GB) — *Preserved*
- **`programindz/kashmiri-tts-single-speaker`** (1.53 GB) — *Preserved*
- **`humairmunirawn/xp3x-kashmiri`** (1.18 GB) — *Preserved*
- **`Omarrran/kashmiri_words_dataset_restructured`** (834.98 MB) — *Preserved*
- **`Omarrran/kashmiri_sample_text_rocognition`** (672.91 MB) — *Preserved*
- **`Omarrran/kashmiri_tts_hf_dataset`** (645.56 MB) — *Preserved*
- **`XKaab/ASR-Kashmiri_4hrs`** (476.49 MB extracted WAVs) — *Preserved*
- **`yashwardhan20417/codemixed-kashmiri`** (417.58 MB) — *Preserved*
- **`aazu34/kashmiri-openslr122-segments`** (234.86 MB extracted WAVs) — *Preserved*
- **`krish187653/rasa-kashmiri-codemixing-transliterations`** (133.99 MB) — *Preserved*
- **`Omarrran/Kashmiri_one_word_text_reconition_dataset`** (18.89 MB) — *Preserved*
- **`nawabhussain/Kashmiri-Language-Corpus`** (3.60 MB) — *Preserved*
- **`yashwardhan20417/rasa-kashmiri-text-only`** (1.31 MB) — *Preserved*
- **`injilashah/Kashmiri-language-text-dataset`** (1.16 MB) — *Preserved*
- **`injilashah/kashmiri_text_and_audio`** (10 KB) — *Preserved*

### 3.2 Genuinely Missing Open Datasets Downloaded
The following open Kashmiri datasets had never been downloaded to the machine. They were cleanly downloaded into isolated directories without touching existing files:

1. **`programindz/kashmiri-audio-corpus`** (Speech / Audio)  
   - Downloaded to: `kashmiri_dataset/external/programindz_kashmiri-audio-corpus/`  
   - Size: **332.50 MB** (`train-00000-of-00001.parquet`)  
   - Contains audio and transcriptions for native Kashmiri speech.
2. **`Aadilgani/kashmiri-text-dataset`** (Text / NLP)  
   - Downloaded to: `kashmiri_dataset/external/aadilgani_kashmiri-text-dataset/`  
   - Size: **2.10 MB** (`kashmiri_dataset_v1.parquet`)  
   - Clean Kashmiri text corpus.
3. **`Omarrran/Sample_kashmiri_sft_dataset`** (Instruction / SFT)  
   - Downloaded to: `kashmiri_dataset/external/omarrran_sample_kashmiri_sft_dataset/`  
   - Size: **0.03 MB** (`kashmiri_sft_dataset.jsonl`)  
   - Supervised instruction tuning dataset in Perso-Arabic Kashmiri.

---

## 4. Summary & Verification

- **Total disk space free**: **2.5 TB** (no storage risk).
- **Existing datasets**: Zero duplicates downloaded; 100% of existing raw and processed files remain intact.
- **Action for Gated Repositories**: The access requests are logged with authors `Omarrran` and `SMUQamar`. As soon as their authors approve your Hugging Face account (`Musavir123`), the automated downloader `kashmiri_dataset/scripts/download_all_unlocked_datasets.py` will pull the remaining data in a single run.
