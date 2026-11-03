# Kashmiri Dataset: Comprehensive Research-Grade Audit Report

**Date**: 2026-11-01  
**Auditor**: Automated Data Engineering Pipeline  
**Scope**: Full audit of 22 archive shards and external expansions.

---

## 1. Physical Architecture & Archive Ingestion

- **Total Multi-part Shards**: 22 directory partitions (`Kashmiri Project-20261101T102755Z-1-001` through `...022`).
- **Ingestion Nature**: Extracted multi-part Google Drive zip download preserving a single unified root directory hierarchy: `Kashmiri Project/`.
- **Cross-shard Duplication**: **0 duplicated relative paths**. Every file is uniquely distributed across the shards.
- **Immutability Status**: Raw directory `/home/musa/Kashmiri Project/Data` is strictly preserved in read-only mode.

## 2. Modality & Category Breakdown

| Category | Directory | Files | Raw Size | Modality | Primary Formats | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **01_Speech_Audio** | `01_OpenSLR_SLR122` | 94 | 846.9 MB | Speech | WAV, TXT | Downloaded & Validated |
| **01_Speech_Audio** | `02_Kashmiri_Audio_Segmented` | 2 | 317.1 MB | Speech | Parquet (embedded audio) | Downloaded & Validated |
| **01_Speech_Audio** | `03_KTTS` | 2,987 | 489.5 MB | Speech | WebM/Opus (labeled .wav), CSV | Validated (Codec Mismatch Resolved) |
| **01_Speech_Audio** | `04_LDCIL_Raw_Speech` | 1 | 0.0 MB | Metadata | ACCESS_STATUS.txt | Formal Access Pending |
| **02_Literature_Text** | `01_KS_LIT_3M` & `02_KS_PRET_5M` | 2 | 0.0 MB | Metadata | Status.txt | Gated Access Pending |
| **03_OCR** | `01_600K_KS_OCR` & `02_Koshur_Pixel` | 2 | 0.0 MB | Metadata | Status.txt | Gated Access Pending |
| **04_Documentation** | Documentation files | 5 | 0.1 MB | Docs | PDF, XLSX, TXT | Complete |
| **05_OCR_Based_Candidates** | Historical Books | 9 | 242.0 MB | Scanned Text | PDF (4 historic volumes) | Digitized Scans |
| **06_Directly_Parsable** | `Parallel Corpus` | 3 | 6.1 MB | Parallel Text | TXT, JSON | 30,000 sentence pairs |
| **06_Directly_Parsable** | `Corpus Raw Excel` | 1 | 3.4 MB | Monolingual Text | XLSX | 57,489 rows |
| **06_Directly_Parsable** | `Koshur_Pixel` | 739 | 41.8 GB | Synthetic OCR | Parquet, HuggingFace cache | 613,078 image-text pairs |

## 3. Speech Data In-Depth Technical Audit

### 3.1 OpenSLR SLR122 Kashmiri
- **Files**: 43 continuous audio WAV recordings + 43 transcript TXT files.
- **Total Raw Duration**: 12,160.13 seconds = **3.38 hours**.
- **Transcript Structure**: Timestamp-aligned utterance records (`start_time \t end_time \t kashmiri_text`).
- **Total Utterance Segments**: **1,983 sentence segments**.
- **Total Active Speech Duration**: **2.54 hours**.
- **Audio Specs**: 16,000 Hz, 32,000 Hz, 44,100 Hz; Mono and Stereo PCM.
- **Speakers**: Multiple native Kashmiri speakers (Abdul, Bilal Ahmad Sheikh, Gh-Nabi, Qazi Muneeb, Abid Rehman, Burhan Aijaz, etc.).
- **Dialect / Regional Coverage**: Spontaneous conversational Kashmiri representing Srinagar, Shopian, Pulwama, and Budgam.
- **Code-Switching**: Frequent colloquial English loanwords (MBA, semester, placement, company, job, degree, Bangalore, etc.) captured verbatim in Perso-Arabic script.

### 3.2 Kashmiri Audio Corpus (`programindz`)
- **Format**: Hugging Face Parquet archive (`train-00000-of-00001.parquet`) with embedded binary audio.
- **Total Records**: 1,955 segments.
- **Total Duration**: 9,107.54 seconds = **2.53 hours**.
- **Sampling Rate**: 16,000 Hz mono.
- **Missing Fields**: 0 missing audio, 0 missing text, 0 missing duration.
- **Duplicate Texts**: 20 duplicate text sentences detected and tracked.

### 3.3 KTTS Single Speaker Dataset (Mendeley Data)
- **Files**: 2,984 audio files + 2 CSV transcript files.
- **Crucial Technical Finding**: Extension/Container mismatch. The files are named `.wav`, but container probing reveals they are **Matroska/WebM containers carrying Opus-encoded audio** (48 kHz mono).
- **Resolution**: FFMPEG-based pipeline converts each WebM stream into clean 16 kHz 16-bit mono PCM WAV.
- **ID Alignment**: 100% bijective match. Exactly 2,984 audio files match 2,984 transcript rows with 0 missing IDs in either direction.
- **Speaker**: 1 adult male speaker, native standard Srinagar dialect.
- **Acoustics**: Studio-recorded, high SNR, low reverberation. Ideal for ASR acoustic anchor and TTS synthesis.

## 4. Text & Literature Technical Audit

### 4.1 Kashmiri-English Parallel Corpus
- **Size**: 30,000 aligned sentence pairs.
- **Quality**: High-quality English to Kashmiri translations spanning cultural, historical, and narrative domains.
- **Scripts**: Perso-Arabic Kashmiri (Nastaliq orthography) and Latin English.

### 4.2 Raw Language Corpus Excel (`Kashmiri_Language_Corpus_Raw.xlsx`)
- **Total Rows**: 57,489 rows (52,645 non-null Kashmiri sentences).
- **Parsing Issue Discovered**: Commas/tabs in unquoted text caused column overflow into `Unnamed: 3` through `Unnamed: 20`.
- **Resolution**: Recombined into single text strings with whitespace normalization.

### 4.3 Koshur Pixel Synthetic OCR Corpus
- **Records**: 613,078 image-text pairs across 364 Parquet files (41.8 GB).
- **Text Core**: Contains the full text stream of KS-PRET-5M rendered in Kashmiri Nastaliq/Naksh fonts.
- **Utility**: Serves as both an OCR training benchmark and a large-scale textual corpus for Kashmiri language modeling.

## 5. Script & Orthographic Harmonization Findings

Kashmiri Perso-Arabic orthography exhibits significant digital inconsistency:
1. **Kaf Variations**: Arabic Kaf (U+0643) vs Urdu/Kashmiri Keheh (U+06A9). Standardized to U+06A9.
2. **Heh Variations**: Arabic Heh (U+0647) vs Heh Goal (U+06C1) vs Teh Marbuta (U+0629). Standardized to U+06C1.
3. **Yeh Variations**: Arabic Yeh (U+064A) vs Farsi Yeh (U+06CC) vs Kashmiri wavy/tail Yeh (U+06D0/U+06D2). Harmonized to standard Kashmiri orthography.
4. **Diacritics**: Kashmiri requires diacritics (Zer, Zabar, Pesh, Ulta Pesh, Khari Zabar) to disambiguate central vowels. Preserved for acoustic training, with an optional stripped mode for standard ASR WER/CER evaluation.
