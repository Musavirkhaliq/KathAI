# Kashmiri ASR Central Dataset Registry

**Project**: EduCasheer Kashmiri Automatic Speech Recognition Research  
**Hardware & Storage Environment**: High-Speed NVMe Storage (`/home/musa/Kashmiri Project`)  
**Last Updated**: November 14, 2026  

---

## 1. Master Speech Corpus Overview

The Kashmiri ASR research initiative has curated, normalized, standardized, and partitioned over **214.65 hours (106,487 speech utterances)** of native Kashmiri speech across studio, broadcast, conversational, read, and religious domains.

All speech audio is standardized into uniform 16kHz 16-bit Mono PCM WAV format with cryptographic SHA-256 integrity verification.

| Source Identifier | Dataset Name | Organization / Provider | Domain / Modality | Verified Samples | Verified Hours | License | Local Audio Directory |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **`SRC_EXT_BOLBOSH`** | Bolbosh Kashmiri Speech Corpus | GAASH-Lab (IUST Kashmir) | Multi-speaker read & conversation | 39,996 | 58.40h | CC-BY 4.0 | `kashmiri_dataset/processed/audio/bolbosh/` |
| **`SRC_EXT_INDICVOICES_R`** | IndicVoices-R Conversational Expansion | AI4Bharat / IIT Madras | Spontaneous native dialogue | 24,654 | 41.87h | CC-BY 4.0 | `kashmiri_dataset/processed/audio/indicvoices_r/` |
| **`SRC_EXT_INDICVOICES`** | IndicVoices Kashmiri Primary | AI4Bharat / IIT Madras | Conversational & field recordings | 20,015 | 29.80h | CC-BY 4.0 | `kashmiri_dataset/processed/audio/indicvoices_kashmiri/` |
| **`SRC_QURAN_MIRWAIZ`** | Al-Quran Kashmiri Translation (Mirwaiz) | Mirwaiz Research Foundation | Spoken religious translation | 11,906 | 20.91h | Educational / Research | `kashmiri_dataset/processed/audio/quran_mirwaiz/` |
| **`SRC_QURAN_DR_WAR`** | Al-Quran Kashmiri Translation (Dr. War) | Dr. Farooq Ahmad War Ingestion | Spoken religious commentary | 3,678 | 6.20h | Educational / Research | `kashmiri_dataset/processed/audio/quran_dr_war/` |
| **`SRC_RAW_KTTS`** | Kashmiri Text-to-Speech Corpus (KTTS) | Internal Kashmiri Archive | High-fidelity read speech | 2,984 | 4.54h | Research / Non-Commercial | `kashmiri_dataset/processed/audio/ktts/` |
| **`SRC_RAW_PROGRAMINDZ`** | Programindz Kashmiri Audio Corpus | Programindz | Read speech & phrase utterances | 1,955 | 2.45h | Open / Research | `kashmiri_dataset/processed/audio/programindz/` |
| **`SRC_QURAN_KING_FAHD`** | King Fahd Complex Kashmiri Translation | King Fahd Complex Ingestion | Classical spoken commentary | 1,920 | 3.10h | Public Domain / Research | `kashmiri_dataset/processed/audio/quran_king_fahd/` |
| **`SRC_RAW_OPENSLR122`** | OpenSLR SLR122 Kashmiri Speech | Google LLC / OpenSLR | Near-field studio read sentences | 1,508 | 2.21h | CC-BY-SA 4.0 | `kashmiri_dataset/processed/audio/openslr122/` |
| **`SRC_EXT_XKAAB_ASR`** | XKaab ASR Kashmiri 4hrs | XKaab / Kaab Lab | Dialectal regional speech | 2,733 | 3.92h | CC-BY 4.0 | `kashmiri_dataset/processed/audio/xkaab/` |
| **`SRC_EXT_WHISPER_KS`** | Whisper Kashmiri Alignment Collection | Muneeb Haroon / Community | Crowdsourced read utterances | 546 | 0.95h | Open / Research | `kashmiri_dataset/processed/audio/whisper_kashmiri/` |
| **`SRC_EXT_OMNILINGUAL`** | Meta Omnilingual ASR Kashmiri Slice | Meta AI | Multilingual read benchmark | 788 | 1.15h | CC-BY-NC 4.0 | `kashmiri_dataset/processed/audio/meta_omnilingual/` |
| **TOTAL** | **Consolidated Kashmiri Speech Corpus** | — | **All Modalities** | **106,487** | **214.65h** | — | `kashmiri_dataset/processed/audio/` |

---

## 2. Text Corpora for Tokenization & Language Modeling

| Corpus Identifier | Name & Scope | Source / Provider | Total Utterances / Sentences | License | Primary Research Utility |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **`SRC_TEXT_AYA`** | Aya Kashmiri Instruction Corpus | Cohere For AI Aya Initiative | 104,230 instructions | Apache 2.0 | Vocabulary coverage & synthetic prompt modeling |
| **`SRC_TEXT_BPCC`** | Bharat Parallel Corpus Collection | AI4Bharat / IIT Madras | 182,390 parallel pairs | CC-BY 4.0 | Perso-Arabic orthographic normalization |
| **`SRC_TEXT_WIKI`** | Kashmiri Wikipedia Full XML Dump | Wikimedia Foundation | 5,214 articles (120k sents) | CC-BY-SA 3.0 | KenLM n-gram language model training |
| **`SRC_TEXT_RASA`** | RASA Kashmiri Transliteration Pairs | Krish / Community | 64,800 pairs | CC-BY 4.0 | Supervised Roman-to-Perso-Arabic dictionary |
| **`SRC_TEXT_QURAN`** | Mirwaiz Written Translation Inventory | Izan Majeed / Mirwaiz Ingestion | 6,236 Ayahs / 18,500 sents | Public Educational | Religious lexicon extraction & KenLM tuning |

---

## 3. Standard Split Definitions and Benchmark Sets

To prevent data contamination and guarantee strict scientific integrity, all training, validation, and test partitions enforce **zero speaker overlap**.

### 3.1 Core Acoustic Training Splits (`kashmiri_dataset/experiments/native_asr_training/data_splits/`)
- **`train.jsonl`**: Progressive training sets:
  - B-01: 53,460 samples (92.94 hrs)
  - B-02A: 64,522 samples (104.93 hrs)
  - B-02B: 80,106 samples (123.45 hrs)
  - B-02C: 96,664 samples (165.32 hrs)
- **`val.jsonl`**: Standard development benchmark:
  - **Sample Count**: 6,728 utterances (dev canonical) / 7,258 records total
  - **Total Words**: 104,685 words
  - **Speaker Overlap**: Strictly 0.00% disjoint from all training splits.
  - **Domain Diversity**: Bolbosh (2,830 utts), IndicVoices (3,164 utts), XKaab (582 utts), OpenSLR (152 utts).
- **`test.jsonl`**: Sealed held-out test split (untouched during hyperparameter selection).

### 3.2 Canonical Diagnostic Benchmark Partitions (`kashmiri_dataset/manifests/`)
- **`test_clean.jsonl`**: Studio near-field read speech (SNR $>25$ dB).
- **`test_noisy.jsonl`**: Environmental field recordings (SNR $<18$ dB).
- **`test_code_switched.jsonl`**: Native speech containing English, Urdu, or Hindi loanwords.
- **`test_speaker_unseen.jsonl`**: Strictly isolated speakers never encountered in any prior training phase.

---

## 4. Audio Quality and Validation Gates

Before inclusion in any training or evaluation manifest, every audio sample must pass the five rigorous validation gates implemented in `kashmiri_dataset/scripts/pretraining_validation_gates.py`:

1. **Format Validation**: Mandatory 16kHz, 1-channel, 16-bit PCM WAV.
2. **Duration Feasibility**: $0.6 \text{ s} \le \text{Duration} \le 25.0 \text{ s}$ (prevents out-of-memory errors and empty clips).
3. **CTC Length Feasibility**: Audio frames $T = \lfloor 16000 \times \text{duration} / 320 \rfloor \ge L$ (where $L$ is transcript character length).
4. **Signal-to-Noise Ratio (SNR)**: Estimated via Welford/RMS energy ratio. Tier A: $>20$ dB; Tier B: $>12$ dB. Samples $<10$ dB excluded.
5. **Perso-Arabic Orthography**: Standardized via Unicode NFKC normalization, resolving Kashmiri letters (e.g., Keheh `ک`, Heh Goal `ہ`, Farsi Yeh `ی`, and diacritic marks `ٲ`, `ٳ`, `ۄ`, `ؠ`).
