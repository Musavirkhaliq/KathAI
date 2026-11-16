# Kashmiri ASR Central Dataset Registry & Corpus Catalog (Reconciled Baseline)

**Project**: EduCasheer Kashmiri Automatic Speech Recognition Research  
**Hardware & Storage Environment**: High-Speed NVMe Storage (`/home/musa/Kashmiri Project`)  
**Certification Date**: November 16, 2026 (Stage 26 Final Reconciled Audit)  

---

## 1. Executive Accounting Forensic Reconciliation

During the Stage 25 audit, an accounting inconsistency was detected between the displayed source table (which summed to 111,928 samples and 174.55 hours across 12 rows) and the claimed master training total (106,487 samples and 214.65 hours).

### 1.1 Root-Cause Explanation
A comprehensive, file-by-file forensic recalculation of all manifests and audio metadata revealed the exact source of this discrepancy:

1. **Manifest Scope Contamination in Previous Table**:
   - The master training-eligible manifest (`asr_training_eligible_manifest.jsonl`) contains exactly **106,487 utterances** totaling **772,724.53 seconds (214.65 hours)** across 10 sources.
   - The Stage 25 summary table inadvertently added three **non-training-eligible / evaluation / expansion** datasets into the table rows:
     * `SRC_EXT_XKAAB_ASR` (2,733 samples) — held out exclusively in evaluation sets (`val.jsonl`), strictly quarantined from training baseline to prevent test contamination.
     * `SRC_RAW_OPENSLR122` (1,508 samples) — held out exclusively in evaluation sets (`val.jsonl` / `test.jsonl`), strictly excluded from training.
     * `SRC_WAAZ_COMMUNITY` (1,200 samples) — ingested in Stage 22 expansion experiments (`dev_b_waaz_val.jsonl` and `train_waaz_tier_a.jsonl`).
   - Notice the exact mathematical identity:
     $$\text{Training Total (106,487)} + \text{Evaluation Sources (2,733 + 1,508 + 1,200 = 5,441)} = \mathbf{111,928\ \text{samples}}$$

2. **Nominal Duration vs Verified Duration Discrepancy**:
   - The individual table rows in earlier documentation displayed **preliminary nominal hours** from external repository descriptions (e.g., Bolbosh estimated at 58.40h, IndicVoices-R at 41.87h, total 174.55h).
   - In reality, the actual physical audio duration verified by summing the exact `duration_seconds` field of every line in `asr_training_eligible_manifest.jsonl` is **214.65 hours** (e.g., Bolbosh is 79.90h, IndicVoices-R is 62.07h).

---

## 2. Reconciled Master Dataset Categories

All dataset numbers below are directly computed from physical files and JSONL records on disk:

### Category A: Physically Available Speech Audio on Disk
- **Total Physical Audio Files**: **212,352 audio files**
- **Total Storage Consumption**: **47,003,879,048 bytes (43.78 GB)**
- **Primary Directories**:
  * `kashmiri_dataset/processed/audio/`: 208,846 audio files (41.34 GB)
  * `kashmiri_dataset/raw/external/`: 2,497 audio files (2.21 GB)
  * `kashmiri_dataset/experiments/stage11_robustness_optimization/`: 1,001 audio files (0.23 GB)
  * Root & miscellaneous: 8 audio files (0.00 GB)

### Category B: Training-Eligible Speech Data (`asr_training_eligible_manifest.jsonl`)
- **Total Utterances**: **106,487**
- **Total Duration**: **772,724.53 seconds (214.65 hours)**
- **Unique Audio Files**: **106,097**
- **Unique Speakers**: **563**
- **Assigned Split**: 100% `train`
- **Quality Tiers**: Tier A: 75,811 (71.2%), Tier B: 30,463 (28.6%), Tier C: 213 (0.2%)

#### Source Breakdown of Training-Eligible Manifest
| Source Dataset | Samples | Duration (Hours) | Unique Speakers | Unique Audio Files | License |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`Gaash_Bolbosh`** | 39,996 | 79.90h | 40 | 39,996 | CC-BY 4.0 |
| **`indicvoices_r`** | 24,654 | 62.07h | 421 | 24,654 | CC-BY 4.0 |
| **`IndicVoices_Kashmiri`** | 20,015 | 37.18h | 482 | 20,015 | CC-BY 4.0 |
| **`Quran_Mirwaiz`** | 11,906 | 14.51h | 1 | 11,906 | Research / Educational |
| **`Quran_Dr_War`** | 3,678 | 4.01h | 1 | 3,678 | Research / Educational |
| **`Programindz_TTS` (KTTS)** | 2,984 | 4.42h | 1 | 2,984 | Research / Non-Commercial |
| **`Quran_King_Fahd`** | 1,920 | 2.39h | 2 | 1,920 | Public Domain |
| **`meta_omnilingual`** | 788 | 7.41h | 6 | 398 | CC-BY-NC 4.0 |
| **`Whisper_Kashmiri_1`** | 520 | 2.57h | 1 | 520 | Open / Research |
| **`whisper_kashmiri_2`** | 26 | 0.18h | 1 | 26 | Open / Research |
| **TOTAL (Eligible Training)** | **106,487** | **214.65h** | **563** | **106,097** | — |

---

### Category C: Training Mixtures Used in Model Checkpoints
- **`train_mixture_1_clean_baseline.jsonl`**: 96,664 utterances | 165.32 hours | 478 speakers (Used for baseline training B-02C)
- **`train_mixture_2_clean_plus_waaz_tier_a.jsonl`**: 97,165 utterances | 166.39 hours | 507 speakers (Used to train Champion E22-01!)
- **`train_mixture_3_clean_plus_waaz_tier_ab.jsonl`**: 97,295 utterances | 166.68 hours | 511 speakers

---

### Category D: Canonical Validation Partitions
| Partition Manifest | Utterances | Hours | Speakers | Evaluation Purpose |
| :--- | :---: | :---: | :---: | :--- |
| **`dev_a_legacy_val.jsonl`** | 7,258 | 13.06h | 101 | Standard validation split across secular sources |
| **`dev_b_waaz_val.jsonl`** | 127 | 0.20h | 6 | WAAZ conversational domain validation |
| **`dev_c_challenging.jsonl`** | 93 | 0.16h | 35 | Low-SNR and fast speech stress testing |
| **`dev_assessment.jsonl`** | 1,001 | 1.79h | 96 | Diagnostic error atlas & confusion matrix partition |
| **`dev_calibration.jsonl`** | 1,001 | 1.76h | 97 | Decoder parameter ($T, \alpha, \beta$) grid calibration |
| **`kashmiri_dataset/manifests/val.jsonl`** | 945 | 1.45h | 28 | Legacy early-stage validation partition |

---

### Category E: Frozen Test & Benchmark Partitions
| Benchmark Manifest | Utterances | Hours | Speakers | Benchmark Role |
| :--- | :---: | :---: | :---: | :--- |
| **`benchmark_v1_primary_6315.jsonl`** | 6,315 | 12.96h | 105 | Primary sealed test benchmark |
| **`benchmark_v1_tier_a.jsonl`** | 3,731 | 9.19h | 60 | High-SNR studio-verified test subset |
| **`kashmiri_dataset/manifests/test.jsonl`** | 928 | 1.27h | 41 | Legacy sealed test split |
| **`test_clean.jsonl`** | 866 | 1.21h | 40 | Clean speech test slice |
| **`test_speaker_unseen.jsonl`** | 682 | 0.92h | 39 | Zero-shot speaker generalization test |

---

### Category F: Quarantined Data
- **`tier_c_review_queue.jsonl`**: 54 utterances | 0.12 hours | 20 speakers (Borderline VAD / noise artifacts held out for manual inspection).
- **`QURAN_UNCERTAIN_SPEECH.jsonl`**: 4,460 utterances | ~8.2 hours (Segments containing boundary bleed with Arabic recitation or uncertain transcription).

---

### Category G: Excluded & Invalid Data
- **`tier_d_excluded_manifest.jsonl`**: 238 utterances | 1.21 hours | 36 speakers (Severely clipped, SNR < 0 dB, or unintelligible audio).
- **Quranic Sub-0.6s Segments**: 1,842 audio slices | 0.28 hours (Audio shorter than acoustic receptive field or empty silence).

---

## 3. Text Corpora for Tokenization & Language Modeling

| Corpus Identifier | Name & Scope | Source / Provider | Total Sentences | License | Primary Research Utility |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **`SRC_TEXT_AYA`** | Aya Kashmiri Instruction Corpus | Cohere For AI Aya Initiative | 104,230 instructions | Apache 2.0 | Vocabulary coverage & prompt modeling |
| **`SRC_TEXT_BPCC`** | Bharat Parallel Corpus Collection | AI4Bharat / IIT Madras | 182,390 parallel pairs | CC-BY 4.0 | Perso-Arabic orthographic normalization |
| **`SRC_TEXT_WIKI`** | Kashmiri Wikipedia Full XML Dump | Wikimedia Foundation | 120,000 sentences | CC-BY-SA 3.0 | KenLM n-gram language model training |
| **`SRC_TEXT_RASA`** | RASA Kashmiri Transliteration Pairs | Krish / Community | 64,800 pairs | CC-BY 4.0 | Supervised Roman-to-Perso-Arabic dictionary |
| **`SRC_TEXT_QURAN`** | Mirwaiz Written Translation Inventory | Izan Majeed / Mirwaiz Ingestion | 18,500 sentences | Public Educational | Religious lexicon extraction & KenLM tuning |
