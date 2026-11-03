# Research Log Entry 01: Complete Master Data Inventory

- **Timestamp**: 2026-11-02T11:28:00+05:30
- **Author/Investigator**: Antigravity AI Pair Programmer & Research Team
- **Phase**: Phase 1 — Complete Data Inventory & Layer Architecture

---

## 1. Objective
Establish an exhaustive, immutable baseline catalog of all Kashmiri language and speech resources currently stored in the workspace. Ensure complete separation between raw, processed, cleaned, merged, and derived layers to guarantee research reproducibility and prevent accidental loss or corruption of ground-truth data.

---

## 2. Directory Architecture & Layer Segregation

To adhere to rigorous scientific standards, the repository enforces a strict, hierarchical 6-layer separation:

```text
/home/musa/Kashmiri Project/
├── Data/                              <-- [LAYER 1: RAW / ORIGINAL (IMMUTABLE)]
│   └── Kashmiri Project-20261101T102755Z-1-* (Shards 001 to 022, 42.69 GB, read-only baseline)
│
├── kashmiri_dataset/
│   ├── external/                      <-- [LAYER 1B: RAW EXTERNAL INGESTION]
│   │   ├── gaash_bolbosh/             (Bolbosh 12.1 GB)
│   │   ├── humair025_kashmirispeech-indicvoices/ (17.35 GB)
│   │   ├── ai4bharat_bpcc/            (BPCC parallel corpus)
│   │   ├── aya_kashmiri/              (Aya 100k+ instructions)
│   │   └── ... (44 external dataset repositories, 48.93 GB)
│   │
│   ├── processed/                     <-- [LAYER 2: PROCESSED STANDARDIZED]
│   │   ├── audio/                     (16 kHz Mono 16-bit PCM WAV standard)
│   │   └── text/                      (Extracted plain text & raw transcript tokens)
│   │
│   ├── cleaned/                       <-- [LAYER 3: CLEANED & NORMALIZED]
│   │   ├── audio_vad_filtered/        (VAD silence trimmed, SNR >= 12 dB)
│   │   └── text_normalized/           (Unicode NFKC + Kashmiri orthographic canonicalization)
│   │
│   ├── merged/                        <-- [LAYER 4: CONSOLIDATED MASTER CORPUS]
│   │   ├── KASHMIRI-TEXT/             (Monolingual pretraining & SFT corpus)
│   │   ├── KASHMIRI-SPEECH/           (Paired ASR/TTS master speech manifests)
│   │   ├── KASHMIRI-PARALLEL/         (Bi-text translations: en-ks, hi-ks, ur-ks)
│   │   ├── KASHMIRI-TRANSLITERATION/  (Perso-Arabic <-> Devanagari <-> Romanized)
│   │   └── KASHMIRI-LEXICON/          (Pronunciation dictionaries, Grierson, Hassan)
│   │
│   ├── splits/                        <-- [LAYER 5: SCIENTIFIC SPLITS (ZERO CONTAMINATION)]
│   │   ├── train/                     (Speaker-isolated training sets)
│   │   ├── val/                       (Validation sets)
│   │   └── test/                      (test_clean, test_noisy, test_unseen_speaker, test_dialect)
│   │
│   ├── provenance/                    <-- [LAYER 6: AUDIT TRAILS & LINEAGE]
│   │   ├── master_dataset_inventory.csv
│   │   ├── master_dataset_inventory.json
│   │   └── sample_provenance_tracker.jsonl
│   │
│   ├── research_log/                  <-- Chronological research journal
│   ├── research_decision_log.md       <-- Explicit design decisions & justification
│   └── paper/                         <-- Evidence collection for academic publication
```

---

## 3. Quantitative Summary of Inventory

- **Total Ingested Repositories/Sources**: 45 distinct collections
- **Total Raw Storage Footprint**: **91.62 GB**
  - Raw GDrive Archives: 42.69 GB (3,845 files)
  - External Downloaded Repositories: 48.93 GB (44 repositories)
- **Modality Breakdown**:
  - **Speech / Audio**: 75.20 GB (Raw GDrive audio + Bolbosh + IndicVoices + Whisper-Kashmiri + Programindz TTS + Omar TTS)
  - **Text / Parallel / Reasoning**: 13.34 GB (BPCC, Aya, XP3x, Rasa Code-mixed, Clinical Reasoning 1–4, Medical Math, Kreedz)
  - **Vision / OCR**: 3.08 GB (Koshur-SynthOCR, Omar Text Recognition)
  - **Lexicon / Pronunciation**: ~1.82 GB (Grierson Dictionary, Hassan Dictionary, Mzmmoazam Native Audio Pronunciations)

---

## 4. Discovered Nuances & Technical Caveats
1. **Opus Audio Hidden in .wav extension**: 2,984 files in `03_KTTS` inside the raw archive are containerized as Opus/WebM despite being named `.wav`. Raw files are left untouched in `Data/`, while standardized 16 kHz Mono PCM WAVs are generated strictly into `processed/audio/`.
2. **Long Unsegmented Audio**: `01_OpenSLR_SLR122` contains 43 long continuous recordings (10–30 minutes) accompanied by TSV timestamps. Segmented slices (1,508 clips) are extracted into `processed/audio/slr122/`.
3. **Multi-Script Diversity**: Kashmiri appears in three scripts:
   - Primary: Perso-Arabic (`kas_Arab`, Nastaliq / Naskh)
   - Secondary: Devanagari (`kas_Deva`, used extensively in religious, literary, and educational texts)
   - Tertiary: Romanized / Latin transliteration (frequent in SMS, social media, and Rasa conversational data)

---

## 5. Artifacts Created in this Step
- `kashmiri_dataset/provenance/master_dataset_inventory.csv`
- `kashmiri_dataset/provenance/master_dataset_inventory.json`
- `kashmiri_dataset/research_log/01_initial_inventory.md`
