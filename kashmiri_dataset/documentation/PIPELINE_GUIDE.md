# Kashmiri Dataset Pipeline: Reproduction & Execution Guide

This guide details how to execute, reproduce, and customize the data engineering and speech processing pipeline for the Kashmiri multimodal dataset.

---

## 1. Directory Structure Overview

All scripts are located in `kashmiri_dataset/scripts/`:

```text
kashmiri_dataset/scripts/
├── audit_dataset.py             # Audits raw files across 22 shards and generates initial audit report
├── validate_audio.py            # Audio converter, slicing, and 16kHz PCM standardization
├── normalize_transcripts.py     # Kashmiri Perso-Arabic normalizer and code-switching detector
├── detect_duplicates.py         # SHA256 and MD5 duplicate audio & text tracker
├── quality_filter.py            # Energy-based SNR estimator and VAD quality scorer
├── build_manifest.py            # Ingests audio sources and outputs master JSONL & CSV manifests
├── create_splits.py             # Partitions dataset with strict speaker isolation
├── dataset_statistics.py        # Computes comprehensive metrics across text and audio
├── discover_external_datasets.py# Cataloger for external public resources
└── run_full_pipeline.py         # Master orchestrator that executes the entire workflow
```

---

## 2. Quickstart: Running the Master Pipeline

To execute the complete end-to-end processing pipeline in one command:

```bash
python3 "/home/musa/Kashmiri Project/kashmiri_dataset/scripts/run_full_pipeline.py"
```

This will automatically:
1. Verify raw data immutability.
2. Standardize audio from OpenSLR 122, KTTS, Audio Corpus, and XKaab ASR.
3. Apply NFC normalization and Kashmiri orthographic rules to all transcripts.
4. Estimate SNR, VAD, and quality scores for every clip.
5. Generate `manifest_master.jsonl` and `manifest_master.csv`.
6. Enforce speaker isolation and generate `train`, `val`, `test_clean`, `test_noisy`, `test_code_switched`, and `test_speaker_unseen` splits.
7. Output quantitative statistics to `quality_reports/dataset_summary_statistics.json`.

---

## 3. Step-by-Step Script Usage

### Step 1: Normalizing Kashmiri Text
To test or use the Kashmiri normalizer independently:

```python
from normalize_transcripts import normalize_kashmiri_text, detect_code_switching

text = "مِہ چھ ناو توسیف جیلانی"
normalized = normalize_kashmiri_text(text)
cs_info = detect_code_switching(text)

print(normalized)
print(cs_info)
```

### Step 2: Audio Conversion & Inspection
To convert any audio file (e.g. WebM Opus, MP3, FLAC) to 16 kHz 16-bit Mono PCM WAV:

```python
from validate_audio import convert_to_wav16k, get_audio_info

convert_to_wav16k("input_file.opus", "output_16k.wav")
info = get_audio_info("output_16k.wav")
print(info)
```

### Step 3: Assessing Audio Quality & SNR
To compute SNR and quality flags:

```python
from quality_filter import estimate_snr_and_quality

q = estimate_snr_and_quality("output_16k.wav")
print(f"SNR: {q['snr_db']} dB, Quality Score: {q['quality_score']}, Flags: {q['flags']}")
```

### Step 4: Rebuilding Manifests & Splits
To re-generate manifests after adding new audio files:

```bash
python3 "/home/musa/Kashmiri Project/kashmiri_dataset/scripts/build_manifest.py"
python3 "/home/musa/Kashmiri Project/kashmiri_dataset/scripts/create_splits.py"
python3 "/home/musa/Kashmiri Project/kashmiri_dataset/scripts/dataset_statistics.py"
```
