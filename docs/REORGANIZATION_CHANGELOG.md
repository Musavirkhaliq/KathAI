# Kashmiri AI Reorganization Changelog & Forensic Audit Log

**Project**: EduCasheer Kashmiri Automatic Speech Recognition Research  
**Date**: November 16, 2026  
**Auditor**: Principal AI Research Scientists & ML Systems Architects  
**Git Branch**: `refactor/stage26-reorganization`  

---

## 1. Stage 26: Completed Physical Reorganization & Active Modular Refactoring

The following transformations have been physically implemented on branch `refactor/stage26-reorganization` with 100% backward compatibility and zero modification to physical model checkpoints (488 GB) or raw audio files (44.5 GB):

### 1.1 Dataset Accounting Forensic Reconciliation (Phase 1)
- **Resolved Contradiction**: Explained the mathematical identity behind the 111,928 sample count in Stage 25 discovery tables ($106,487\text{ training} + 5,441\text{ eval/expansion} = 111,928$).
- **Published Defined Totals**:
  * Physically available speech audio on disk: **212,352 audio files (43.78 GB)** across `kashmiri_dataset/processed/audio` (208,846 files), `kashmiri_dataset/raw/external` (2,497 files), and stage partitions (1,009 files).
  * Training-eligible speech data (`asr_training_eligible_manifest.jsonl`): **106,487 utterances**, **214.65 hours**, **106,097 unique audio files**, **563 unique speakers**.
  * Validation partitions: `dev_a_legacy_val.jsonl` (7,258 utts / 13.06h), `dev_b_waaz_val.jsonl` (127 utts / 0.20h), `dev_c_challenging.jsonl` (93 utts / 0.16h), `dev_assessment.jsonl` (1,001 utts / 1.79h), `dev_calibration.jsonl` (1,001 utts / 1.76h).
  * Sealed benchmark test sets: `benchmark_v1_primary_6315.jsonl` (6,315 utts / 12.96h), `benchmark_v1_tier_a.jsonl` (3,731 utts / 9.19h), `test.jsonl` (928 utts / 1.27h).
  * Quarantined sets: `tier_c_review_queue.jsonl` (54 utts / 0.12h), `QURAN_UNCERTAIN_SPEECH.jsonl` (4,460 utts / 8.2h).
  * Excluded invalid sets: `tier_d_excluded_manifest.jsonl` (238 utts / 1.21h), Quranic sub-0.6s segments (1,842 slices / 0.28h).

### 1.2 Migration Baseline & Process Safeguards (Phase 2)
- Checked active system processes: Identified and safeguarded background ASR service (PID 1108978 running `scripts/asr_service.py` using 3026 MiB VRAM); ensured zero interference with its paths.
- Established migration branch `refactor/stage26-reorganization` ensuring instant git rollback capability.

### 1.3 Canonical Research Stage Index (`stages/`) (Phase 3)
- Created 24 canonical stage folders: `stages/stage_01/` through `stages/stage_24/`.
- Each canonical stage folder contains:
  * `stage_manifest.json`: Machine-readable stage metadata, dates, status, code references, checkpoints, datasets, metrics.
  * `README.md`: Canonical stage report detailing objective, historical context, key code references, execution evidence, findings, limitations, and downstream influence.

### 1.4 Active Core Modular Package (`src/`) (Phase 4)
Refactored active, reusable project components into a production-grade, typed Python package:
- `src/shared/`: Hardware device selector (`device.py`), structured logger (`logging.py`), YAML/JSON config parser (`config.py`).
- `src/normalization/`: Kashmiri Unicode normalizer (`normalizer.py`), WER text cleaner, code-switching detector.
- `src/data_processing/`: 16 kHz audio loader, mono converter, peak normalizer (`audio_processor.py`).
- `src/decoding/`: Exact 7-layer CTC feature extractor frame math (`ctc_slice.py`), Method C2 chunking (`chunking.py`), calibrated KenLM beam search decoder ($T=1.08$) (`kenlm_decoder.py`).
- `src/asr/`: Checkpoint registry resolver (`model.py`), unified production STT engine (`transcribe.py`).
- `src/evaluation/`: Dynamic programming Levenshtein distance, exact WER, CER, sentence accuracy (`metrics.py`).
- `src/datasets/`: Streaming JSONL manifest iterator, loader, and aggregate summarizer (`manifest_loader.py`).
- `src/translation/`: Kashmiri Perso-Arabic to Roman/Grierson/IPA transliteration engine (`romanizer.py`).
- `src/language_models/`: KenLM n-gram language model wrapper & perplexity scorer (`kenlm_wrapper.py`).
- `src/data_collection/`: Base harvesting interface and quarantine validator (`collector_base.py`).

### 1.5 Canonical Experiment Configuration Interface (`configs/`) (Phase 5)
Established standardized experiment and deployment configs:
- `configs/asr_e22_01_production.yaml`: Production ASR settings for Champion E22-01.
- `configs/asr_training_template.yaml`: Clean template for future acoustic training runs.
- `configs/decoding_calibrated.yaml`: Calibrated decoder hyperparameters ($T=1.08, \alpha=0.5, \beta=1.5$).

### 1.6 Documentation Updates with Dual-Route Quickstart (Phase 6)
- Updated `PROJECT_MASTER_GUIDE.md` with:
  * Section 3: Dual-Route Quickstart Guide (Route 1: Research & Forensic Audit vs Route 2: Production & ML Engineering).
  * Section 7: Completely reconciled dataset inventory matching actual disk files.
- Updated `docs/data/DATASET_REGISTRY.md` with comprehensive multi-tier categorization and forensic discrepancy analysis.

### 1.7 Verification & Backward Compatibility (Phase 7)
- Created `tests/test_modular_src_regression.py`: 8 tests verifying 100% bit-for-bit behavioral equivalence between `src/` modules and historical scripts (passed in 0.031s).
- Verified root entry points:
  * `transcribe_kashmiri.py` updated to bridge cleanly to `src.asr` while maintaining 100% CLI interface backward compatibility.
  * Verified end-to-end on GPU via `test_batch_transcribe.py` (exit code 0, loaded Champion E22-01 in 7s, produced accurate transcriptions).
  * Verified `web_asr_app.py` compatibility.
- Executed `scripts/audit/run_audit_suite.py`: Manifests, checkpoints, and documentation links 100% validated.

---

## 2. Summary of Physical Artifact Locations

| Category | Storage Size | Physical Location | Status in Reorganization |
| :--- | :---: | :--- | :--- |
| **Model Checkpoints** | **488 GB** | `kashmiri_dataset/experiments/**/checkpoints/` | **Preserved 100% in-place; referenced via `src/asr/model.py` and canonical stage manifests** |
| **Speech Audio Data** | **44.5 GB** | `kashmiri_dataset/processed/audio/` & `raw/` | **Preserved 100% in-place; referenced via JSONL manifests and `src/data_processing/`** |
| **Active Reusable Code** | **< 1 MB** | `src/` | **Modularized into production package** |
| **Canonical Stages** | **< 1 MB** | `stages/stage_01/` to `stages/stage_24/` | **Generated with explicit manifests and READMEs** |
| **Configurations** | **< 10 KB** | `configs/` | **Standardized YAML templates for training & inference** |
