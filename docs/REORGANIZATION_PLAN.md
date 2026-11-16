# Kashmiri AI Scientific Repository Reorganization & Migration Plan

**Project**: EduCasheer Kashmiri Automatic Speech Recognition Research  
**Authors**: Principal AI Research Scientists, ML Infrastructure Architects, Reproducibility Auditors  
**Date**: November 16, 2026  
**Status**: Architecture Approved; Phased Non-Destructive Staging  

---

## 1. Executive Summary & Problem Formulation

The EduCasheer Kashmiri AI project has developed organically across 24 research stages, resulting in over 243,700 files and 669.3 GB of storage assets. While the scientific outcomes are groundbreaking (establishing the SOTA E22-01 champion at 24.74% WER), the repository exhibits significant structural friction:

1. **Root Directory Clutter**: Operational inference scripts (`transcribe_kashmiri.py`, `test_batch_transcribe.py`, `web_asr_app.py`), large manifests (`asr_training_eligible_manifest.jsonl`, 183MB), and historical stage reports (`QURAN_...`, `WORKSTREAM_9_...`) reside in the root alongside Git-tracked symlinks.
2. **Monolithic `kashmiri_dataset` Namespace**: A single 622 GB directory contains datasets, checkpoints, tokenizers, research logs, scripts, and 37 distinct experiment subdirectories.
3. **External Data Shards (`Data/`)**: 43 GB of raw uncompressed Google Drive zip extractions sit at root with arbitrary archive folder names (`Kashmiri Project-20261101T102755Z-1-001..022`).
4. **Scattered Script Hierarchy**: 154+ scripts reside in `kashmiri_dataset/scripts/`, while others reside inside individual experiment directories.

---

## 2. Target Professional Architecture (Part 9 Design)

The target architectural layout organizes the codebase into modular, self-contained functional domains:

```text
kashmiri-ai/
├── README.md                                  # Executive overview, SOTA metrics, quickstart
├── PROJECT_MASTER_GUIDE.md                    # 20-section definitive technical handbook
├── REPOSITORY_MAP.md                          # Global component & navigation directory map
├── pyproject.toml                             # Standardized packaging & dependency manifest
│
├── configs/                                   # Universal configuration profiles
│   ├── asr_champion_e22_01.yaml               # Certified production ASR hyperparameters
│   ├── decoder_kenlm_calibrated.yaml          # Optimal decoding parameters (W=64, a=0.30, b=-1.00)
│   └── text_normalization_rules.json          # Kashmiri Unicode standard glyph mappings
│
├── src/                                       # Active, Reusable First-Party Source Code
│   ├── data_collection/                       # Harvesting, community ingestion, WAAZ pipelines
│   ├── data_processing/                       # Sinc resampling (16kHz), VAD, SNR filtering (>12dB)
│   ├── normalization/                         # Kashmiri Unicode NFKC & central vowel preservation
│   ├── datasets/                              # Manifest loaders, PyTorch audio datasets, dynamic batching
│   ├── asr/                                   # Wav2Vec2 / XLS-R 300M model wrappers, CTC classification
│   ├── decoding/                              # Exact CTC frame slicing, KenLM shallow fusion, MBR search
│   ├── evaluation/                            # Exact WER/CER computation, Error Atlas, significance tests
│   ├── translation/                           # ByT5 Romanization & transliteration neural adapters
│   ├── language_models/                       # KenLM 4-gram training & ARPA model wrappers
│   └── shared/                                # Common utilities, logging, GPU memory management
│
├── data/                                      # Data Management Pillar (Manifests & Audio References)
│   ├── README.md                              # Data governance, licensing, and access policy
│   ├── raw/                                   # Immutable raw archives & Google Drive shards
│   ├── processed/                             # Verified 16kHz mono PCM speech audio (Bolbosh, IndicVoices, etc.)
│   ├── manifests/                             # Training-eligible, acquisition, and QC manifests
│   ├── splits/                                # Strict zero-speaker overlap partitions (val, test, dev_assessment)
│   └── registries/                            # Dataset provenance & SHA-256 fingerprint catalogs
│
├── experiments/                               # Chronological Empirical Research Stages
│   ├── README.md                              # Stage index & research methodology handbook
│   ├── asr/                                   # Acoustic recognition experiments
│   │   ├── baselines/                         # B-01 (92.9h), Whisper zero-shot, IndicConformer
│   │   ├── scaling/                           # B-02A (104.9h), B-02B (123.5h), B-02C (165.3h)
│   │   └── stages/                            # Chronological stages (Stage 11 through Stage 24)
│   │       ├── 2026-11-14_stage-11_robustness_pilot/
│   │       ├── 2026-11-14_stage-14_mwer_optimization/
│   │       ├── 2026-11-14_stage-15_mwer_scaling/
│   │       ├── 2026-11-15_stage-20r_forensic_audit/
│   │       ├── 2026-11-15_stage-20s_audio_integrity/
│   │       ├── 2026-11-15_stage-22_data_expanded_training/
│   │       ├── 2026-11-15_stage-22c_inference_certification/
│   │       └── 2026-11-16_stage-24_actual_reranker_training/
│   └── transliteration/                       # ByT5 and neural adapter experiments
│
├── benchmarks/                                # Frozen Evaluation Benchmarks & Releases
│   ├── releases/                              # v1.0 and v1.1 canonical benchmark releases
│   ├── audit_reports/                         # Contamination audits & sample-rate integrity proofs
│   └── results/                               # Verified prediction JSONLs and bootstrap distributions
│
├── models/                                    # Model Registry & Certified Weights
│   ├── README.md                              # Model selection guide & checkpoint index
│   ├── registry/                              # Canonical manifests for E22-01, B-02C, B-02B
│   └── checkpoints/                           # Symbolic references to validated safetensors
│
├── docs/                                      # Central Documentation Hub
│   ├── architecture/                          # Technical diagrams, receptive field math
│   ├── codebase/                              # File-by-file audit & reading order guide
│   ├── data/                                  # Dataset lineage & source catalog
│   ├── research/                              # Stage registry, leaderboard, failed experiments
│   ├── reproducibility/                       # Step-by-step reproduction instructions
│   ├── decisions/                             # Architecture Decision Records (ADRs)
│   └── troubleshooting/                       # Known issues & forensic anomaly post-mortems
│
├── scripts/                                   # Maintenance & Operational Tools
│   ├── audit/                                 # Non-destructive inventory, manifest & link checkers
│   └── reproduce/                             # Automated evaluation harnesses
│
├── tests/                                     # Automated Unit & Regression Tests
└── archive/                                   # Deprecated & Superseded Artifacts
```

---

## 3. Phased Migration Strategy & Safeguards

To strictly obey **Non-Negotiable Rule 7** (*Never silently move files that active training or inference processes may depend on*) and **Rule 14** (*Preserve Git history and document original paths*), migration is strictly staged:

### Phase A: Read-Only Discovery (Completed)
- Mapped all 243,701 repository files (669.3 GB).
- Inspected all 416 first-party code files line-by-line via AST static analysis.
- Audited 37 research stages and verified model checkpoint weights on disk.

### Phase B: Organization Design (Completed)
- Designed target architecture.
- Built `docs/MIGRATION_MANIFEST.csv` detailing old path, new path, action, reason, and validation status for every artifact.

### Phase C: Non-Destructive Documentation & Tooling Implementation (Current State)
- Generated the complete documentation hub under `docs/` (`codebase/`, `data/`, `research/`, `architecture/`, `reproducibility/`, `decisions/`, `troubleshooting/`).
- Implemented automated maintenance and audit suite under `scripts/audit/run_audit_suite.py`.
- **Zero original data, checkpoint, or script files have been moved or deleted**, guaranteeing 100.0% backwards compatibility for running jobs.

### Phase D: Controlled Physical Migration (Future User-Approved Execution)
When the user explicitly authorizes physical directory relocation:
1. Create dedicated git migration branch `refactor/modular-architecture`.
2. Move files using `git mv` to preserve git commit history.
3. Replace original paths of key entry points (`transcribe_kashmiri.py`, `kashmiri_dataset/scripts/...`) with lightweight backward-compatibility forwarders/symlinks so existing shell scripts and external processes never break.
4. Verify checksums of moved checkpoints using `scripts/audit/run_audit_suite.py`.

### Phase E: Post-Migration Verification
- Run test suite and inference smoke test (`test_batch_transcribe.py`).
- Validate that all imports in `src/` resolve cleanly.
- Verify that documentation links remain valid.
