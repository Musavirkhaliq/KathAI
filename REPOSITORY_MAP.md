# EduCasheer Kashmiri AI: Master Repository Directory Map

**Project**: EduCasheer Kashmiri Automatic Speech Recognition Research  
**Last Audited**: November 16, 2026 (Stages 1 through 24 Fully Reconstructed)  
**Total Cataloged Assets**: 243,701 files | 669.3 GB Storage Volume  

---

## 1. Top-Level Directory Navigation

```text
/home/musa/Kashmiri Project/
├── README.md                                  # Executive overview, SOTA metrics, quickstart
├── PROJECT_MASTER_GUIDE.md                    # 20-section definitive technical handbook
├── REPOSITORY_MAP.md                          # This file: global component & navigation map
│
├── docs/                                      # Central Documentation Hub
│   ├── architecture/                          # Technical diagrams, receptive field math
│   │   └── SYSTEM_ARCHITECTURE.md            # End-to-end architecture & mathematical derivations
│   ├── codebase/                              # Code inspection & onboarding
│   │   ├── FILE_BY_FILE_AUDIT.md              # Line-by-line AST audit of all 416 first-party files
│   │   ├── code_inventory.csv                 # Machine-readable code inventory with hyperparameters
│   │   └── CODE_READING_ORDER.md              # 8-step dependency-aware learning sequence
│   ├── data/                                  # Data governance & provenance
│   │   ├── DATASET_REGISTRY.md                # Authoritative catalog of all 12 speech sources (214.65h)
│   │   └── DATASET_LINEAGE.md                 # Transformation lifecycle from raw to 16kHz splits
│   ├── research/                              # Scientific records & verified results
│   │   ├── STAGE_REGISTRY.md                  # Detailed history of Stages 1 through 24
│   │   ├── stage_registry.csv                 # Machine-readable stage register
│   │   ├── RESEARCH_TIMELINE.md               # Chronological narrative of project milestones
│   │   ├── EXPERIMENT_RESULTS.md              # Reconciled canonical leaderboards post-Stage 20R/20S
│   │   ├── experiment_results.csv             # Machine-readable numerical metrics
│   │   ├── MODEL_REGISTRY.md                  # Checkpoint paths & SHA-256 cryptographic hashes
│   │   ├── FAILED_EXPERIMENTS.md              # Exhaustive register of 7 negative results & post-mortems
│   │   └── OPEN_RESEARCH_QUESTIONS.md         # Open research directions & future hypotheses
│   ├── reproducibility/                       # Verification & execution
│   │   └── REPRODUCTION_GUIDE.md              # Exact shell commands for smoke tests and audits
│   ├── decisions/                             # Architecture Decision Records (ADRs)
│   │   └── RESEARCH_DECISION_LOG.md           # 6 ADRs documenting major technical choices
│   ├── troubleshooting/                       # Debugging & issue post-mortems
│   │   └── KNOWN_ISSUES.md                    # Forensic anomaly register (resampling, OOM, trailing tokens)
│   ├── inventory/                             # Complete filesystem inventory
│   │   ├── complete_repo_inventory.csv        # Inventory of all 243,701 repository files
│   │   └── repo_inventory_summary.json        # High-level volume and type breakdown
│   ├── REORGANIZATION_PLAN.md                 # Modular target architecture & safe migration plan
│   ├── MIGRATION_MANIFEST.csv                 # Exact mapping from old paths to target paths
│   └── REORGANIZATION_CHANGELOG.md            # Audit trail of completed and proposed changes
│
├── kashmiri_dataset/                          # Primary Research & Production Workspace
│   ├── experiments/                           # All 24 research stages (B-01 through Stage 24)
│   │   ├── native_asr_training/               # Foundation training: B-01, B-02A, B-02B, B-02C
│   │   ├── stage20r_integrity_audit/          # Forensic discovery of 22.05 kHz resampling bug
│   │   ├── stage20s_audio_integrity_recovery/ # Canonical 16kHz benchmark re-evaluation
│   │   ├── stage22_data_expanded_training/    # SOTA Champion E22-01 (166.4h, 24.74% WER)
│   │   ├── stage22c_inference_certification/  # Exact receptive field frame slicing & chunking
│   │   ├── stage23_acoustic_lm_interaction/   # Search space topology & Oracle headroom mapping
│   │   └── stage24_actual_reranker_training/  # GPU neural candidate ranker & acoustic continuation
│   ├── processed/audio/                       # Standardized 16kHz mono PCM WAV speech audio
│   ├── manifests/                             # Canonical acquisition, QC, and split manifests
│   ├── tokenizers/                            # 159-token Character CTC and SentencePiece models
│   └── scripts/                               # 154 standalone research and pipeline scripts
│
├── Data/                                      # Raw Google Drive uncompressed shards (43 GB)
│
├── scripts/audit/                             # Automated non-destructive maintenance tools
│   └── run_audit_suite.py                     # Scans broken paths, validates manifests & checkpoints
│
├── research_papers/                           # Academic Research Manuscripts
│   └── kashmiri_asr_systematic_study/         # Primary systematic research paper suite
│       └── MASTER_RESEARCH_REPORT.md          # Comprehensive institutional scientific research report
│
├── transcribe_kashmiri.py                     # Primary CLI inference utility (Auto-loads Champion E22-01)
├── test_batch_transcribe.py                   # 30-second validation smoke test
├── web_asr_app.py                             # Browser GUI for microphone transcription & Romanization
├── record_and_transcribe.sh                   # Interactive terminal recorder + transcriber
└── run_training_terminal.sh                   # Safe terminal training launcher
```
