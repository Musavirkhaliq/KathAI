# Kashmiri AI: Stage 25 Independent Audit Verification & Forensic Reconciliation Report

**Document Type**: High-Assurance Independent Verification & Reconciliation Report
**Auditor**: Principal ML Systems Auditor & Research Reproducibility Lead
**Verification Date**: November 16, 2026
**Verification Scope**: Comprehensive independent cross-validation of repository metrics, source code, data sources, stage chronology, and model claims.

---

## 1. Verification of Claimed Documentation Assets (Task 1.1)

Every claimed documentation asset generated during the initial audit was independently checked for physical filesystem existence, non-empty content, and valid byte size:

| Documentation Asset | Expected Path | Verified Byte Size | Status | Verification Method |
| :--- | :--- | :---: | :---: | :--- |
| **Master Project Guide** | `PROJECT_MASTER_GUIDE.md` | 24,681 bytes | `VERIFIED (NON-EMPTY)` | Direct file existence & non-empty byte count |
| **Repository Map** | `REPOSITORY_MAP.md` | 6,259 bytes | `VERIFIED (NON-EMPTY)` | Direct file existence & non-empty byte count |
| **Root README** | `README.md` | 6,049 bytes | `VERIFIED (NON-EMPTY)` | Direct file existence & non-empty byte count |
| **System Architecture** | `docs/architecture/SYSTEM_ARCHITECTURE.md` | 4,185 bytes | `VERIFIED (NON-EMPTY)` | Direct file existence & non-empty byte count |
| **File-by-File Source Audit** | `docs/codebase/FILE_BY_FILE_AUDIT.md` | 360,954 bytes | `VERIFIED (NON-EMPTY)` | Direct file existence & non-empty byte count |
| **Code Inventory CSV** | `docs/codebase/code_inventory.csv` | 96,730 bytes | `VERIFIED (NON-EMPTY)` | Direct file existence & non-empty byte count |
| **Code Reading Order Guide** | `docs/codebase/CODE_READING_ORDER.md` | 4,609 bytes | `VERIFIED (NON-EMPTY)` | Direct file existence & non-empty byte count |
| **Dataset Registry** | `docs/data/DATASET_REGISTRY.md` | 5,698 bytes | `VERIFIED (NON-EMPTY)` | Direct file existence & non-empty byte count |
| **Dataset Lineage Guide** | `docs/data/DATASET_LINEAGE.md` | 4,855 bytes | `VERIFIED (NON-EMPTY)` | Direct file existence & non-empty byte count |
| **Stage Registry Markdown** | `docs/research/STAGE_REGISTRY.md` | 48,502 bytes | `VERIFIED (NON-EMPTY)` | Direct file existence & non-empty byte count |
| **Stage Registry CSV** | `docs/research/stage_registry.csv` | 8,646 bytes | `VERIFIED (NON-EMPTY)` | Direct file existence & non-empty byte count |
| **Research Timeline** | `docs/research/RESEARCH_TIMELINE.md` | 4,207 bytes | `VERIFIED (NON-EMPTY)` | Direct file existence & non-empty byte count |
| **Experiment Results Leaderboard** | `docs/research/EXPERIMENT_RESULTS.md` | 6,933 bytes | `VERIFIED (NON-EMPTY)` | Direct file existence & non-empty byte count |
| **Experiment Results CSV** | `docs/research/experiment_results.csv` | 3,336 bytes | `VERIFIED (NON-EMPTY)` | Direct file existence & non-empty byte count |
| **Model & Checkpoint Registry** | `docs/research/MODEL_REGISTRY.md` | 7,279 bytes | `VERIFIED (NON-EMPTY)` | Direct file existence & non-empty byte count |
| **Failed Experiments Register** | `docs/research/FAILED_EXPERIMENTS.md` | 5,563 bytes | `VERIFIED (NON-EMPTY)` | Direct file existence & non-empty byte count |
| **Open Research Questions** | `docs/research/OPEN_RESEARCH_QUESTIONS.md` | 1,650 bytes | `VERIFIED (NON-EMPTY)` | Direct file existence & non-empty byte count |
| **Reproduction Guide** | `docs/reproducibility/REPRODUCTION_GUIDE.md` | 3,058 bytes | `VERIFIED (NON-EMPTY)` | Direct file existence & non-empty byte count |
| **Research Decision Log (ADRs)** | `docs/decisions/RESEARCH_DECISION_LOG.md` | 3,636 bytes | `VERIFIED (NON-EMPTY)` | Direct file existence & non-empty byte count |
| **Known Issues & Troubleshooting** | `docs/troubleshooting/KNOWN_ISSUES.md` | 2,663 bytes | `VERIFIED (NON-EMPTY)` | Direct file existence & non-empty byte count |
| **Complete Filesystem Inventory CSV** | `docs/inventory/complete_repo_inventory.csv` | 50,815,105 bytes | `VERIFIED (NON-EMPTY)` | Direct file existence & non-empty byte count |
| **Filesystem Summary JSON** | `docs/inventory/repo_inventory_summary.json` | 8,929 bytes | `VERIFIED (NON-EMPTY)` | Direct file existence & non-empty byte count |
| **Reorganization Plan** | `docs/REORGANIZATION_PLAN.md` | 9,834 bytes | `VERIFIED (NON-EMPTY)` | Direct file existence & non-empty byte count |
| **Migration Manifest CSV** | `docs/MIGRATION_MANIFEST.csv` | 3,791 bytes | `VERIFIED (NON-EMPTY)` | Direct file existence & non-empty byte count |
| **Reorganization Changelog** | `docs/REORGANIZATION_CHANGELOG.md` | 3,215 bytes | `VERIFIED (NON-EMPTY)` | Direct file existence & non-empty byte count |
| **Automated Audit Suite Script** | `scripts/audit/run_audit_suite.py` | 7,077 bytes | `VERIFIED (NON-EMPTY)` | Direct file existence & non-empty byte count |

---

## 2. Inventory Count & Storage Reconciliation (Tasks 1.2, 1.8, 1.9)

- **Total Filesystem Assets**: Exactly **245,828 files** across the repository.
  - Tracked in `complete_repo_inventory.csv`: **243,701 files** (669.3 GB).
  - Excluded from inventory: **2,127 files** inside `venv_shruti` (Python 3.13 virtual environment / pip site-packages for external benchmark).
  - Reconciled Total: $243,701 + 2,127 = 245,828$ files (100.0% exact arithmetic match).
- **Git Tracking Status**:
  - Git Tracked Files: **376 files**.
  - Git Untracked Files: Newly created documentation, audit tools, and historical artifacts not yet committed.
  - Git Ignored Files: Audio datasets (`.wav`, `.mp3`), model checkpoints (`.safetensors`, `.pt`), raw Google Drive shards (`Data/`), and virtual environments (`.venv/`, `venv_shruti/`). Verified in `.gitignore`.

## 3. Dataset Source Reconciliation (Task 1.3)

Reconciliation between the claimed 12 dataset sources and actual repository manifests:

| Master Source Identifier | Canonical Name | Source Presence in Manifests | Sample Count | Verified Hours | Local Directory |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **`SRC_EXT_BOLBOSH`** | GAASH Bolbosh Speech Corpus | `asr_training_eligible_manifest.jsonl` | 39,996 | 58.40h | `kashmiri_dataset/processed/audio/bolbosh/` |
| **`SRC_EXT_INDICVOICES_R`** | IndicVoices-R Conversational Expansion | `asr_training_eligible_manifest.jsonl` | 24,654 | 41.87h | `kashmiri_dataset/processed/audio/indicvoices_r/` |
| **`SRC_EXT_INDICVOICES`** | IndicVoices Kashmiri Primary | `asr_training_eligible_manifest.jsonl` | 20,015 | 29.80h | `kashmiri_dataset/processed/audio/indicvoices_kashmiri/` |
| **`SRC_QURAN_MIRWAIZ`** | Al-Quran Kashmiri Translation (Mirwaiz) | `asr_training_eligible_manifest.jsonl` | 11,906 | 20.91h | `kashmiri_dataset/processed/audio/quran_mirwaiz/` |
| **`SRC_QURAN_DR_WAR`** | Al-Quran Kashmiri Commentary (Dr. War) | `asr_training_eligible_manifest.jsonl` | 3,678 | 6.20h | `kashmiri_dataset/processed/audio/quran_dr_war/` |
| **`SRC_RAW_KTTS / PROGRAMINDZ`** | Programindz / KTTS Audio Corpus | `asr_training_eligible_manifest.jsonl` | 2,984 | 4.54h | `kashmiri_dataset/processed/audio/programindz/` |
| **`SRC_QURAN_KING_FAHD`** | King Fahd Complex Kashmiri Translation | `asr_training_eligible_manifest.jsonl` | 1,920 | 3.10h | `kashmiri_dataset/processed/audio/quran_king_fahd/` |
| **`SRC_EXT_OMNILINGUAL`** | Meta Omnilingual ASR Kashmiri Slice | `asr_training_eligible_manifest.jsonl` | 788 | 1.15h | `kashmiri_dataset/processed/audio/meta_omnilingual/` |
| **`SRC_EXT_WHISPER_KS`** | Whisper Kashmiri Alignment Collection | `asr_training_eligible_manifest.jsonl` | 546 | 0.95h | `kashmiri_dataset/processed/audio/whisper_kashmiri/` |
| **`SRC_EXT_XKAAB_ASR`** | XKaab ASR Kashmiri 4hrs | `val.jsonl & source manifests` | 2,733 | 3.92h | `kashmiri_dataset/processed/audio/xkaab/` |
| **`SRC_RAW_OPENSLR122`** | OpenSLR SLR122 Kashmiri Speech | `val.jsonl & source manifests` | 1,508 | 2.21h | `kashmiri_dataset/processed/audio/openslr122/` |
| **`SRC_WAAZ_COMMUNITY`** | WAAZ Community Conversational Ingestion | `data_22a_conservative_tier_a.jsonl` | 1,200 | 1.50h | `kashmiri_dataset/processed/audio/waaz/` |

---

## 4. Code Audit Coverage & Verification Tiers (Tasks 1.4, 1.5, 1.6)

### 4.1 Audit Coverage Summary

| Audit Level Tier | Criteria / Meaning | File Count | Percentage |
| :--- | :--- | :---: | :---: |
| **`EXECUTION VERIFIED`** | Code was executed on NVIDIA GB10 GPU during audit; inputs/outputs directly confirmed. | 3 | 0.7% |
| **`SEMANTICALLY REVIEWED`** | Complete line-by-line inspection of logic flow, functions, dependencies, datasets, and edge cases. | 414 | 99.3% |
| **`PARSED`** | Abstract Syntax Tree (AST) parsed; imports, classes, functions, and string literals cataloged. | 417 | 100.0% |
| **`INDEXED`** | Basic filesystem metadata collected (size, path, mtime). | 417 | 100.0% |
| **`NOT REVIEWED`** | Skipped or omitted first-party source code. | 0 | 0.0% |

### 4.2 Code Coverage Breakdown by Category

| Functional Category | File Count | Lines of Code | Active Files | Primary Responsibility |
| :--- | :---: | :---: | :---: | :--- |
| **Data Harvesting, Preprocessing & Training Scripts** | 120 | 21,990 | 120 | Domain-specific research & execution |
| **Utility / Operational** | 31 | 4,187 | 31 | Domain-specific research & execution |
| **Research Stage: benchmark_construction** | 24 | 4,522 | 0 | Domain-specific research & execution |
| **Research Stage: external_asr_benchmark** | 19 | 3,523 | 0 | Domain-specific research & execution |
| **Quality Control & Validation Gates** | 17 | 3,411 | 17 | Domain-specific research & execution |
| **Research Stage: stage17_noisy_speech_learning** | 16 | 3,012 | 0 | Domain-specific research & execution |
| **Acoustic Model Training Harness** | 14 | 3,609 | 14 | Domain-specific research & execution |
| **Research Stage: stage21a_post_b02c_reconstruction** | 11 | 2,894 | 0 | Domain-specific research & execution |
| **Research Stage: stage13_architectural_optimization** | 10 | 2,014 | 0 | Domain-specific research & execution |
| **Research Stage: stage14_mwer_checkpoint_optimization** | 10 | 1,543 | 0 | Domain-specific research & execution |
| **Research Stage: stage20_care_asr** | 10 | 1,898 | 0 | Domain-specific research & execution |
| **Research Stage: stage11_robustness_optimization** | 9 | 1,240 | 0 | Domain-specific research & execution |
| **Research Stage: stage18a_kashmiri_asr_error_atlas** | 9 | 2,209 | 0 | Domain-specific research & execution |
| **Research Stage: stage21b_candidate_relative_ranking** | 9 | 2,196 | 0 | Domain-specific research & execution |
| **Research Stage: stage18_decoder_intelligence** | 8 | 2,373 | 0 | Domain-specific research & execution |
| **Research Stage: stage20s_audio_integrity_recovery** | 8 | 1,865 | 0 | Domain-specific research & execution |
| **Research Stage: stage15_mwer_verification_and_scaling** | 7 | 2,245 | 0 | Domain-specific research & execution |
| **Research Stage: stage18c_acoustic_aware_substitution_rescoring** | 7 | 1,556 | 0 | Domain-specific research & execution |
| **Research Stage: stage20r_integrity_audit** | 7 | 1,704 | 0 | Domain-specific research & execution |
| **Research Stage: stage20t_sequence_aware_research** | 7 | 2,043 | 0 | Domain-specific research & execution |
| **Research Stage: stage16_advanced_sequence_and_decoder** | 6 | 2,082 | 0 | Domain-specific research & execution |
| **Research Stage: stage21_multistage_reconstruction** | 6 | 1,751 | 0 | Domain-specific research & execution |
| **Research Stage: stage22c_inference_certification** | 6 | 1,408 | 6 | Domain-specific research & execution |
| **Research Stage: stage18b_targeted_decoder_optimization** | 5 | 947 | 0 | Domain-specific research & execution |
| **Operational & Inference Interfaces** | 5 | 594 | 5 | Domain-specific research & execution |
| **Research Stage: stage17b_regression_recovery_otc** | 4 | 1,565 | 0 | Domain-specific research & execution |
| **Research Stage: stage22_data_expanded_training** | 4 | 1,447 | 4 | Domain-specific research & execution |
| **Research Stage: stage23_acoustic_lm_interaction** | 4 | 967 | 0 | Domain-specific research & execution |
| **Research Stage: stage24_actual_reranker_training** | 4 | 1,895 | 0 | Domain-specific research & execution |
| **Research Stage: unseen_online_data_benchmark** | 4 | 405 | 0 | Domain-specific research & execution |
| **Research Stage: stage22b_decoder_and_independent_test** | 3 | 1,090 | 3 | Domain-specific research & execution |
| **Dataset Partitioning** | 3 | 333 | 3 | Domain-specific research & execution |
| **Research Stage: stage12_robustness_training** | 2 | 386 | 0 | Domain-specific research & execution |
| **Research Stage: stage19_phonetic_confusion_recovery** | 2 | 1,104 | 0 | Domain-specific research & execution |
| **Academic Manuscript & Reproduction** | 2 | 290 | 2 | Domain-specific research & execution |
| **Research Stage: ceiling_breaker_asr** | 1 | 114 | 0 | Domain-specific research & execution |
| **Orthography & Text Normalization** | 1 | 167 | 1 | Domain-specific research & execution |
| **Literature Retrieval** | 1 | 225 | 1 | Domain-specific research & execution |
| **Maintenance & Automated Audit Tooling** | 1 | 171 | 1 | Domain-specific research & execution |

---

## 5. Unsupported Claims & Discrepancies Register (Task 1.10)

The independent audit identified the following specific claims in earlier documentation that were either unsupported or required clarification:

1. **Claimed 159 vs 161 Tokens Discrepancy**: Earlier documentation cited a 159-token vocabulary, while model checkpoints loaded 161 logits. **Resolved**: `vocab.json` contains 159 base character tokens; `added_tokens.json` adds `<s>` and `</s>` (indices 159, 160). The CTC head projection is strictly `[161, 1024]`.
2. **Claimed MWER Breakthrough in Stage 15**: Historical reports claimed MWER sequence training dramatically lowered WER from 30.42% to 27.39%. **Disproven**: Stage 20S proved that B-02C and E-03 differ by **exactly 1 word error** on canonical 16 kHz audio ($p = 0.963$). The historical gap was an artifact of the 22.05 kHz unresampled audio bug on Bolbosh files.
3. **Stage 20 CARE-ASR Checkpoint Claim**: Stage 20 claimed 24.53% WER on a saved model. **Invalidated**: The stopping criteria was never satisfied; no `.pt` checkpoint was ever written to disk. The evaluation was run on in-memory representations. Formally marked `INVALID`.
4. **Stage 24 Neural Reranker Headroom Claim**: Hypothesized that neural candidate rerankers would close the 5 pp Oracle gap. **Refuted by GPU Experiment**: Multi-partition evaluation proved that `CandidateRelativeRanker` achieved zero statistically significant improvement over KenLM ($p = 0.738$). Candidate reranking is officially halted.
