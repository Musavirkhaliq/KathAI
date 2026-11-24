# EduCasheer Kashmiri AI — Master Audio-Text Inventory & Training Audit
**Directory**: `docs/data/full_audio_text_inventory/`  
**Author**: Principal Speech Dataset Researcher, Audio Data Engineer & Computational Linguist  
**Execution Timestamp**: November 17, 2026  
**Audit Scope**: Complete Forensics of All Physical Audio, Transcripts, Manifests, Shards, and Historical Training Models  

---

## 1. Overview & Directory Contents

This directory contains the machine-readable data files and documentation resulting from the comprehensive forensic audit of all Kashmiri speech and text resources across the EduCasheer project and the WAAZ Community Data Collection platform.

### Canonical Machine-Readable Deliverables
- [`audio_text_pairs.parquet`](file:///home/musa/Kashmiri%20Project/docs/data/full_audio_text_inventory/audio_text_pairs.parquet): Primary machine-readable registry containing **174,766 audio records** with exact physical durations, cryptographic hashes, linguistic tags, alignment tiers (A–I), benchmark collision flags, and training memberships.
- [`MASTER_AUDIO_TEXT_INVENTORY.csv`](file:///home/musa/Kashmiri%20Project/docs/data/full_audio_text_inventory/MASTER_AUDIO_TEXT_INVENTORY.csv): Stratified summary cross-tabulating physical location, dataset source, alignment category, and eligibility.
- [`source_registry.csv`](file:///home/musa/Kashmiri%20Project/docs/data/full_audio_text_inventory/source_registry.csv): Granular catalog of all 16 recognized Kashmiri speech and text corpora.
- [`source_summary.csv`](file:///home/musa/Kashmiri%20Project/docs/data/full_audio_text_inventory/source_summary.csv): Source-by-source metrics for clips, hours, transcripts, E22-01 usage, and benchmark exclusions.
- [`source_overlap_matrix.csv`](file:///home/musa/Kashmiri%20Project/docs/data/full_audio_text_inventory/source_overlap_matrix.csv): Cross-corpus overlap matrix against E22-01, E22-03, and test/val evaluation benchmarks.
- [`training_usage_matrix.csv`](file:///home/musa/Kashmiri%20Project/docs/data/full_audio_text_inventory/training_usage_matrix.csv): Historical acoustic training run usage for B-01, B-02A, B-02B, B-02C, E22-01, E22-02, E22-03, and Master 214h.
- [`new_data_beyond_e22_01.csv`](file:///home/musa/Kashmiri%20Project/docs/data/full_audio_text_inventory/new_data_beyond_e22_01.csv): Exhaustive inventory of every speech utterance available beyond the current champion model (E22-01).

### Forensic Audit & Scientific Reports
1. [`MASTER_AUDIO_TEXT_INVENTORY.md`](file:///home/musa/Kashmiri%20Project/docs/data/full_audio_text_inventory/MASTER_AUDIO_TEXT_INVENTORY.md): Comprehensive executive audit report answering the 9 central research questions with empirical proof.
2. [`waaz_collection_audit.md`](file:///home/musa/Kashmiri%20Project/docs/data/full_audio_text_inventory/waaz_collection_audit.md): Deep-dive audit of the WAAZ crowdsourced community platform (`datacollect.db`).
3. [`recoverable_data_pools.md`](file:///home/musa/Kashmiri%20Project/docs/data/full_audio_text_inventory/recoverable_data_pools.md): Detailed roadmap and yield estimates for recovering 107.48 hours of hidden, unaligned, or unsegmented speech.
4. [`duplicate_audio_report.md`](file:///home/musa/Kashmiri%20Project/docs/data/full_audio_text_inventory/duplicate_audio_report.md): 5-level deduplication audit (file paths, PCM hashes, republishing aliases, subsegment overlaps).
5. [`benchmark_contamination_report.md`](file:///home/musa/Kashmiri%20Project/docs/data/full_audio_text_inventory/benchmark_contamination_report.md): Benchmark isolation audit protecting frozen test sets from data leakage.
6. [`transcript_alignment_audit.md`](file:///home/musa/Kashmiri%20Project/docs/data/full_audio_text_inventory/transcript_alignment_audit.md): Multi-tier classification (Categories A through I) of audiotext relationships.
7. [`linguistic_validation_report.md`](file:///home/musa/Kashmiri%20Project/docs/data/full_audio_text_inventory/linguistic_validation_report.md): Perso-Arabic Nastaliq script verification, Kashmiri diacritic audit, Arabic recitation isolation.
8. [`data_quality_summary.md`](file:///home/musa/Kashmiri%20Project/docs/data/full_audio_text_inventory/data_quality_summary.md): Signal-to-noise ratio (SNR), clipping, silence ratio, and acoustic environment evaluation.
9. [`dataset_opportunity_ranking.md`](file:///home/musa/Kashmiri%20Project/docs/data/full_audio_text_inventory/dataset_opportunity_ranking.md): Scientific ranking of all prospective data sources for conversational Kashmiri ASR fine-tuning.
10. [`inventory_methodology.md`](file:///home/musa/Kashmiri%20Project/docs/data/full_audio_text_inventory/inventory_methodology.md): Reproducible execution commands, streaming metadata scripts, and toolchain versions.
11. [`unresolved_inventory_questions.md`](file:///home/musa/Kashmiri%20Project/docs/data/full_audio_text_inventory/unresolved_inventory_questions.md): Open forensic questions, gated external corpora, and human adjudication requirements.

---

## 2. Executive Dashboard (At-A-Glance)

```text
+-----------------------------------------------------------------------------------------+
|                              KASHMIRI SPEECH INVENTORY AUDIT                             |
+-------------------------------------------------------------+---------------------------+
| Metric                                                      | Quantified Finding        |
+-------------------------------------------------------------+---------------------------+
| Total Physical Audio Discovered (on local storage)          | 174,766 files / 369.06 h  |
| Total Audio with Candidate Transcripts                      | 133,535 files / 245.46 h  |
| Verified Utterance-Level Kashmiri Pairs (Category A)        | 103,958 files / 207.21 h  |
| Already Trained in Current Best Model (E22-01)              |  97,165 files / 166.39 h  |
| Verified Pairs Not Used in E22-01 (All Sources)             |  27,176 files /  65.75 h  |
| Genuinely Leak-Free New Training Supervision Beyond E22-01  |   7,824 files /  22.07 h  |
| Potential Recoverable Speech (Unaligned / Unsegmented)      |  34,883 files / 107.48 h  |
| Contaminated / Benchmark-Colliding Speech (Quarantined)     |  10,215 files /  24.93 h  |
| Unusable Speech (Arabic Recitation / Boundary Leaks)        |     596 files /   0.69 h  |
| WAAZ Crowdsourced Community Speech (Recorded in DB)         |   1,453 files /   3.99 h  |
| WAAZ Community Speech Remaining Unused                      |     862 files /   2.63 h  |
+-------------------------------------------------------------+---------------------------+
```
