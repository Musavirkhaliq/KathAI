# Reproducibility Checklist & Verification Audit

This checklist follows the NeurIPS / ICML / ACL Reproducibility Criteria to verify the completeness of empirical claims.

---

## 1. Mathematical and Algorithmic Claims
- [x] **Full Mathematical Formulation**: All loss functions (CTC, MWER, CR-CTC, Inter-CTC) are fully specified with explicit equations (`models/training_objectives.md`).
- [x] **Sample-Rate Derivations**: Mathematical impact of temporal stretching and semitone shifts derived in full (`results/historical_results_reconciliation.csv`).
- [x] **Oracle Formulation**: Explicit formula for candidate-set edit distance minimization provided (`results/oracle_analysis.md`).

---

## 2. Dataset and Corpus Documentation
- [x] **Dataset Cards Provided**: Detailed dataset card constructed with full metadata schema (`data/dataset_card.md`).
- [x] **Source Inventories**: All 12 sub-corpora cataloged with sample counts, hours, and sample rates (`data/source_inventory.csv`).
- [x] **Licensing Audited**: Legal redistribution rights, CC-BY terms, and restrictions explicitly documented (`data/source_licenses.csv`).
- [x] **Contamination Audited**: Strict zero-speaker and zero-sample overlap verified across splits (`data/data_contamination_audit.md`).
- [x] **Limitations Disclosed**: Repeated diagnostic inspection of `dev_assessment.jsonl` transparently acknowledged (`benchmark/benchmark_limitations.md`).

---

## 3. Acoustic Modeling and Training
- [x] **Hyperparameters Explicit**: Complete configuration table with optimizer, learning rate, warmup, batch size, and steps provided (`models/training_configuration_matrix.csv`).
- [x] **Hardware Specified**: Full hardware details for the NVIDIA DGX Spark GB10 documented (`reproducibility/environment.md`).
- [x] **Checkpoints Verified on Disk**: 23 historical checkpoints checked for file existence and format (`reproducibility/reproduce_paper_results.sh --mode audit`).
- [x] **Missing Checkpoints Disclosed**: Serialization failure of Stage 20 CARE-ASR explicitly reported as unrecoverable (`results/results_integrity_report.md`).

---

## 4. Evaluation and Statistical Verification
- [x] **Evaluation Protocol Defined**: Canonical protocol `EduCasheer-Kashmiri-ASR-Eval-v2` documented (`benchmark/evaluation_protocols.md`).
- [x] **Resampler Standardized**: 64-point Kaiser windowed sinc resampling mandated for all non-16k audio (`evidence/code_provenance.md`).
- [x] **Confidence Intervals Reported**: 10,000 paired bootstrap resamples computed for all primary model comparisons (`results/statistical_analysis.md`).
- [x] **Negative Results Included**: Complete documentation of failed augmentation, layer-unfreezing, and pseudo-labeling experiments (`models/advanced_experiments.md`).
- [x] **Traceability Ensured**: Every claim mapped to an underlying evidence artifact (`evidence/claim_to_evidence.csv`).
