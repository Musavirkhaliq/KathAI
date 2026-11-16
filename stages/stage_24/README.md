# Actual Neural Candidate Reranker & Acoustic Continuation

- **Stage Index**: 24
- **Verified Chronology**: 2026-11-15 to 2026-11-16
- **Status**: `COMPLETED (DECISIVE EVALUATION)`
- **Evidence Confidence**: `VERIFIED`

## 1. Objective & Context
Train full neural CandidateRelativeRanker (329k params) and test acoustic continuation fine-tuning.

## 2. Key Code References
- [`kashmiri_dataset/experiments/stage24_actual_reranker_training/`](file:///home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage24_actual_reranker_training/)

## 3. Associated Datasets
- `stage24_dev_a_disjoint_nbest.jsonl`
- `stage24_dev_calibration_nbest.jsonl`

## 4. Checkpoints & Models
- `CandidateRelativeRanker/best_checkpoint`
- `E24-CTC-01`

## 5. Verified Metrics & Empirical Outcomes
```json
{
  "reranker_wer": 24.28,
  "e22_01_baseline_wer": 24.12
}
```

## 6. Scientific Conclusions
Neural reranker provided no statistically significant improvement over calibrated KenLM (24.28% vs 24.12%).

## 7. Failures, Bugs & Identified Limitations
Decisive negative result documented: lightweight neural rerankers cannot overcome acoustic-LM calibrated beam search.

## 8. Complete Forensic Audit Cross-Reference
For exhaustive line-by-line code audits, training curves, confusion matrices, and audit logs, see [`PROJECT_MASTER_GUIDE.md`](file:///home/musa/Kashmiri Project/PROJECT_MASTER_GUIDE.md) and [`docs/research/STAGE_REGISTRY.md`](file:///home/musa/Kashmiri Project/docs/research/STAGE_REGISTRY.md).
