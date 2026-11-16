# Candidate Parity Repair, Oracle Ceiling & Calibration

- **Stage Index**: 21
- **Verified Chronology**: 2026-11-15 to 2026-11-15
- **Status**: `COMPLETED`
- **Evidence Confidence**: `CORROBORATED`

## 1. Objective & Context
Repair beam candidate extraction pruning defect and determine true oracle ceiling of KenLM search space.

## 2. Key Code References
- [`kashmiri_dataset/experiments/stage21a_post_b02c_reconstruction/`](file:///home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage21a_post_b02c_reconstruction/)
- [`kashmiri_dataset/experiments/stage21b_candidate_relative_ranking/`](file:///home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage21b_candidate_relative_ranking/)

## 3. Associated Datasets
- `dev_calibration.jsonl`
- `dev_assessment.jsonl`

## 4. Checkpoints & Models
- `Pilot CandidateRelativeRanker (312k params)`

## 5. Verified Metrics & Empirical Outcomes
```json
{
  "oracle_wer_ceiling": 16.2,
  "calibrated_temperature": 1.08
}
```

## 6. Scientific Conclusions
Oracle WER of 16.20% proves acoustic search space contains high-quality hypotheses; calibrated T=1.08.

## 7. Failures, Bugs & Identified Limitations
Pilot neural ranker showed risk of overfitting on small N-best lists.

## 8. Complete Forensic Audit Cross-Reference
For exhaustive line-by-line code audits, training curves, confusion matrices, and audit logs, see [`PROJECT_MASTER_GUIDE.md`](file:///home/musa/Kashmiri Project/PROJECT_MASTER_GUIDE.md) and [`docs/research/STAGE_REGISTRY.md`](file:///home/musa/Kashmiri Project/docs/research/STAGE_REGISTRY.md).
