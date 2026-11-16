# Minimum Word Error Rate (MWER) & Weight Averaging Pilots

- **Stage Index**: 14
- **Verified Chronology**: 2026-11-14 to 2026-11-14
- **Status**: `COMPLETED`
- **Evidence Confidence**: `VERIFIED`

## 1. Objective & Context
Pilot sequence-level Minimum Word Error Rate (MWER) loss and stochastic weight averaging.

## 2. Key Code References
- [`kashmiri_dataset/experiments/stage14_mwer_checkpoint_optimization/`](file:///home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage14_mwer_checkpoint_optimization/)

## 3. Associated Datasets
- `Pilot 10h split`

## 4. Checkpoints & Models
- `MWER-01`
- `WA-01`

## 5. Verified Metrics & Empirical Outcomes
```json
{
  "reported_wer": 21.3,
  "reconciled_wer": 26.1
}
```

## 6. Scientific Conclusions
MWER sequence loss provided small gains but early numerical reports were inflated by evaluation metric differences.

## 7. Failures, Bugs & Identified Limitations
Numerical discrepancy identified: Levenshtein normalization was not harmonized across runs.

## 8. Complete Forensic Audit Cross-Reference
For exhaustive line-by-line code audits, training curves, confusion matrices, and audit logs, see [`PROJECT_MASTER_GUIDE.md`](file:///home/musa/Kashmiri Project/PROJECT_MASTER_GUIDE.md) and [`docs/research/STAGE_REGISTRY.md`](file:///home/musa/Kashmiri Project/docs/research/STAGE_REGISTRY.md).
