# Scaled MWER Verification Series (E-01..E-04)

- **Stage Index**: 15
- **Verified Chronology**: 2026-11-14 to 2026-11-14
- **Status**: `COMPLETED`
- **Evidence Confidence**: `CORROBORATED`

## 1. Objective & Context
Rigorously verify MWER scaling on 100-hour and 300-hour checkpoints (E-01, E-02, E-03, E-04).

## 2. Key Code References
- [`kashmiri_dataset/experiments/stage15_mwer_verification_and_scaling/`](file:///home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage15_mwer_verification_and_scaling/)

## 3. Associated Datasets
- `100h and 135h splits`

## 4. Checkpoints & Models
- `E-01`
- `E-02`
- `E-03`
- `E-04`

## 5. Verified Metrics & Empirical Outcomes
```json
{
  "e01_wer": 27.2,
  "e03_wer": 25.8
}
```

## 6. Scientific Conclusions
E-03 validated modest real-world gains over pure CTC (+1.4% absolute WER reduction).

## 7. Failures, Bugs & Identified Limitations
High computational overhead for on-the-fly N-best beam generation during training.

## 8. Complete Forensic Audit Cross-Reference
For exhaustive line-by-line code audits, training curves, confusion matrices, and audit logs, see [`PROJECT_MASTER_GUIDE.md`](file:///home/musa/Kashmiri Project/PROJECT_MASTER_GUIDE.md) and [`docs/research/STAGE_REGISTRY.md`](file:///home/musa/Kashmiri Project/docs/research/STAGE_REGISTRY.md).
