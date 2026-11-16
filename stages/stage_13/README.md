# Architectural Exploration: Unfreezing & Auxiliary Losses

- **Stage Index**: 13
- **Verified Chronology**: 2026-11-14 to 2026-11-14
- **Status**: `COMPLETED (NEGATIVE RESULT)`
- **Evidence Confidence**: `VERIFIED`

## 1. Objective & Context
Explore CNN feature extractor unfreezing (U-01..U-03) and auxiliary consistency regularization (CR-01).

## 2. Key Code References
- [`kashmiri_dataset/experiments/stage13_architectural_exploration/`](file:///home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage13_architectural_exploration/)

## 3. Associated Datasets
- `135.8h speech`

## 4. Checkpoints & Models
- `U-01`
- `U-02`
- `U-03`
- `CR-01`

## 5. Verified Metrics & Empirical Outcomes
```json
{
  "u01_wer": 32.4,
  "u02_wer": 35.1
}
```

## 6. Scientific Conclusions
Unfreezing the 7-layer CNN encoder caused catastrophic representational drift.

## 7. Failures, Bugs & Identified Limitations
Negative result documented: CNN feature encoder must remain frozen during low-resource fine-tuning.

## 8. Complete Forensic Audit Cross-Reference
For exhaustive line-by-line code audits, training curves, confusion matrices, and audit logs, see [`PROJECT_MASTER_GUIDE.md`](file:///home/musa/Kashmiri Project/PROJECT_MASTER_GUIDE.md) and [`docs/research/STAGE_REGISTRY.md`](file:///home/musa/Kashmiri Project/docs/research/STAGE_REGISTRY.md).
