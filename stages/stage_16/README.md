# Advanced Sequence & Decoder Merging

- **Stage Index**: 16
- **Verified Chronology**: 2026-11-14 to 2026-11-15
- **Status**: `COMPLETED`
- **Evidence Confidence**: `VERIFIED`

## 1. Objective & Context
Evaluate advanced sequence training and model weight soup merging (S16-01 to S16-05).

## 2. Key Code References
- [`kashmiri_dataset/experiments/stage16_advanced_sequence_loss/`](file:///home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage16_advanced_sequence_loss/)

## 3. Associated Datasets
- `135h speech`

## 4. Checkpoints & Models
- `S16-01`
- `S16-02`
- `S16-05`

## 5. Verified Metrics & Empirical Outcomes
```json
{
  "s16_05_wer": 25.4
}
```

## 6. Scientific Conclusions
Weight soup merging yielded stable inference but marginal gains over best individual checkpoint.

## 7. Failures, Bugs & Identified Limitations
Interpolation between checkpoints with divergent loss landscapes created slight phonetic blurriness.

## 8. Complete Forensic Audit Cross-Reference
For exhaustive line-by-line code audits, training curves, confusion matrices, and audit logs, see [`PROJECT_MASTER_GUIDE.md`](file:///home/musa/Kashmiri Project/PROJECT_MASTER_GUIDE.md) and [`docs/research/STAGE_REGISTRY.md`](file:///home/musa/Kashmiri Project/docs/research/STAGE_REGISTRY.md).
