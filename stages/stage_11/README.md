# Diagnostic Benchmark Isolation & Robustness Pilots

- **Stage Index**: 11
- **Verified Chronology**: 2026-11-14 to 2026-11-14
- **Status**: `COMPLETED (DEFECT RECOVERED)`
- **Evidence Confidence**: `CORROBORATED`

## 1. Objective & Context
Create isolated evaluation benchmark partitions: Dev Assessment, Dev Calibration, and Degraded splits.

## 2. Key Code References
- [`kashmiri_dataset/experiments/stage11_robustness_optimization/`](file:///home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage11_robustness_optimization/)

## 3. Associated Datasets
- `dev_assessment.jsonl`
- `dev_calibration.jsonl`

## 4. Checkpoints & Models
- *No persistent checkpoints created or checkpoint evaluation only.*

## 5. Verified Metrics & Empirical Outcomes
```json
{
  "dev_assessment_utts": 1001,
  "dev_calibration_utts": 1001
}
```

## 6. Scientific Conclusions
Isolated evaluation subsets created for fair decoder calibration and error profiling.

## 7. Failures, Bugs & Identified Limitations
22.05 kHz resampling defect was accidentally introduced in Bolbosh audio loader during this stage.

## 8. Complete Forensic Audit Cross-Reference
For exhaustive line-by-line code audits, training curves, confusion matrices, and audit logs, see [`PROJECT_MASTER_GUIDE.md`](file:///home/musa/Kashmiri Project/PROJECT_MASTER_GUIDE.md) and [`docs/research/STAGE_REGISTRY.md`](file:///home/musa/Kashmiri Project/docs/research/STAGE_REGISTRY.md).
