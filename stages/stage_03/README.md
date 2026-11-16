# Zero-Speaker Overlap Train/Val/Test Partitioning

- **Stage Index**: 03
- **Verified Chronology**: 2026-11-02 to 2026-11-02
- **Status**: `COMPLETED`
- **Evidence Confidence**: `VERIFIED`

## 1. Objective & Context
Partition speech corpus into disjoint train, validation, and test splits with strict speaker isolation.

## 2. Key Code References
- [`kashmiri_dataset/splits/`](file:///home/musa/Kashmiri Project/kashmiri_dataset/splits/)
- [`kashmiri_dataset/manifests/`](file:///home/musa/Kashmiri Project/kashmiri_dataset/manifests/)

## 3. Associated Datasets
- `kashmiri_dataset/manifests/train.jsonl`
- `kashmiri_dataset/manifests/val.jsonl`
- `kashmiri_dataset/manifests/test.jsonl`

## 4. Checkpoints & Models
- *No persistent checkpoints created or checkpoint evaluation only.*

## 5. Verified Metrics & Empirical Outcomes
```json
{
  "train_utts": 7307,
  "val_utts": 945,
  "test_utts": 928
}
```

## 6. Scientific Conclusions
Zero-speaker overlap enforced across all splits to guarantee unbiased evaluation.

## 7. Failures, Bugs & Identified Limitations
Initial splits covered only 9,180 samples before full 106k expansion.

## 8. Complete Forensic Audit Cross-Reference
For exhaustive line-by-line code audits, training curves, confusion matrices, and audit logs, see [`PROJECT_MASTER_GUIDE.md`](file:///home/musa/Kashmiri Project/PROJECT_MASTER_GUIDE.md) and [`docs/research/STAGE_REGISTRY.md`](file:///home/musa/Kashmiri Project/docs/research/STAGE_REGISTRY.md).
