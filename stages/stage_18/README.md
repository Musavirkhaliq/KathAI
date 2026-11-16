# Kashmiri ASR Error Atlas, Decoder Tuning & MBR

- **Stage Index**: 18
- **Verified Chronology**: 2026-11-15 to 2026-11-15
- **Status**: `COMPLETED`
- **Evidence Confidence**: `VERIFIED`

## 1. Objective & Context
Systematic phonetic confusion matrix profiling, Minimum Bayes Risk decoding, and adaptive rescoring.

## 2. Key Code References
- [`kashmiri_dataset/experiments/stage18a_kashmiri_asr_error_atlas/`](file:///home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage18a_kashmiri_asr_error_atlas/)

## 3. Associated Datasets
- `dev_assessment.jsonl`

## 4. Checkpoints & Models
- `MBR decoder configuration`

## 5. Verified Metrics & Empirical Outcomes
```json
{
  "top_confusion": "Kaf / Keheh and central vowels (\u0672 vs \u0655)"
}
```

## 6. Scientific Conclusions
Identified primary error sources: central vowel neutralization and dialectal affricates.

## 7. Failures, Bugs & Identified Limitations
MBR decoding latency too high for real-time streaming.

## 8. Complete Forensic Audit Cross-Reference
For exhaustive line-by-line code audits, training curves, confusion matrices, and audit logs, see [`PROJECT_MASTER_GUIDE.md`](file:///home/musa/Kashmiri Project/PROJECT_MASTER_GUIDE.md) and [`docs/research/STAGE_REGISTRY.md`](file:///home/musa/Kashmiri Project/docs/research/STAGE_REGISTRY.md).
