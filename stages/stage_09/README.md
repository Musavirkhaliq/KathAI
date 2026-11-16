# Acoustic Scaling Campaign: B-02A, B-02B, B-02C

- **Stage Index**: 09
- **Verified Chronology**: 2026-11-09 to 2026-11-12
- **Status**: `COMPLETED`
- **Evidence Confidence**: `VERIFIED`

## 1. Objective & Context
Scale acoustic pretraining: B-02A (135.8h), B-02B (+specaug/curriculum), B-02C (optimal learning schedule).

## 2. Key Code References
- [`kashmiri_dataset/experiments/native_asr_training/experiments/stage9_b02_acoustic_comparison/`](file:///home/musa/Kashmiri Project/kashmiri_dataset/experiments/native_asr_training/experiments/stage9_b02_acoustic_comparison/)

## 3. Associated Datasets
- `135.8 hours multi-domain speech`

## 4. Checkpoints & Models
- `B-02A/checkpoints/best_model`
- `B-02B/checkpoints/best_model`
- `B-02C/checkpoints/best_model`

## 5. Verified Metrics & Empirical Outcomes
```json
{
  "b02a_wer": 28.12,
  "b02b_wer": 26.5,
  "b02c_wer": 24.47
}
```

## 6. Scientific Conclusions
B-02C achieved landmark 24.47% WER on the legacy evaluation split.

## 7. Failures, Bugs & Identified Limitations
Evaluation split suffered from minor contamination and 22.05 kHz resampling bug discovered later in Stage 20R.

## 8. Complete Forensic Audit Cross-Reference
For exhaustive line-by-line code audits, training curves, confusion matrices, and audit logs, see [`PROJECT_MASTER_GUIDE.md`](file:///home/musa/Kashmiri Project/PROJECT_MASTER_GUIDE.md) and [`docs/research/STAGE_REGISTRY.md`](file:///home/musa/Kashmiri Project/docs/research/STAGE_REGISTRY.md).
