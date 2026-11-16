# Foundation Acoustic Model B-01 (92.9 Hours)

- **Stage Index**: 08
- **Verified Chronology**: 2026-11-07 to 2026-11-08
- **Status**: `COMPLETED`
- **Evidence Confidence**: `VERIFIED`

## 1. Objective & Context
Train baseline foundation acoustic model B-01 using Wav2Vec2-XLS-R-300M on 92.9 hours.

## 2. Key Code References
- [`kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc/`](file:///home/musa/Kashmiri Project/kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc/)

## 3. Associated Datasets
- `92.9h subset of consolidated speech`

## 4. Checkpoints & Models
- `B-01-xlsr300m-character-ctc/best_model`

## 5. Verified Metrics & Empirical Outcomes
```json
{
  "dev_wer_greedy": 31.85,
  "dev_wer_kenlm": 28.4
}
```

## 6. Scientific Conclusions
B-01 established the first competitive native Kashmiri acoustic foundation.

## 7. Failures, Bugs & Identified Limitations
Slight degradation on low-resource conversational dialects.

## 8. Complete Forensic Audit Cross-Reference
For exhaustive line-by-line code audits, training curves, confusion matrices, and audit logs, see [`PROJECT_MASTER_GUIDE.md`](file:///home/musa/Kashmiri Project/PROJECT_MASTER_GUIDE.md) and [`docs/research/STAGE_REGISTRY.md`](file:///home/musa/Kashmiri Project/docs/research/STAGE_REGISTRY.md).
