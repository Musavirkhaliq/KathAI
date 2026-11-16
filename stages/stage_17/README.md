# Noisy Speech Learning & Pseudo-Labeling

- **Stage Index**: 17
- **Verified Chronology**: 2026-11-15 to 2026-11-15
- **Status**: `COMPLETED (NEGATIVE RESULT)`
- **Evidence Confidence**: `VERIFIED`

## 1. Objective & Context
Iterative self-training with pseudo-labeling on unannotated Kashmiri broadcast audio.

## 2. Key Code References
- [`kashmiri_dataset/experiments/stage17_noisy_speech_learning/`](file:///home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage17_noisy_speech_learning/)

## 3. Associated Datasets
- `Unannotated radio broadcasts (50h)`

## 4. Checkpoints & Models
- `P17-PL`

## 5. Verified Metrics & Empirical Outcomes
```json
{
  "p17_wer": 27.9,
  "baseline_wer": 25.8
}
```

## 6. Scientific Conclusions
Self-training with imperfect pseudo-labels reinforced phonetic hallucinations.

## 7. Failures, Bugs & Identified Limitations
Negative result documented: confidence thresholding without strict acoustic filtering harms CTC convergence.

## 8. Complete Forensic Audit Cross-Reference
For exhaustive line-by-line code audits, training curves, confusion matrices, and audit logs, see [`PROJECT_MASTER_GUIDE.md`](file:///home/musa/Kashmiri Project/PROJECT_MASTER_GUIDE.md) and [`docs/research/STAGE_REGISTRY.md`](file:///home/musa/Kashmiri Project/docs/research/STAGE_REGISTRY.md).
