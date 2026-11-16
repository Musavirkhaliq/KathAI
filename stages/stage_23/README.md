# Acoustic-LM Interaction & Search Space Topology

- **Stage Index**: 23
- **Verified Chronology**: 2026-11-15 to 2026-11-15
- **Status**: `COMPLETED`
- **Evidence Confidence**: `VERIFIED`

## 1. Objective & Context
Analyze loss surface and decoding entropy between acoustic logits and KenLM scores.

## 2. Key Code References
- [`kashmiri_dataset/experiments/stage23_acoustic_lm_interaction/`](file:///home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage23_acoustic_lm_interaction/)

## 3. Associated Datasets
- `dev_a_disjoint.jsonl (6,257 utts)`

## 4. Checkpoints & Models
- `E22-01`

## 5. Verified Metrics & Empirical Outcomes
```json
{
  "entropy_reduction": 0.35,
  "acoustic_dominance_ratio": 0.72
}
```

## 6. Scientific Conclusions
Acoustic posterior confidence dominates over LM in low-resource tokens.

## 7. Failures, Bugs & Identified Limitations
Pure language model rescoring cannot recover phonemes missed by acoustic frontend.

## 8. Complete Forensic Audit Cross-Reference
For exhaustive line-by-line code audits, training curves, confusion matrices, and audit logs, see [`PROJECT_MASTER_GUIDE.md`](file:///home/musa/Kashmiri Project/PROJECT_MASTER_GUIDE.md) and [`docs/research/STAGE_REGISTRY.md`](file:///home/musa/Kashmiri Project/docs/research/STAGE_REGISTRY.md).
