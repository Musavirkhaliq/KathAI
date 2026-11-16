# Tokenization Architecture: Subword vs Character CTC

- **Stage Index**: 04
- **Verified Chronology**: 2026-11-02 to 2026-11-03
- **Status**: `COMPLETED`
- **Evidence Confidence**: `VERIFIED`

## 1. Objective & Context
Evaluate character-level CTC vs subword BPE tokenization for morphologically rich Kashmiri Perso-Arabic script.

## 2. Key Code References
- [`kashmiri_dataset/experiments/native_asr_training/`](file:///home/musa/Kashmiri Project/kashmiri_dataset/experiments/native_asr_training/)

## 3. Associated Datasets
- `kashmiri_dataset/manifests/train.jsonl`

## 4. Checkpoints & Models
- *No persistent checkpoints created or checkpoint evaluation only.*

## 5. Verified Metrics & Empirical Outcomes
```json
{
  "vocab_size": 159,
  "tokenizer": "Character-level CTC"
}
```

## 6. Scientific Conclusions
Character-level CTC selected as optimal due to strict alignment and superior performance on out-of-vocabulary words.

## 7. Failures, Bugs & Identified Limitations
Subword BPE suffered from severe fragmentation of Kashmiri morphemes.

## 8. Complete Forensic Audit Cross-Reference
For exhaustive line-by-line code audits, training curves, confusion matrices, and audit logs, see [`PROJECT_MASTER_GUIDE.md`](file:///home/musa/Kashmiri Project/PROJECT_MASTER_GUIDE.md) and [`docs/research/STAGE_REGISTRY.md`](file:///home/musa/Kashmiri Project/docs/research/STAGE_REGISTRY.md).
