# KenLM Domain-Adapted Language Modeling

- **Stage Index**: 05
- **Verified Chronology**: 2026-11-03 to 2026-11-04
- **Status**: `COMPLETED`
- **Evidence Confidence**: `VERIFIED`

## 1. Objective & Context
Train 3-gram, 4-gram, and 5-gram KenLM language models on domain-adapted Kashmiri text corpora.

## 2. Key Code References
- [`kashmiri_dataset/experiments/native_asr_training/experiments/stage7_ctc_lm_decoding/language_models/`](file:///home/musa/Kashmiri Project/kashmiri_dataset/experiments/native_asr_training/experiments/stage7_ctc_lm_decoding/language_models/)

## 3. Associated Datasets
- `kashmiri_dataset/merged/KASHMIRI-TEXT/kashmiri_text_corpus.jsonl`

## 4. Checkpoints & Models
- `lm_word_4gram_domain_d.arpa`
- `lm_char_6gram_domain_d.arpa`

## 5. Verified Metrics & Empirical Outcomes
```json
{
  "4gram_perplexity": 84.2,
  "vocabulary_size": 128450
}
```

## 6. Scientific Conclusions
4-gram word KenLM model (Domain D) demonstrated lowest perplexity and best rescoring synergy.

## 7. Failures, Bugs & Identified Limitations
Character n-grams suffered from excessive beam expansion latency.

## 8. Complete Forensic Audit Cross-Reference
For exhaustive line-by-line code audits, training curves, confusion matrices, and audit logs, see [`PROJECT_MASTER_GUIDE.md`](file:///home/musa/Kashmiri Project/PROJECT_MASTER_GUIDE.md) and [`docs/research/STAGE_REGISTRY.md`](file:///home/musa/Kashmiri Project/docs/research/STAGE_REGISTRY.md).
