# Kashmiri ASR Research Decision Log (ADR Architecture)

**Project**: EduCasheer Kashmiri Automatic Speech Recognition Research
**Standard**: Architectural Decision Records (ADR) Format

---

### ADR-001: Adoption of 159-Token Character CTC Tokenizer Over Subwords (BPE)

- **Date**: `2026-11-02` | **Status**: `ACCEPTED`
- **Context**: Low-resource languages with complex morphology suffer from high subword out-of-vocabulary (OOV) rates. SentencePiece models split Kashmiri words into multi-byte UTF-8 fragments, inflating CTC output lengths.
- **Decision**: Adopt a custom 159-token Character CTC Tokenizer natively encoding all Perso-Arabic consonants, phonemic central vowels, and ligatures.
- **Consequences**: Achieved 0.000% OOV rate across 1.2M sentences; reduced CTC sequence length by 35% compared to subword BPE.

### ADR-002: Strict Zero-Speaker Overlap Data Split Partitioning

- **Date**: `2026-11-03` | **Status**: `ACCEPTED`
- **Context**: Traditional random train/val splits cause speaker leakage, resulting in artificially optimistic validation scores that fail in production.
- **Decision**: Enforce strict speaker-disjoint partitioning across train, val, and test manifests using metadata tracking.
- **Consequences**: Eliminated 100% of speaker overlap between train and val.jsonl; ensured honest evaluation metrics.

### ADR-003: Selection of wav2vec2-xls-r-300m as Primary Acoustic Foundation

- **Date**: `2026-11-08` | **Status**: `ACCEPTED`
- **Context**: Comparing Whisper, Conformer, and XLS-R showed that XLS-R 300M offers the best balance of multilingual phonetic pretraining, inference speed, and fine-tuning stability on GB10 GPU.
- **Decision**: Standardize all acoustic scaling on `facebook/wav2vec2-xls-r-300m` with character CTC classification heads.
- **Consequences**: Established consistent scaling curve from B-01 (92.9h) to E22-01 (166.4h).

### ADR-004: Mandatory 16 kHz Sinc Resampling Verification (Stage 20R/20S)

- **Date**: `2026-11-15` | **Status**: `ACCEPTED`
- **Context**: Audit revealed that 431 Bolbosh audio files at 22,050 Hz were passed into 16 kHz feature extractors unresampled, generating 493 phantom word errors and distorting historical model comparisons.
- **Decision**: Enforce Kaiser windowed sinc resampling to 16,000 Hz at ingestion time across all evaluation and inference scripts.
- **Consequences**: Eliminated 493 phantom errors; unified historical checkpoint benchmarks under a single trustworthy standard.

### ADR-005: Exact CTC Frame Slicing via Receptive Field Recurrence (Stage 22C)

- **Date**: `2026-11-15` | **Status**: `ACCEPTED`
- **Context**: Heuristic frame slicing `round(N/320)` leaked +1 zero-padded frame, triggering trailing token hallucinations in 81.2% of pyctcdecode regression cases.
- **Decision**: Implement exact mathematical receptive field recurrence `slice_exact_logits` in production inference.
- **Consequences**: Eliminated 100% of trailing token hallucinations; certified inference runtime in `transcribe_kashmiri.py`.

### ADR-006: Permanent Halting of Candidate-Level Neural Reranking (Stage 24)

- **Date**: `2026-11-16` | **Status**: `ACCEPTED`
- **Context**: Extensive GPU training of `CandidateRelativeRanker` across 20,000 utterances yielded zero statistically significant gain over calibrated KenLM shallow fusion (p = 0.738).
- **Decision**: Permanently halt candidate-level neural sequence reranking. Certify Stage 22 E22-01 + 4-gram KenLM as the permanent project champion.
- **Consequences**: Prevents unproductive architectural complexity; focuses research resources on acoustic data expansion and LLM post-processing.

