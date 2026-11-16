# External SOTA ASR Landscape Benchmarking

- **Stage Index**: 06
- **Verified Chronology**: 2026-11-04 to 2026-11-06
- **Status**: `COMPLETED`
- **Evidence Confidence**: `VERIFIED`

## 1. Objective & Context
Benchmark pre-existing public Kashmiri ASR models (Whisper, IndicConformer, Aazu34, Bodhan AI).

## 2. Key Code References
- [`kashmiri_dataset/experiments/external_asr_benchmark/`](file:///home/musa/Kashmiri Project/kashmiri_dataset/experiments/external_asr_benchmark/)

## 3. Associated Datasets
- `val_eligible_6728.jsonl`

## 4. Checkpoints & Models
- `Aazu34/kashmiri-asr-ctc`
- `bodhan-ai/indic-transcribe-core`
- `openai/whisper-small`

## 5. Verified Metrics & Empirical Outcomes
```json
{
  "aazu34_wer": 72.4,
  "whisper_small_wer": 84.1,
  "indicconformer_wer": 68.9
}
```

## 6. Scientific Conclusions
Existing public models exhibit severe WER (>68%) on conversational and regional Kashmiri speech.

## 7. Failures, Bugs & Identified Limitations
Token mismatch and lack of diacritic normalization in external models.

## 8. Complete Forensic Audit Cross-Reference
For exhaustive line-by-line code audits, training curves, confusion matrices, and audit logs, see [`PROJECT_MASTER_GUIDE.md`](file:///home/musa/Kashmiri Project/PROJECT_MASTER_GUIDE.md) and [`docs/research/STAGE_REGISTRY.md`](file:///home/musa/Kashmiri Project/docs/research/STAGE_REGISTRY.md).
