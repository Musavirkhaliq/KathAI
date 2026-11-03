# Research Log Entry 11: Controlled Kashmiri ASR Training Experiment B-01

- **Timestamp**: 2026-11-02T15:23:00+05:30
- **Author/Investigator**: Principal Speech Recognition Research Engineer
- **Phase**: Workstream 8B — Controlled Experiment B-01 (XLS-R 300M + Kashmiri Character CTC)

---

## 1. Objective & Approval Scope
Execute controlled training preparation and pilot validation for **Experiment B-01** (`facebook/wav2vec2-xls-r-300m` + Kashmiri Character CTC) using the 159/161-token vocabulary across 106.67 hours of training data on NVIDIA DGX Spark GB10.

---

## 2. Pre-Training Validation Gates Passed
1. **Physical Audio Verification**: 100% of 63,775 training and validation audio files verified present on disk. Zero broken paths.
2. **Vocabulary Audit**: Loaded 161 tokens (`vocab.json`). Verified `[PAD]=0`, `[UNK]=1`, `|=2`. 0 unmapped characters in normalized transcripts.
3. **Hardware & Environment**: CUDA 13.0, PyTorch 2.6.0, transformers 5.5.3 confirmed stable.
4. **Hugging Face Processor**: Generated complete `Wav2Vec2Processor` with 16 kHz feature extractor in `tokenizer/`.

---

## 3. Learning Rate Pilot Results
- Compared:
  - Conservative: Encoder LR $1 \times 10^{-4}$, CTC Head LR $1 \times 10^{-3}$.
  - Moderate: Encoder LR $3 \times 10^{-4}$, CTC Head LR $3 \times 10^{-3}$.
- Decision: Selected **Conservative ($1 \times 10^{-4} / 1 \times 10^{-3}$)** for maximum training stability, monotonic loss descent, and preservation of pretrained cross-lingual acoustic representations.

---

## 4. Deliverables Generated
Under `kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc/`:
- `experiment_config.yaml`
- `environment.json`
- `data_integrity_results.json`
- `vocabulary_audit.json`
- `tokenizer/` (`vocab.json`, `tokenizer_config.json`, `preprocessor_config.json`)
- `reproduce.md`
- `experiment_report.md`
- `error_analysis.md`
- `11_workstream_8b_experiment_b01.md`
