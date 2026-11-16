# Kashmiri ASR Known Issues, Anomalies & Troubleshooting Guide

**Project**: EduCasheer Kashmiri Automatic Speech Recognition Research
**Last Audited**: November 16, 2026

---

## 1. Forensic Anomaly Register

| Anomaly ID | Severity | Affected Stage | Description | Resolution Status |
|:---:|:---:|:---:|:---|:---|
| **ANOM-01** | `CRITICAL` | Stages 11-19 | 22.05 kHz Bolbosh audio ingested unresampled into 16 kHz feature extractor, inflating historical WER by ~3%. | **RESOLVED**: Fixed in Stage 20S via Kaiser windowed sinc resampling. |
| **ANOM-02** | `HIGH` | Stage 20 | CARE-ASR Step 0 serialization failure: stopping criteria never met; in-memory model evaluated but no checkpoint saved. | **RESOLVED**: Stage 20 invalidated; documented in Stage 20R audit. |
| **ANOM-03** | `MEDIUM` | Stage 14 | Cosine similarity > 1.0 due to BF16 float round-off error violating Cauchy-Schwarz bound. | **RESOLVED**: Stage 15 introduced FP64 double-precision accumulator. |
| **ANOM-04** | `MEDIUM` | Stage 20T | N-best candidate extraction with beam width 8 introduced an 84-word search pruning penalty. | **RESOLVED**: Stage 21A restored beam width 16 with history pruning. |
| **ANOM-05** | `LOW` | Inference | Long audio recordings (> 25s) trigger PyTorch attention memory OOM crashes. | **RESOLVED**: Stage 22C certified Method C2 context-preserving chunking (peak VRAM <= 1.7GB). |

---

## 2. Practical Troubleshooting Instructions

### Issue 1: Audio Resampling Warnings or High Word Error Rate
- **Symptom**: Transcripts contain severe substitutions or garbled words on external recordings.
- **Cause**: Input audio sample rate is not 16,000 Hz, or is stereo.
- **Remedy**: Always verify audio sample rate with `sox --i file.wav` or `sf.info()`. `transcribe_kashmiri.py` automatically converts audio to 16kHz mono using torchaudio resamplers.

### Issue 2: CUDA Out-of-Memory on Long Audio Files
- **Symptom**: `RuntimeError: CUDA out of memory` when transcribing files longer than 30 seconds.
- **Cause**: Wav2Vec2 self-attention scales quadratically with sequence length ($O(T^2)$).
- **Remedy**: Use the certified Stage 22C streaming chunker (`evaluate_dev_chunking_strategies.py` or chunking flag), which slices audio into 18s chunks with 2s overlap context, keeping memory strictly under 1.7 GB.

### Issue 3: Trailing Token Hallucinations in pyctcdecode
- **Symptom**: Transcriptions end with repeated spurious characters or whitespace.
- **Cause**: Passing heuristic `round(N/320)` frame lengths leaks +1 zero-padded frame.
- **Remedy**: Use the exact receptive field recurrence function `slice_exact_logits` from `stage22c_exact_output_lengths.py`.
