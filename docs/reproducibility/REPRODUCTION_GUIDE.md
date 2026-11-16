# Kashmiri ASR Master Reproducibility Guide

**Project**: EduCasheer Kashmiri Automatic Speech Recognition Research  
**Hardware Environment Tested**: NVIDIA DGX Spark (NVIDIA GB10 GPU, PyTorch 2.14.1+cu130, CUDA 13.0)  
**Root Working Directory**: `/home/musa/Kashmiri Project`  
**Last Verified**: November 16, 2026  

---

## 1. Fast Smoke Test (30 Seconds)

To verify the PyTorch GPU runtime, audio decoding, and SOTA model weights:
```bash
cd "/home/musa/Kashmiri Project"
python3 test_batch_transcribe.py
```

Expected output:
```text
Loading Kashmiri ASR model from .../best_model on cuda...
Model loaded successfully!
Sample: xkaab_000221
Ref: فیورِٹ بُک چھ مےٚ قُرآنِ شریٖف تِکیازِ قُرآنِ شریٖف چھُ اَسِہ حضرت محمدصلى الله عليه وسلم سٔندِس پؠٹھ نٲزِل گوٚمُت
Hyp: فیورٹ بک چھ مےٚ قرانۍ شرف تکیاز قرانہ شرف چھ اسہ حضرت محمد صالیہ لہ السلم سنس پٮ ٹھ نٲزل گوٚمت
```

---

## 2. Reproducing Certified SOTA Benchmarks (Stage 22C / 23)

To reproduce the certified benchmarks across all disjoint evaluation partitions on GPU:
```bash
python3 kashmiri_dataset/experiments/stage22c_inference_certification/inference/reproduce_certified_stage22c_benchmarks.py
```

Expected output:
```text
[STAGE 22C BENCHMARK CERTIFICATION]
Evaluating Champion E22-01 on dev_calibration.jsonl (1,001 utts)...
WER (Greedy):  26.82%
WER (KenLM):   24.74% (4,114 errors)
Evaluating Champion E22-01 on dev_a_disjoint.jsonl (6,257 utts)...
WER (KenLM):   24.94% (26,566 errors)
100.0% Exact Mathematical Parity Confirmed.
```

---

## 3. Reproducing the Stage 20R Forensic Resampling Audit

To verify the elimination of 493 phantom word errors caused by the 22.05 kHz resampling bug:
```bash
python3 kashmiri_dataset/experiments/stage20s_audio_integrity_recovery/eval_all_canonical_fixed.py
```

Expected output:
```text
[FORENSIC RESAMPLING AUDIT]
Unresampled Evaluation (Historical Bug): 4,614 errors (27.39% WER)
Kaiser Sinc Resampled (Canonical Standard): 4,121 errors (24.46% WER)
Delta Eliminated: -493 phantom word errors on frozen weights.
Cohort Breakdown: 100% of eliminated errors belong to 431 Bolbosh files at 22,050 Hz.
```

---

## 4. Reproducing Stage 24 Neural Candidate Reranker Evaluation

To verify that the neural candidate reranker does not outperform KenLM shallow fusion:
```bash
python3 kashmiri_dataset/experiments/stage24_actual_reranker_training/evaluate_stage24_reranker.py
```

Expected output:
```text
[STAGE 24 EVALUATION]
Baseline KenLM WER: 24.74% (4,114 errors)
Neural Ranker WER:   24.74% (4,115 errors)
Delta: +1 error (p = 0.738, not statistically significant)
Conclusion: Candidate-level neural sequence reranking does not outperform KenLM.
```

---

## 5. Running Production CLI Transcription

Transcribe any arbitrary speech file (WAV, MP3, FLAC) with automatic resampling and dual script output:
```bash
python3 transcribe_kashmiri.py path/to/audio.wav --normalize
```
