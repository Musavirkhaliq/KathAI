# EduCasheer Kashmiri ASR: Master Capstone Project Specification

## 1. Project Mission & Objective
The Capstone Project is the culminating engineering and research synthesis of the course. Working individually or in pairs, students will design, train, evaluate, and defend a complete end-to-end Kashmiri Automatic Speech Recognition system that satisfies certified reproducibility standards.

---

## 2. Mandatory System Deliverables
Every capstone submission must deliver five integrated components:

### Component 1: Data Curation & Normalization Engine
- Ingest raw audio files and manifests.
- Apply automated audio quality filtering (duration boundaries, clipping check, energy SNR estimation).
- Enforce Certified Normalizer v3 compliance (Unicode NFC, homoglyph neutralization, non-printing character stripping).
- Partition the dataset into strict zero-speaker disjoint splits (Train, Validation, Test).

### Component 2: Acoustic Model Training Harness
- Fine-tune a self-supervised foundation backbone (XLS-R-300M or certified equivalent) using PyTorch.
- Implement linear warmup (first 10% steps) and cosine learning rate decay.
- Utilize Automatic Mixed Precision (AMP FP16) and gradient accumulation (effective batch size $\ge 32$).
- Regularize with SpecAugment (time and feature masking).

### Component 3: Calibrated Decoding & Language Model Fusion
- Build or integrate a 5-gram Kashmiri KenLM language model.
- Implement CTC Prefix Beam Search with acoustic temperature calibration ($T pprox 1.08$).
- Conduct a 2D grid search over $(lpha, eta)$ to optimize validation WER.

### Component 4: Multi-Domain Scientific Benchmarking
- Evaluate the certified system across multiple test domains (Clean Studio, Noisy Community, Spontaneous Speech).
- Report micro-averaged Word Error Rate (WER) and Character Error Rate (CER).
- Compute $95\%$ Non-Parametric Bootstrap Confidence Intervals.
- Conduct a Paired Wilcoxon Signed-Rank Test against baseline Model B-01.

### Component 5: Production Streaming Inference Chunker
- Implement Method C2 Overlap-Add chunking for audio streams $> 30	ext{ seconds}$.
- Verify that chunk boundaries snap to integer multiples of $R=320$ samples.
- Demonstrate seamless transcription without boundary word mutilation.

---

## 3. Defense and Scientific Report
Students will submit a 10-page scientific defense report formatted according to Interspeech/ACL templates addressing:
1. Acoustic architecture and hyperparameter choices.
2. Dialectal performance variations and failure analysis.
3. An empirical investigation of a historical dilemma (e.g. resampling aliasing, speaker leakage, or sequence loss).
