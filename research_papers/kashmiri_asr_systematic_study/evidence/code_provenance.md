# Code Provenance and Execution Lineage Audit

**Project**: EduCasheer Kashmiri Automatic Speech Recognition Research  
**Hardware Environment**: NVIDIA DGX Spark GB10 (NVIDIA GB10 GPU, PyTorch 2.6.0+cu128 / PyTorch 2.14.1+cu130, CUDA 12.8 / 13.0)  
**Standard Evaluation Protocol**: `EduCasheer-Kashmiri-ASR-Eval-v2` (`eval_v2.yaml`)  
**Audit Date**: November 15, 2026  

---

## 1. Executive Code Lineage Overview

Every primary claim, result, and metric reported in the EduCasheer Kashmiri ASR systematic study is mapped to audited, reproducible software components in the repository. This document establishes the code-level provenance for all stages of data preprocessing, acoustic model training, inference evaluation, language model integration, and statistical significance testing.

```
Raw Audio Sources (WAV, FLAC, MP3, 16k & 22.05k)
   │
   ▼
[Data Preprocessing & Audio Normalization]
   ├── `kashmiri_dataset/audio/canonical_audio.py` (Kaiser windowed sinc resampling to 16,000 Hz)
   ├── `kashmiri_dataset/scripts/normalize_transcripts.py` (Unicode NFKC, Kashmiri diacritic preservation)
   └── `kashmiri_dataset/scripts/pretraining_validation_gates.py` (5-stage quality gating: SNR, duration, CTC feasibility)
   │
   ▼
[Acoustic Model Training & Fine-Tuning]
   ├── `kashmiri_dataset/scripts/train_b01_xlsr.py` (Stage 8 B-01 baseline, 92.9h)
   ├── `kashmiri_dataset/scripts/train_stage9_full_schedule.py` (Stage 9 B-02A/B/C scaling, up to 165.3h)
   ├── `kashmiri_dataset/experiments/stage12_robustness_training/train_stage12_experiment.py` (Stage 12 R-01/02)
   ├── `kashmiri_dataset/experiments/stage13_architectural_optimization/train_stage13_progressive_unfreeze.py` (Stage 13 U-01..03)
   ├── `kashmiri_dataset/experiments/stage14_mwer_checkpoint_optimization/train_mwer_pilot.py` (Stage 14 MWER & WA)
   └── `kashmiri_dataset/experiments/stage15_mwer_verification_and_scaling/train_stage15_matched_continuation.py` (Stage 15 E-01..04)
   │
   ▼
[Language Modeling & Vocabulary Construction]
   ├── `kashmiri_dataset/tokenizers/` (159-token Kashmiri Character CTC Tokenizer, 0.00% OOV)
   └── `kashmiri_dataset/experiments/native_asr_training/experiments/stage7_ctc_lm_decoding/language_models/`
         └── `lm_word_4gram_domain_d.arpa` (KenLM 4-gram domain model, 1.2M sentences)
   │
   ▼
[Decoding, Rescoring & Evaluation Engines]
   ├── `kashmiri_dataset/experiments/stage20s_audio_integrity_recovery/evaluation/stage20s_eval_engine.py` (Canonical 16k eval)
   ├── `kashmiri_dataset/experiments/stage21a_post_b02c_reconstruction/generate_final_report.py` (Stage 21A ranking & parity)
   └── `kashmiri_dataset/experiments/stage21b_candidate_relative_ranking/calibration/run_grid_calibration.py` (Stage 21B ranker)
   │
   ▼
[Statistical Significance & Bootstrap Verification]
   ├── `kashmiri_dataset/scripts/evaluate_test_bootstrap.py` (10,000 paired bootstrap iterations)
   └── `kashmiri_dataset/experiments/stage21b_candidate_relative_ranking/statistics/run_statistical_tests.py`
```

---

## 2. Audio Processing Code Provenance

### 2.1 The Canonical Audio Resampling Pipeline
- **Source Module**: `kashmiri_dataset/audio/canonical_audio.py`
- **Implementation**:
  ```python
  def load_canonical_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
      waveform, orig_sr = torchaudio.load(path)
      # Convert multi-channel to mono
      if waveform.shape[0] > 1:
          waveform = torch.mean(waveform, dim=0, keepdim=True)
      # Fast path: exact sample rate match bypasses resampling
      if orig_sr == target_sr:
          return waveform.squeeze(0)
      # Canonical Kaiser windowed sinc interpolation
      resampler = torchaudio.transforms.Resample(
          orig_freq=orig_sr,
          new_freq=target_sr,
          resampling_method="sinc_interp_kaiser",
          lowpass_filter_width=64,
          rolloff=0.9475937167399596,
          beta=14.769656459379492
      )
      return resampler(waveform).squeeze(0)
  ```
- **Verification**: Tested in `kashmiri_dataset/audio/test_canonical_audio.py`. Confirmed zero signal degradation on native 16,000 Hz waveforms and mathematically exact pitch restoration on 22,050 Hz Gaash Bolbosh recordings.

### 2.2 Historical Resampling Failure vs Corrected Implementation
- **Historical Defect (Stages 15–19 Evaluation)**:
  `soundfile.read(path)` yielded a raw NumPy array that was passed directly to `Wav2Vec2Processor(audio, sampling_rate=16000)`. Because HuggingFace processors only set metadata and do not resample raw NumPy buffers, 22.05 kHz audio was fed to the 16 kHz model unresampled, causing temporal stretching ($1.378\times$) and downward pitch shift ($-5.53$ semitones).
- **Corrected Code (Stage 20S, 21A, 21B)**:
  Unified in `stage20s_eval_engine.py` and `canonical_audio.py`, guaranteeing that all audio reaching `Wav2Vec2ForCTC` is strictly 16,000 Hz PCM.

---

## 3. Acoustic Training Code Lineage

### 3.1 Stage 8 Foundation Baseline (B-01)
- **Primary Script**: `kashmiri_dataset/scripts/train_b01_xlsr.py`
- **Base Architecture**: `facebook/wav2vec2-xls-r-300m`
- **Key Hyperparameters**:
  - Optimizer: AdamW ($\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-8}$)
  - Peak Learning Rate: $3 \times 10^{-4}$ with 2,000 warm-up steps and linear decay
  - Effective Batch Size: 32 (length-bucketed dynamic batching)
  - Precision: BF16 Mixed Precision
  - Epochs: 20 epochs (45,000 steps)
  - Loss: Connectionist Temporal Classification (CTC) over 159 characters

### 3.2 Stage 9 Acoustic Scaling (B-02A, B-02B, B-02C)
- **Primary Script**: `kashmiri_dataset/scripts/train_stage9_full_schedule.py`
- **Resampling Verification (Lines 188–194)**:
  `torchaudio.transforms.Resample(orig_sr, 16000)` was executed dynamically during dataloading, preserving valid acoustic representations in B-02C.
- **Corpus Scaling**:
  - B-02A: 104.93 hours (64,522 utterances)
  - B-02B: 123.45 hours (80,106 utterances)
  - B-02C: 165.32 hours (96,664 utterances)

### 3.3 Stage 14 & 15 Sequence Optimization (MWER and E-Series)
- **Primary Scripts**:
  - `kashmiri_dataset/experiments/stage14_mwer_checkpoint_optimization/train_mwer_pilot.py`
  - `kashmiri_dataset/experiments/stage15_mwer_verification_and_scaling/train_stage15_matched_continuation.py`
- **Objective Function**:
  $$\mathcal{L} = (1 - \lambda) \mathcal{L}_{\text{CTC}} + \lambda \mathcal{L}_{\text{MWER}}$$
  where $\lambda \in \{0.00, 0.05, 0.10, 0.20\}$ across arms E-01, E-02, E-03, E-04.
- **Audit Outcome**: Training was executed for 100 steps from B-02C initialization. E-03 achieved 4,121 errors vs 4,122 errors for B-02C on canonical 16 kHz evaluation ($p=0.963$).

---

## 4. Decoder & Language Model Code Provenance

### 4.1 CTC Beam Search Decoder
- **Library**: `pyctcdecode` (v0.5.0+) with KenLM C++ binding
- **Decoder Configuration Contract**:
  - `beam_width`: 16 (crucial: beam width 8 caused an 84-error search pruning penalty in Stage 20T)
  - `prune_history`: `True`
  - `alpha`: Language model weight ($\alpha \in [0.20, 0.35]$)
  - `beta`: Word insertion bonus ($\beta \in [-0.50, 0.00]$)
  - `unk_score_offset`: $-10.0$
- **Language Model Artifact**:
  - Path: `kashmiri_dataset/experiments/native_asr_training/experiments/stage7_ctc_lm_decoding/language_models/lm_word_4gram_domain_d.arpa`
  - Corpus: 1.2M Kashmiri sentences compiled from Wikipedia, Aya, BPCC, and Mirwaiz texts.

### 4.2 Candidate-Relative Sequence Ranker (Stage 21B)
- **Primary Script**: `kashmiri_dataset/experiments/stage21b_candidate_relative_ranking/training/train_candidate_relative_ranker.py`
- **Model Architecture**:
  - 312,705 trainable parameters.
  - Zero-initialized linear projection head guaranteeing exact step-zero parity with the baseline.
  - Bounded tanh score fusion: $S = S_{\text{base}} + \lambda \tanh(f_\theta(\Delta x))$.
- **Evaluation**: `kashmiri_dataset/experiments/stage21b_candidate_relative_ranking/evaluation/evaluate_ranker.py`.

---

## 5. Statistical Significance and Audit Verification Scripts

| Script Path | Purpose | Execution Characteristics |
|:---|:---|:---|
| `kashmiri_dataset/scripts/evaluate_test_bootstrap.py` | 10,000 paired bootstrap resamples | Non-parametric confidence intervals for $\Delta$WER |
| `kashmiri_dataset/experiments/stage21a_post_b02c_reconstruction/generate_final_report.py` | Audited checkpoint ranking | Evaluates all 30+ checkpoints on identical manifest |
| `kashmiri_dataset/experiments/stage21b_candidate_relative_ranking/statistics/run_statistical_tests.py` | Statistical significance battery | Paired t-test, Wilcoxon signed-rank, bootstrap p-values |
| `kashmiri_dataset/audio/test_canonical_audio.py` | Resampler unit test | Validates 16 kHz identity and 22.05 kHz Kaiser interpolation |
