# Kashmiri ASR Central Experiment Registry

**Project**: EduCasheer Kashmiri Automatic Speech Recognition Research  
**Author**: Principal AI Research Engineer & ML Infrastructure Architect  
**Hardware Environment**: NVIDIA DGX Spark (NVIDIA GB10 GPU, 128GB Unified Memory)  
**Last Updated**: November 14, 2026  

---

## 1. Executive Summary of Best Available Models

| Model Rank | Experiment ID | Architecture & Parameters | Training Corpus | Best Checkpoint Path | Dev Greedy Norm WER | Dev +KenLM Norm WER | Recommended Use Case |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| 🥇 **SOTA Acoustic** | **B-02C** | XLS-R 300M (315.6M params) + Char CTC (vocab: 159) | 165.32 hrs (96,664 utts) | `kashmiri_dataset/experiments/native_asr_training/experiments/stage9_b02_acoustic_comparison/B-02C/checkpoints/best_model` | **27.84%** | **25.24%** | **Primary production engine.** Highest overall accuracy across secular, conversational, and Qur'anic speech. |
| 🥈 **Sequence Fine-Tuned** | **E-03 (Stage 15)** | XLS-R 300M + MWER Sequence Loss ($\lambda=0.10$) | B-02C Initialization + MWER Calibration | `kashmiri_dataset/experiments/stage15_mwer_verification_and_scaling/checkpoints/E-03/best_calib` | **30.42%** | **21.68% (tolerant)** | Best for N-best re-ranking and word-level alignment where minimum edit cost is strictly targeted. |
| 🥉 **Religious Specialized** | **B-02B** | XLS-R 300M + Char CTC | 123.45 hrs (80,106 utts) | `kashmiri_dataset/experiments/native_asr_training/experiments/stage9_b02_acoustic_comparison/B-02B/checkpoints/best_model` | **29.39%** | **26.41%** | Specialized high-recall model for Islamic terminology and Arabic-Kashmiri religious code-mixing. |
| 🎖️ **Proven Baseline** | **B-02A** | XLS-R 300M + Char CTC | 104.93 hrs (64,522 utts) | `kashmiri_dataset/experiments/native_asr_training/experiments/stage9_b02_acoustic_comparison/B-02A/checkpoints/best_model` | **29.42%** | **26.30%** | Proven stable foundation with comprehensive heldout test verification (`b02a_followup_test_report.md`). |

---

## 2. Chronological Research Experiments

### 2.1 External & Zero-Shot Baselines (`asr_baselines`, `external_asr_benchmark`)
Evaluated across standardized held-out benchmarks:

| Model ID | Provider / Architecture | Parameters | Norm WER (%) | Norm CER (%) | Evaluation Notes |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Whisper-Large-v3** | OpenAI / Encoder-Decoder Transformer | 1,550M | 84.12% | 42.18% | Severe catastrophic hallucination; low zero-shot Kashmiri capacity. |
| **Whisper-Medium** | OpenAI / Encoder-Decoder Transformer | 769M | 88.65% | 46.50% | Unusable without native Kashmiri supervised fine-tuning. |
| **IndicConformer CTC** | AI4Bharat / Conformer CTC | 115M | 62.14% | 28.32% | Off-the-shelf Hindi/Urdu transfer; struggles with Kashmiri vowels. |
| **Khushi1234455687** | Wav2Vec2 Large XLS-R 53 Kashmiri v6 | 315M | 52.40% | 23.10% | Early public community fine-tune; incomplete character vocabulary. |
| **Muzaffar786** | Wav2Vec2 Kashmiri Jhon Data One | 315M | 49.80% | 21.60% | Limited domain diversity. |

---

### 2.2 Stage 8 & 9: Native Controlled Acoustic Scaling (`stage9_b02_acoustic_comparison`)
Landmark study investigating the scaling laws of Kashmiri speech data by fixing architecture (`facebook/wav2vec2-xls-r-300m`), optimizer (AdamW, LR $3\times 10^{-4}$ with linear warmup and decay), dynamic length batching (batch size 32), precision (BF16 AMP), and evaluating on the identical 6,728 validation set (104,685 words):

#### Experiment Specifications:
- **Base Pretrained Model**: `facebook/wav2vec2-xls-r-300m` (Local checkpoint: `kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc/checkpoints/base_xlsr`)
- **Tokenizer**: Kashmiri Character CTC (159 tokens including whitespace and blank). Local path: `kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc/tokenizer`
- **Validation Dataset**: `kashmiri_dataset/experiments/native_asr_training/data_splits/val.jsonl` (6,728 utterances, zero speaker overlap)
- **LM Shallow Fusion**: KenLM 4-gram domain-adapted language model trained on 1.2M Kashmiri sentences (`stage7_ctc_lm_decoding/language_models/lm_word_4gram_domain_d.arpa`) with beam width $W=16$, $\alpha=0.25$, $\beta=0.00$.

#### Landmark Empirical Results:

| Model ID | Corpus Composition | Training Hours | Utterance Count | Greedy Norm WER | Greedy Norm CER | Beam (No LM) WER | Beam + KenLM WER | Relative WERR vs Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B-01** | Initial verified baseline | 92.94h | 53,460 | 33.86% | 13.08% | 33.50% | 31.80% | Reference |
| **B-02A** | B-01 + Recovered speech | 104.93h | 64,522 | 29.42% | 11.66% | 28.92% | 26.30% | **-13.11%** |
| **B-02B** | B-02A + Qur'anic speech | 123.45h | 80,106 | 29.39% | 11.61% | 28.94% | 26.41% | **-13.20%** |
| **B-02C** | B-02B + IndicVoices-R | 165.32h | 96,664 | **27.84%** | **10.96%** | **27.48%** | **25.24%** | **-17.78%** |

#### Source-Wise Breakdown (Greedy Norm WER %):

| Corpus Source | Sample Count | B-01 WER | B-02A WER | B-02B WER | B-02C WER | B-02C CER | Net Gain vs B-01 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GAASH-Lab / Bolbosh** | 2,830 | 31.10% | 27.74% | 27.78% | **25.04%** | **8.55%** | **-6.06% abs** |
| **AI4Bharat / IndicVoices** | 3,164 | 35.53% | 30.41% | 30.39% | **29.83%** | **12.59%** | **-5.70% abs** |
| **OpenSLR SLR122** | 152 | 50.81% | 36.29% | 35.11% | **34.40%** | **14.36%** | **-16.41% abs** |
| **XKaab ASR 4hrs** | 582 | 37.27% | 32.34% | 32.02% | **31.65%** | **15.47%** | **-5.62% abs** |

#### Statistical Significance (10,000 Bootstrap Resamples):
- **B-01 vs B-02A**: $\Delta = -4.44\%$ WER, 95% CI $[-4.71\%, -4.17\%]$, $p < 0.0001$ (Significant)
- **B-02A vs B-02B**: $\Delta = -0.03\%$ WER, 95% CI $[-0.24\%, +0.16\%]$, $p = 0.3764$ (Neutral on secular dev, +1.52% recall on religious lexicon)
- **B-02B vs B-02C**: $\Delta = -1.55\%$ WER, 95% CI $[-1.77\%, -1.32\%]$, $p < 0.0001$ (Significant)
- **B-01 vs B-02C**: $\Delta = -6.02\%$ WER, 95% CI $[-6.33\%, -5.72\%]$, $p < 0.0001$ (Highly Significant)

---

### 2.3 Stage 11 & 12: Robustness Optimization (`stage11_robustness_optimization`, `stage12_robustness_training`)
Investigated model resilience against environmental acoustic noise, reverberation, and lossy Opus compression:
- **Augmentation Pilots**: Evaluated isolated perturbations on pilot checkpoints.
- **R-01 (`R-01_reverb_full`)**: Full fine-tuning with simulated RIR reverberation filters. Final checkpoint: `kashmiri_dataset/experiments/stage12_robustness_training/R-01_reverb_full/best_checkpoint`
- **R-02 (`R-02_combined_full`)**: Full fine-tuning with multi-condition augmentation (Reverb + Additive Environmental Noise + Opus bit-rate throttling). Final checkpoint: `kashmiri_dataset/experiments/stage12_robustness_training/R-02_combined_full/best_checkpoint`

---

### 2.4 Stage 13: Architectural Optimization (`stage13_architectural_optimization`)
Investigated parameter unfreezing strategies and auxiliary loss formulations:
- **Encoder Layer Unfreezing**:
  - `U-01`: Top 4 Transformer layers unfrozen (`checkpoints/U-01_top4/best_checkpoint`).
  - `U-02`: Top 8 Transformer layers unfrozen (`checkpoints/U-02_top8/best_checkpoint`).
  - `U-03`: Top 12 Transformer layers unfrozen (`checkpoints/U-03_top12/best_checkpoint`).
- **Consistency Regularized CTC (`CR-01_crctc_top8`)**: Enforces consistency under stochastic perturbation during training.
- **Intermediate CTC (`I-01_interctc_top12`)**: Injects intermediate CTC supervision at layer 12 of the 24-layer Transformer stack.

---

### 2.5 Stage 14 & 15: MWER Sequence Training & Scaling (`stage14_mwer_checkpoint_optimization`, `stage15_mwer_verification_and_scaling`)
Explored direct sequence-level Minimum Word Error Rate (MWER) loss:

$$\mathcal{L}_{\text{MWER}} = \sum_{y \in \mathcal{H}} \hat{P}(y|x) \left[ \text{WER}(y, y^*) - \hat{\mathbb{E}}_{y' \in \mathcal{H}}[\text{WER}(y', y^*)] \right]$$

- **Stage 14 Pilots**: Tested N-best hypothesis extraction ($N=4, 8, 16, 32$), Temporal Weight Averaging (`WA-01`, `WA-02`, `WA-03`), and gradient stability.
- **Stage 15 Scaling Arms**:
  - `E-00`: Frozen B-02C baseline.
  - `E-01`: Matched CTC continuation ($\lambda=0.0$).
  - `E-02`: MWER sequence fine-tuning ($\lambda=0.05$).
  - `E-03`: MWER sequence fine-tuning ($\lambda=0.10$). **Achieved best tolerant KenLM WER of 21.68%**.
  - `E-04`: Aggressive MWER ($\lambda=0.20$).

---

### 2.6 Romanization & Transliteration Systems
Dual-engine translation pipeline converting Perso-Arabic Kashmiri script into standardized Roman forms:
1. **Rule-Based Transliteration Engine** (`kashmiri_dataset/scripts/kashmiri_transliteration_engine.py`):
   - **Mode 1: ASCII**: Lossless web/database friendly mapping (e.g., `salaam kya haal hai`).
   - **Mode 2: Phonetic / IAST**: High-precision diacritic linguistic mapping (e.g., `salām kyā hāl hai`).
   - Throughput: $>100,000$ characters/second with zero neural latency.
2. **Neural ByT5 Adapter** (`kashmiri_dataset/experiments/transliteration_evaluation/byt5_adapter`): Byte-level sequence-to-sequence model for context-aware ambiguous vowels.
