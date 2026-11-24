# Research-to-Lecture Mapping Guide: Stages 1 to 27 & Historical Models

This document establishes the bidirectional mapping between the **27 Historical Research Stages**, **Historical Experiment Series**, **Key Acoustic Checkpoints**, and the **35 Educational Lectures** of the curriculum.

---

## 1. Stage-to-Lecture Mapping Matrix

| Stage | Historic Research Scope | Primary Deliverables / Findings | Educational Lecture Mapping | Research Case Study |
|---|---|---|---|---|
| **Stage 01** | Initial Environment & Kashmiri Corpus Discovery | First text normalization heuristics, baseline audio collection | **Lecture 01**, **Lecture 06** | CS-01: First Steps in Kashmiri Speech |
| **Stage 02** | Speech Data Scraping & Pre-filtering | Initial Bolbosh and community speech acquisition | **Lecture 08** | CS-02: Scraped Data Realities |
| **Stage 03** | Audio Standardization & Conversion | 16 kHz mono resampling, bit-depth standardizations | **Lecture 03**, **Lecture 04** | CS-03: The 22.05 kHz Resampling Disaster |
| **Stage 04** | Log-Mel Filterbank Feature Engineering | Traditional acoustic feature extraction and cepstral analysis | **Lecture 05** | CS-04: Filterbank vs Raw Waveform Representations |
| **Stage 05** | Kashmiri Orthography & Unicode Mapping | Kashmiri character vocabulary definition, Perso-Arabic diacritics | **Lecture 06**, **Lecture 07** | CS-05: Orthographic Normalization Anomalies |
| **Stage 06** | Audio Validation, Clipping & SNR Filtering | Manifest construction, energy thresholding, silence trimming | **Lecture 08** | CS-06: Signal-to-Noise Ratio Filtering |
| **Stage 07** | Baseline Deep Neural Network Experiments | Early feedforward and recurrent acoustic model baselines | **Lecture 10** | CS-07: Why RNNs Failed on Kashmiri Sequences |
| **Stage 08** | Wav2Vec 2.0 Feature Encoder Exploration | 7-layer temporal convolution receptive field verification | **Lecture 11** | CS-08: Dissecting the 20ms Convolutional Strides |
| **Stage 09** | Transformer Self-Attention Layer Analysis | Attention weight analysis on Kashmiri phonetic clusters | **Lecture 12** | CS-09: Attention Patterns in Vowel Extensions |
| **Stage 10** | Pretrained XLS-R Cross-Lingual Foundation | Evaluation of XLS-R-300M and XLS-R-1B cross-lingual features | **Lecture 13** | CS-10: Cross-Lingual Transfer from Indic Languages |
| **Stage 11** | CTC Alignment & Blank Probability Discovery | Empirical validation of blank token dominance in CTC | **Lecture 14** | CS-11: The Geometry of CTC Blank Spikes |
| **Stage 12** | CTC Forward-Backward DP Implementation | Custom PyTorch CTC loss and numerical log-sum-exp checks | **Lecture 15** | CS-12: Underflow in Forward-Backward Recurrence |
| **Stage 13** | Kashmiri Linear Acoustic Projection Head | Output vocabulary classification head design ($V=98$ tokens) | **Lecture 16** | CS-13: Output Vocabulary Compression |
| **Stage 14** | Distributed PyTorch Training Harness | Mixed precision AMP, gradient accumulation, multi-GPU scaling | **Lecture 17** | CS-14: Scaling Distributed Speech Training |
| **Stage 15** | Minimum Word Error Rate (MWER) Optimization | Investigation of sequence loss training and domain collapse | **Lecture 28** | CS-15: The Sequence Optimization Illusion |
| **Stage 16** | Baseline Model B-01 (10h Data Pilot) | Initial XLS-R-300M fine-tuning on clean audio (WER ~48.2%) | **Lecture 19**, **Lecture 27** | CS-01: Baseline Overfitting Dynamics |
| **Stage 17** | Scaled Model B-02 (50h Data Scaling) | Multi-speaker data expansion, learning rate decay (WER ~35.4%) | **Lecture 19** | CS-02: Acoustic Capacity Scaling |
| **Stage 18** | High-Capacity Model B-02C (100h Milestone) | 100-hour balanced dataset training (disjoint WER ~25.82%) | **Lecture 19**, **Lecture 30** | CS-06: The 100-Hour Benchmark Barrier |
| **Stage 19** | Training Stability & Gradient Clamping | Eliminating NaNs in mixed precision backpropagation | **Lecture 20** | CS-12: Vanishing and Exploding Gradients in XLS-R |
| **Stage 20** | CTC Prefix Beam Search Implementation | Language-model free beam decoding and pruning algorithms | **Lecture 21** | CS-13: Beam Width vs Latency Pareto Frontiers |
| **Stage 21** | Kashmiri KenLM N-Gram Language Models | 3-gram, 4-gram, 5-gram Kneser-Ney language model estimation | **Lecture 22** | CS-14: Smoothing Morphologically Rich Kashmiri |
| **Stage 22** | Model E22-01 & Massive Data Expansion | Scaling to 106,487 utterances, certified 24.12% WER | **Lecture 29**, **Lecture 30** | CS-15: The E22-01 Production Milestone |
| **Stage 23** | Levenshtein Dynamic Programming Suite | Rigorous Word and Character Error Rate evaluation suite | **Lecture 24** | CS-05: Alignment Artifacts in Morphological Errors |
| **Stage 24** | Neural Language Model Rescoring Explorations | Transformer-based N-best rescoring and oracle analyses | **Lecture 31**, **Lecture 32** | CS-11: Oracle Ceilings and Lattice Rescoring |
| **Stage 25** | Independent Forensic Audit & Bug Discovery | Discovery of 22.05 kHz resampling bug and dataset leakage | **Lecture 04**, **Lecture 09** | CS-03 & CS-06: Forensic Audit Methodology |
| **Stage 26** | Modular Codebase Physical Restructuring | Creation of production `src/` modules and clean pipelines | **Lecture 34** | CS-14: Architecture of Modular Research Systems |
| **Stage 27** | Certified Reproducibility & Parity Certification | 100% numerical parity verification across 8 audio domains | **Lecture 34**, **Lecture 35** | CS-15: Certified Mathematical Parity |

---

## 2. Checkpoint Provenance to Educational Experiments

| Model ID | Historic Stage | Checkpoint Architecture | Certified Test WER (Disjoint) | Production Config | Relevant Lectures |
|---|---|---|---|---|---|
| **Model B-01** | Stage 16 | XLS-R-300M + Linear CTC Head | 48.20% | `configs/historical/b01_pilot.yaml` | Lecture 19, Lecture 27 |
| **Model B-02** | Stage 17 | XLS-R-300M + Linear CTC Head | 35.40% | `configs/historical/b02_scaled.yaml` | Lecture 19, Lecture 27 |
| **Model B-02C** | Stage 18 | XLS-R-300M + Linear CTC Head | 25.82% (Greedy) / 25.40% (KenLM) | `configs/asr_b02c_certified.yaml` | Lecture 19, Lecture 23, Lecture 30 |
| **Model S20R** | Stage 20 | XLS-R-300M + Rescored Beam Head | 24.94% | `configs/historical/s20r_rescore.yaml` | Lecture 21, Lecture 30 |
| **Model S22C** | Stage 22 | XLS-R-300M + Calibrated Softmax | 24.40% | `configs/historical/s22c_calibrated.yaml` | Lecture 23, Lecture 30 |
| **Model E22-01** | Stage 22 / 27 | XLS-R-300M + Production Decoder | **24.12% (Certified Benchmark)** | `configs/asr_e22_01_production.yaml` | Lecture 25, Lecture 29, Lecture 33, Lecture 35 |

---

## 3. Notable Historical Traps Covered in Curriculum

1. **The 22.05 kHz Resampling Trap (Lecture 04, Case Study 03)**:
   - *Phenomenon*: Audio from Bolbosh was recorded at 22,050 Hz. Naive downsampling using polynomial or cubic interpolation caused spectral aliasing around 8 kHz, degrading fine-tuning representations.
   - *Solution*: Band-limited sinc interpolation with strict Nyquist low-pass filtering.

2. **The Speaker Contamination Trap (Lecture 09, Case Study 06)**:
   - *Phenomenon*: Random train/test splitting placed identical speakers in both partitions, yielding falsely optimistic WER numbers (up to 7% absolute WER reduction).
   - *Solution*: Graph-based disjoint speaker partitioning with SHA-256 audio identity matching.

3. **The Minimum Word Error Rate (MWER) Illusion (Lecture 28, Case Study 15)**:
   - *Phenomenon*: Training with expected edit-distance loss on beam hypotheses reduced train-set WER but caused catastrophic generalization degradation on out-of-domain evaluation splits (WAAZ speech degraded by +6.8% WER).
   - *Solution*: Regularized multi-task interpolation between sequence loss and token-level CTC cross-entropy.

4. **Exact CTC Slicing & Boundary Artifacts (Lecture 11, Lecture 33)**:
   - *Phenomenon*: Splitting long audio streams at arbitrary second boundaries cut phones in half, producing gibberish tokens at chunk edges.
   - *Solution*: Method C2 overlap-add chunking with 1.0s context margins and exact $R=320$ frame alignment.
