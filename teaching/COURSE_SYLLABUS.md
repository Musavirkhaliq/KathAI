# EduCasheer Kashmiri ASR: University Course Syllabus

**Course Code**: CS-892 / EE-892  
**Course Title**: Automatic Speech Recognition: From First Principles to Low-Resource Production Systems  
**Academic Level**: Graduate / Advanced Undergraduate  
**Credits**: 4.0 Semester Units (3 Hours Lecture, 2 Hours Laboratory per week)  
**Prerequisites**: Multivariable Calculus, Linear Algebra, Probability & Statistics, Deep Learning (PyTorch), Basic Digital Signal Processing.

---

## 1. Course Description

This course provides an advanced, comprehensive treatment of modern Automatic Speech Recognition (ASR), with a rigorous focus on the unique challenges of building production systems for morphologically rich, low-resource languages. Using the **EduCasheer Kashmiri ASR Research Project** (Stages 1 through 27) as an authentic empirical laboratory, students explore the full pipeline: acoustic physics, Kashmiri phonology and orthography, self-supervised representation learning (Wav2Vec 2.0 / XLS-R), Connectionist Temporal Classification (CTC), dynamic programming decoding, n-gram language model shallow fusion, scientific benchmarking, and production streaming inference.

Students will not merely learn theoretical models; they will examine actual historical failures—such as sample-rate interpolation distortion, speaker data leakage across splits, and minimum WER sequence loss instability—and master the engineering disciplines required to deliver certified state-of-the-art ASR systems.

---

## 2. Grading Scheme

| Component | Weight | Description |
|---|---|---|
| **Homework Assignments (6)** | **30%** | Theoretical problem sets and PyTorch implementation modules (5% each). |
| **Laboratory Sessions & Quizzes** | **15%** | Weekly hands-on lab notebooks and end-of-lecture comprehension quizzes. |
| **Midterm Examination** | **20%** | Comprehensive written & analytical exam covering Modules 01 through 06. |
| **Capstone Project** | **25%** | Complete end-to-end Kashmiri ASR system design, ablation study, and oral defense. |
| **Final Examination** | **10%** | Synoptic examination covering decoding, benchmarking, and production engineering. |

---

## 3. 14-Week Course Schedule

```
+------+-----------------------------+----------------------------------------------------+--------------------------+
| Week | Module                      | Lectures Covered                                   | Lab & Deliverables       |
+------+-----------------------------+----------------------------------------------------+--------------------------+
| 01   | Mod 01: Foundations         | L01: What is ASR? Probability Formulation          | Lab 01: Audio I/O        |
|      |                             | L02: Anatomy of Modern ASR Pipelines               | Setup Environment        |
+------+-----------------------------+----------------------------------------------------+--------------------------+
| 02   | Mod 02: Signal Processing   | L03: Sound Waves, Sampling & Quantization          | Lab 02: FFT & Spectrogram|
|      |                             | L04: Resampling Physics & The 22.05 kHz Defect     | **Assignment 1 Out**     |
+------+-----------------------------+----------------------------------------------------+--------------------------+
| 03   | Mod 02 & Mod 03             | L05: Short-Time Fourier Analysis & Filterbanks     | Lab 03: Mel Filterbanks  |
|      |                             | L06: Kashmiri Phonology & Perso-Arabic Orthography | **Assignment 1 Due**     |
+------+-----------------------------+----------------------------------------------------+--------------------------+
| 04   | Mod 03: Language & Data     | L07: Text Normalization & Unicode Standardization  | Lab 04: Normalizer Impl  |
|      |                             | L08: Speech Corpora Collection & Curation          | **Assignment 2 Out**     |
+------+-----------------------------+----------------------------------------------------+--------------------------+
| 05   | Mod 03 & Mod 04             | L09: Dataset Manifests & Split Contamination Audit | Lab 05: Contamination   |
|      |                             | L10: Deep Learning Foundations: CNNs & Residuals   | **Assignment 2 Due**     |
+------+-----------------------------+----------------------------------------------------+--------------------------+
| 06   | Mod 04: Deep Architectures  | L11: Temporal Convolutions & Receptive Fields      | Lab 06: Wav2Vec2 CNN     |
|      |                             | L12: Self-Attention & Multi-Head Transformers      | **Assignment 3 Out**     |
+------+-----------------------------+----------------------------------------------------+--------------------------+
| 07   | Mod 04 & Mod 05             | L13: Self-Supervised Learning & XLS-R Foundation   | Lab 07: Feature Extract  |
|      |                             | L14: The Alignment Problem in Speech Recognition   | **MIDTERM REVIEW**       |
+------+-----------------------------+----------------------------------------------------+--------------------------+
| 08   | **MIDTERM & CTC**           | **MIDTERM EXAMINATION** (Modules 01 - 04)          | Lab 08: CTC Toy Loss     |
|      | Mod 05: CTC Foundations     | L15: Connectionist Temporal Classification (CTC)   | **Assignment 3 Due**     |
+------+-----------------------------+----------------------------------------------------+--------------------------+
| 09   | Mod 05 & Mod 06             | L16: The Kashmiri CTC Acoustic Model Architecture  | Lab 09: CTC Training     |
|      |                             | L17: Building an End-to-End PyTorch Training Loop  | **Assignment 4 Out**     |
|      |                             | L18: Fine-Tuning Dynamics, Schedules & Warmup      |                          |
+------+-----------------------------+----------------------------------------------------+--------------------------+
| 10   | Mod 06: Scaling & Stability | L19: Acoustic Scaling: Model B-01 to B-02C         | Lab 10: Loss Auditing    |
|      |                             | L20: Training Stability, Gradient Clamping & OOMs  | **Assignment 4 Due**     |
|      |                             |                                                    | **Assignment 5 Out**     |
+------+-----------------------------+----------------------------------------------------+--------------------------+
| 11   | Mod 07: Decoding & LMs      | L21: Greedy Decoding & CTC Prefix Beam Search      | Lab 11: Prefix Search    |
|      |                             | L22: N-gram Language Modeling & Kneser-Ney         | Lab 12: KenLM Shallow    |
|      |                             | L23: Acoustic Temperature & Shallow Fusion         | **Assignment 5 Due**     |
+------+-----------------------------+----------------------------------------------------+--------------------------+
| 12   | Mod 08: Evaluation & Stats  | L24: Levenshtein Distance, WER and CER             | Lab 13: WER Engine       |
|      |                             | L25: Scientific Benchmarking Methodology           | **Assignment 6 Out**     |
|      |                             | L26: Statistical Significance & Bootstrap Tests    |                          |
+------+-----------------------------+----------------------------------------------------+--------------------------+
| 13   | Mod 09 & Mod 10             | L27: Forensic Reconstruction of Stages 1-9         | Lab 14: MWER Analysis    |
|      |                             | L28: The MWER Sequence Optimization Illusion       | **Assignment 6 Due**     |
|      |                             | L29: Dataset Expansion & The E22-01 Breakthrough   |                          |
|      |                             | L30: Auditing S20R, S22C, and S22E Checkpoints     |                          |
+------+-----------------------------+----------------------------------------------------+--------------------------+
| 14   | Mod 11 & Mod 12             | L31: Neural LM Rescoring & Oracle Analysis         | **CAPSTONE DEFENSE**     |
|      |                             | L32: Code-Switching & Dialectal Adaptation         | **FINAL EXAMINATION**    |
|      |                             | L33: Production Streaming & Method C2 Chunking     |                          |
|      |                             | L34: MLOps, Parity Testing & Reproducibility       |                          |
|      |                             | L35: Capstone Synthesis: The Future of Kashmiri AI |                          |
+------+-----------------------------+----------------------------------------------------+--------------------------+
```

---

## 4. Required Texts and References

1. **Course Textbook**: *EduCasheer Kashmiri ASR: Principles, Algorithms, and Research Audits* (Compiled Lecture Notes, `teaching/modules/`).
2. Graves, A., et al. (2006). *Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks*. ICML.
3. Baevski, A., et al. (2020). *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*. NeurIPS.
4. Babu, A., et al. (2021). *XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale*. Interspeech.
5. Heafield, K. (2011). *KenLM: Faster and Smaller Language Model Queries*. WMT.
6. Koul, O. N. (2005). *Modern Kashmiri Grammar*. Dunwoody Press.

---

## 5. Course Policies

- **Academic Rigor**: All implementations must pass unit tests matching production criteria in `src/`.
- **Reproducibility Mandate**: Every experimental claim submitted must include random seeds, manifest hashes, and hardware specifications.
- **Data Ethics**: Students will adhere strictly to ethical guidelines regarding indigenous and low-resource community recordings.
