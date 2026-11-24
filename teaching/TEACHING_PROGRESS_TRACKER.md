# Kashmiri ASR Teaching Course: Progress & Completion Tracker
================================================================

This document provides real-time verification of the curricular components across all 35 lectures, 12 modules, 15 case studies, 6 assignments, and course tooling.

## Status Legend:
- `NOT STARTED`: Directory planned but files not yet written.
- `OUTLINED`: High-level summary created.
- `DRAFTED`: Initial draft in place.
- `CONTENT COMPLETE`: Full textbook narrative, instructor script, math proofs, code walkthrough, lab, exercises, and quiz written.
- `CODE VERIFIED`: All accompanying Python demonstrations and lab code executed and verified.
- `TEACHING READY`: Fully reviewed, cross-referenced, and ready for university instruction.

---

## 1. Master Curriculum & Navigational Guides
| Artifact | Path | Status | Verification Evidence |
| :--- | :--- | :---: | :--- |
| **Course Master Guide** | `teaching/COURSE_MASTER_GUIDE.md` | `TEACHING READY` | Full technical textbook guide written |
| **Course Syllabus** | `teaching/COURSE_SYLLABUS.md` | `TEACHING READY` | 14-week university syllabus created |
| **Lecture Index** | `teaching/LECTURE_INDEX.md` | `TEACHING READY` | Master index linking 35 lectures |
| **Prerequisites Guide** | `teaching/PREREQUISITES.md` | `TEACHING READY` | Math and programming roadmap written |
| **Research-to-Lecture Map** | `teaching/RESEARCH_TO_LECTURE_MAP.md` | `TEACHING READY` | All 27 stages mapped to lectures |
| **Teaching Progress Tracker** | `teaching/TEACHING_PROGRESS_TRACKER.md` | `TEACHING READY` | Real-time tracking document |

---

## 2. The 35 Lectures Tracker
| Lecture ID | Title | Module | Status | Core Verification Elements |
| :---: | :--- | :---: | :---: | :--- |
| **`L-01`** | What Is Automatic Speech Recognition? | Mod 01 | `TEACHING READY` | 11/11 files complete; Intro to ASR & Kashmiri AI |
| **`L-02`** | Anatomy of an End-to-End ASR Pipeline | Mod 01 | `TEACHING READY` | 11/11 files complete; Full pipeline walkthrough |
| **`L-03`** | The Mathematics of Sound and Digital Audio | Mod 02 | `TEACHING READY` | 11/11 files complete; Waveform, sampling, Nyquist |
| **`L-04`** | Audio Preprocessing, Resampling, and the 22.05 kHz Defect | Mod 02 | `TEACHING READY` | 11/11 files complete; Sinc resampling & Stage 20R bug |
| **`L-05`** | Fourier Transforms, Spectrograms, and Filterbanks | Mod 02 | `TEACHING READY` | 11/11 files complete; DFT, STFT, Mel filterbanks |
| **`L-06`** | Kashmiri Linguistics, Phonology, and Perso-Arabic Orthography | Mod 03 | `TEACHING READY` | 11/11 files complete; Kashmiri vowels, Keheh, Goal Heh |
| **`L-07`** | Audio Corpus Harvesting, Field Recording, and Speaker Diversity | Mod 03 | `TEACHING READY` | 11/11 files complete; Ingestion, WAAZ, Quran harvest |
| **`L-08`** | Research-Grade Manifest Engineering and Quality Gates | Mod 03 | `TEACHING READY` | 11/11 files complete; JSONL manifests, SHA-256 gates |
| **`L-09`** | Zero-Speaker Overlap Partitioning and Benchmark Contamination | Mod 03 | `TEACHING READY` | 11/11 files complete; Leakage detection & disjoint splits |
| **`L-10`** | Neural Networks, Computation Graphs, and Sequence Modeling | Mod 04 | `TEACHING READY` | 11/11 files complete; Tensors, autograd, backpropagation |
| **`L-11`** | Temporal Convolutions and the 7-Layer Wav2Vec2 Feature Extractor | Mod 04 | `TEACHING READY` | 11/11 files complete; Conv1d, receptive field derivation |
| **`L-12`** | Multi-Head Self-Attention and Transformer Architectures | Mod 04 | `TEACHING READY` | 11/11 files complete; QKV attention, positional encodings |
| **`L-13`** | Self-Supervised Speech Representation: wav2vec 2.0 & XLS-R | Mod 04 | `TEACHING READY` | 11/11 files complete; Contrastive loss, quantization, XLS-R |
| **`L-14`** | Sequence Alignment and the Fundamental CTC Problem | Mod 05 | `TEACHING READY` | 11/11 files complete; Variable length, blank token collapse |
| **`L-15`** | CTC Mathematics: Forward-Backward Algorithm and Loss Derivation | Mod 05 | `TEACHING READY` | 11/11 files complete; Dynamic programming proof & derivatives |
| **`L-16`** | Building the Kashmiri Acoustic Model (XLS-R + Char CTC) | Mod 05 | `TEACHING READY` | 11/11 files complete; 161-token head, parameter breakdown |
| **`L-17`** | Anatomy of the PyTorch ASR Training Loop | Mod 05 | `TEACHING READY` | 11/11 files complete; Collation, CTC loss forward/backward |
| **`L-18`** | Fine-Tuning Pretrained Models: Schedulers, Mixed Precision, and Regularization | Mod 06 | `TEACHING READY` | 11/11 files complete; AdamW, BF16 AMP, SpecAugment |
| **`L-19`** | Acoustic Data Scaling: The Empirical Trajectory from B-01 to B-02C | Mod 06 | `TEACHING READY` | 11/11 files complete; 92.9h to 165.3h scaling analysis |
| **`L-20`** | Debugging ASR Systems: NaNs, Exploding Gradients, and Memory Profiling | Mod 06 | `TEACHING READY` | 11/11 files complete; CTC gradient stability, OOM avoidance |
| **`L-21`** | CTC Greedy Decoding, Prefix Beam Search, and Token Collapsing | Mod 07 | `TEACHING READY` | 11/11 files complete; Search tree, prefix merging DP |
| **`L-22`** | N-Gram Language Models, Kneser-Ney Smoothing, and KenLM Integration | Mod 07 | `TEACHING READY` | 11/11 files complete; Kneser-Ney, ARPA format, pyctcdecode |
| **`L-23`** | Acoustic-Language Interaction, Temperature Scaling (T=1.08), and Shallow Fusion | Mod 07 | `TEACHING READY` | 11/11 files complete; Posterior entropy, temperature scaling |
| **`L-24`** | Word Error Rate (WER), Character Error Rate (CER), and Levenshtein DP | Mod 08 | `TEACHING READY` | 11/11 files complete; Edit distance DP, corpus aggregation |
| **`L-25`** | Designing Scientific Speech Benchmarks: Isolation, Integrity, and Freezing | Mod 08 | `TEACHING READY` | 11/11 files complete; Benchmark freezing, test set discipline |
| **`L-26`** | Statistical Significance in Speech Research: Paired Bootstrap and Wilcoxon Tests | Mod 08 | `TEACHING READY` | 11/11 files complete; 10,000 bootstrap resamples, p-values |
| **`L-27`** | Research History I: Genesis Baseline B-01 to Foundation SOTA B-02C | Mod 09 | `TEACHING READY` | 11/11 files complete; Chronological deep dive Stages 1-9 |
| **`L-28`** | Research History II: MWER Sequence Loss, Model Soups, and Negative Results | Mod 09 | `TEACHING READY` | 11/11 files complete; MWER illusion, unfreezing failures |
| **`L-29`** | Research History III: Data Expansion, WAAZ Integration, and Champion E22-01 | Mod 09 | `TEACHING READY` | 11/11 files complete; Stage 22 expansion, 24.12% SOTA |
| **`L-30`** | Forensic Integrity Audits: Sinc Resampling Recovery & Exact CTC Slicing | Mod 09 | `TEACHING READY` | 11/11 files complete; S20R audit, S22C exact frame slicing |
| **`L-31`** | Neural Sequence Reranking: Oracle Ceilings, N-Best Lists, and Negative Limits | Mod 10 | `TEACHING READY` | 11/11 files complete; 19.23% Oracle, Stage 24 reranker |
| **`L-32`** | Overcoming Dialectal Mismatch, Code-Switching, and Low-Resource Bottlenecks | Mod 10 | `TEACHING READY` | 11/11 files complete; Multi-dialect, ByT5 Romanization |
| **`L-33`** | Production Deployment: Method C2 Chunking, Web UI, and Latency Optimization | Mod 11 | `TEACHING READY` | 11/11 files complete; Method C2 chunking, Web ASR app |
| **`L-34`** | Reproducible Machine Learning Research Engineering and Code Architecture | Mod 11 | `TEACHING READY` | 11/11 files complete; Modular src/ refactoring, git baselines |
| **`L-35`** | Capstone: Building and Evaluating an End-to-End Kashmiri ASR System | Mod 12 | `TEACHING READY` | 11/11 files complete; Full end-to-end synthesis capstone |

---

## 3. Curricular Support Components
| Component | Directory | Count / Target | Status |
| :--- | :--- | :---: | :---: |
| **Research Case Studies** | `teaching/research_case_studies/` | 15 Case Studies | `TEACHING READY` |
| **Hands-On Assignments** | `teaching/assignments/` | 6 Assignments + Rubrics | `TEACHING READY` |
| **Assessments (Exams)** | `teaching/assessment/` | Midterm & Final Exams | `TEACHING READY` |
| **Presentation Outlines** | `teaching/instructor_resources/presentation_outlines/` | 35 Slide Outlines | `TEACHING READY` |
| **Teaching Datasets** | `teaching/datasets_for_teaching/` | Lightweight manifests | `TEACHING READY` |
| **Automated Validator** | `teaching/tools/validate_course.py` | Full validation suite | `TEACHING READY` |
