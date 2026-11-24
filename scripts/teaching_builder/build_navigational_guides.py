#!/usr/bin/env python3
"""
build_navigational_guides.py
Generates the core navigational and pedagogical documents for the EduCasheer Kashmiri ASR Teaching Course:
1. teaching/COURSE_MASTER_GUIDE.md
2. teaching/COURSE_SYLLABUS.md
3. teaching/LECTURE_INDEX.md
4. teaching/RESEARCH_TO_LECTURE_MAP.md
"""

from pathlib import Path

TEACHING_DIR = Path("/home/musa/Kashmiri Project/teaching")
TEACHING_DIR.mkdir(parents=True, exist_ok=True)

def generate_master_guide():
    content = """# EduCasheer Kashmiri ASR Research: Course Master Guide

Welcome to the **EduCasheer Kashmiri ASR Comprehensive University Course**, an end-to-end, graduate-level curriculum transforming years of cutting-edge research into a rigorous 35-lecture educational experience.

---

## 1. Pedagogical Philosophy & The Dual Narrative

Traditional machine learning and speech recognition textbooks present algorithms in idealized hindsight: clean datasets, convex losses, flawless decoders, and monotonic improvements. In real-world research—particularly for **low-resource, morphologically complex languages like Kashmiri**—the reality is messy, full of subtle bugs, acoustic traps, and mathematical dilemmas.

This course is built upon a **Dual-Narrative Pedagogy**:

```
+-------------------------------------------------------------+
|                     DUAL NARRATIVE ENGINE                   |
+-------------------------------------------------------------+
|                                                             |
|   NARRATIVE 1: First-Principles Speech Science              |
|   - Acoustic wave physics, sampling, filterbanks            |
|   - Convolutional receptive fields & self-attention         |
|   - Dynamic programming (CTC forward-backward, Levenshtein) |
|   - Weighted Finite-State Transducers & Prefix Beam Search  |
|                                                             |
|                              V                              |
|                                                             |
|   NARRATIVE 2: The Kashmiri Empirical Research Odyssey       |
|   - 22.05 kHz sinc-resampling distortion in Bolbosh         |
|   - Speaker contamination across train/val/test splits      |
|   - Exact CTC frame slicing (downsampling factor R=320)     |
|   - The Minimum WER (MWER) sequence-loss training illusion  |
|   - Production Method C2 overlap-add inference chunking     |
|                                                             |
+-------------------------------------------------------------+
```

Every single concept taught in this course follows the **Theory-to-Reality Pipeline**:
$$\\text{Mathematical Derivation} \\longrightarrow \\text{PyTorch Implementation} \\longrightarrow \\text{Historical Empirical Failure/Success} \\longrightarrow \\text{Production Deployment}$$

---

## 2. Curriculum Architecture

The course spans **12 Modules**, comprising **35 Lectures**, **15 Research Case Studies**, **6 Rigorous Assignments**, a **Midterm & Final Examination**, and a **Complete Capstone System Implementation**:

```
+-------------------------------------------------------------------------------+
| MODULE 01: Foundations & System Architecture               (Lectures 01 - 02) |
+-------------------------------------------------------------------------------+
| MODULE 02: Speech Signal Processing & Acoustics            (Lectures 03 - 05) |
+-------------------------------------------------------------------------------+
| MODULE 03: Kashmiri Language, Linguistics & Data           (Lectures 06 - 09) |
+-------------------------------------------------------------------------------+
| MODULE 04: Deep Learning Foundations for Speech            (Lectures 10 - 13) |
+-------------------------------------------------------------------------------+
| MODULE 05: Connectionist Temporal Classification (CTC)     (Lectures 14 - 17) |
+-------------------------------------------------------------------------------+
| MODULE 06: Acoustic Model Training & Scaling               (Lectures 18 - 20) |
+-------------------------------------------------------------------------------+
| MODULE 07: Language Modeling & Decoding Algorithms         (Lectures 21 - 23) |
+-------------------------------------------------------------------------------+
| MODULE 08: Scientific Benchmarking & Error Analysis        (Lectures 24 - 26) |
+-------------------------------------------------------------------------------+
| MODULE 09: Reconstructing Research History                 (Lectures 27 - 30) |
+-------------------------------------------------------------------------------+
| MODULE 10: Advanced ASR Research Frontiers                 (Lectures 31 - 32) |
+-------------------------------------------------------------------------------+
| MODULE 11: Production Systems & Research Engineering       (Lectures 33 - 34) |
+-------------------------------------------------------------------------------+
| MODULE 12: Capstone Synthesis & Defense                    (Lecture 35)       |
+-------------------------------------------------------------------------------+
```

---

## 3. The 11-File Lecture Standard

To guarantee institutional rigor and self-contained teaching, **every single lecture folder** (`teaching/modules/module_XX/lecture_YY/`) contains exactly 11 substantive documents:

| # | File Name | Target Role & Contents |
|---|---|---|
| 1 | `README.md` | Lecture overview, prerequisites, learning objectives, and reading assignments. |
| 2 | `lecture_notes.md` | Comprehensive textbook-style chapter (deep conceptual exposition, figures, derivations). |
| 3 | `instructor_script.md` | Minute-by-minute speaking script, blackboard prompts, common student misconceptions, and demo cues. |
| 4 | `mathematical_foundations.md` | Full mathematical proofs, derivations, tensor shapes, and loss formulations. |
| 5 | `code_walkthrough.md` | Line-by-line inspection of production research code from `src/` and historical stages. |
| 6 | `practical_lab.md` | Hands-on laboratory exercise with runnable scripts, sample inputs, and expected outputs. |
| 7 | `exercises.md` | Homework problems categorized into Analytical, Implementation, and Forensic/Debugging tasks. |
| 8 | `solutions.md` | Fully worked out mathematical solutions, reference code, and grading rubrics. |
| 9 | `quiz.md` | 5-question multiple choice & short-answer conceptual assessment with explanations. |
| 10 | `research_connections.md` | Direct cross-links to Stages 1-27, model checkpoints (B-01..E22-01), logs, and bug audits. |
| 11 | `references.md` | Academic citations, seminal papers (Graves, Baevski, Radford), and Kashmiri linguistic references. |

---

## 4. Course Directory Layout

```
teaching/
├── README.md                           # Quickstart & course orientation
├── PREREQUISITES.md                    # Environment, math, and software requirements
├── TEACHING_PROGRESS_TRACKER.md        # Comprehensive milestone checklist
├── COURSE_MASTER_GUIDE.md             # This document
├── COURSE_SYLLABUS.md                 # 14-week university syllabus
├── LECTURE_INDEX.md                   # Complete hyperlinked lecture directory
├── RESEARCH_TO_LECTURE_MAP.md         # Stage 1-27 cross-reference index
├── modules/                           # 12 Modules containing 35 Lecture directories
│   ├── module_01_introduction/
│   │   ├── README.md
│   │   ├── lecture_01_what_is_asr/    # 11 standard files
│   │   └── lecture_02_asr_anatomy/   # 11 standard files
│   ├── ...
│   └── module_12_future_research/
│       ├── README.md
│       └── lecture_35_capstone/       # 11 standard files
├── research_case_studies/             # 15 In-depth empirical forensic case studies
├── assignments/                       # 6 Multi-part graded programming assignments
├── assessment/                        # Midterm & Final examinations with grading keys
├── capstone/                          # Full End-to-End Kashmiri ASR Capstone Project
├── instructor_resources/              # Presentation outlines and blackboard schematics
├── datasets_for_teaching/             # Lightweight synthetic and audited sample manifests
└── tools/
    └── validate_course.py             # Automated curriculum integrity test suite
```

---

## 5. Lab Environment and Resource Constraints

ASR training typically demands server-grade GPUs. To ensure this course is universally accessible in university labs:
- **Core Laboratory Labs** run on CPU or modest consumer GPUs (e.g., T4/RTX 3060) using lightweight synthetic audio tensors ($T=16000$ samples) or isolated 5-second Kashmiri audio clips.
- **Reference Checkpoints**: All production evaluations reference pre-extracted logits and metadata in `docs/` and `src/`, avoiding the need to redistribute 488 GB of binary checkpoints to students.
- **Strict Parity**: All student exercises adhere to the certified evaluation protocols established in Stage 27 (`tests/test_comprehensive_system_certification.py`).

---

## 6. How to Use This Course

- **For Course Instructors**: Start with [COURSE_SYLLABUS.md](file:///home/musa/Kashmiri%20Project/teaching/COURSE_SYLLABUS.md). For each lecture, read `lecture_notes.md` alongside `instructor_script.md` to pace your blackboard derivations and live coding.
- **For Students & Researchers**: Read `lecture_notes.md`, derive the formulas in `mathematical_foundations.md`, run `practical_lab.md`, and complete the exercises in `exercises.md`.
- **For Reproducibility Auditors**: Consult [RESEARCH_TO_LECTURE_MAP.md](file:///home/musa/Kashmiri%20Project/teaching/RESEARCH_TO_LECTURE_MAP.md) to verify how historical stage artifacts map to educational demonstrations.
"""
    (TEACHING_DIR / "COURSE_MASTER_GUIDE.md").write_text(content, encoding="utf-8")
    print("Generated teaching/COURSE_MASTER_GUIDE.md")

def generate_syllabus():
    content = """# EduCasheer Kashmiri ASR: University Course Syllabus

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
"""
    (TEACHING_DIR / "COURSE_SYLLABUS.md").write_text(content, encoding="utf-8")
    print("Generated teaching/COURSE_SYLLABUS.md")

def generate_lecture_index():
    lectures_meta = [
        (1, "module_01_introduction", "lecture_01_what_is_asr", "What is ASR? Speech Recognition Formulation", "Bayesian formulation, acoustic/language separation, acoustic feature spaces", "src/shared/logging.py", "Stage 1 Foundations"),
        (2, "module_01_introduction", "lecture_02_asr_anatomy", "Anatomy of Modern ASR Pipelines", "End-to-end vs hybrid, encoder-decoder vs CTC, vocabulary design", "src/asr/model.py", "Stage 2 Baseline"),
        (3, "module_02_speech_and_audio", "lecture_03_sound_waves_and_sampling", "Sound Waves, Sampling & Quantization", "Nyquist-Shannon sampling theorem, PCM quantization, bit-depth dynamics", "src/data_processing/audio.py", "Stage 3 Preprocessing"),
        (4, "module_02_speech_and_audio", "lecture_04_resampling_and_22khz_defect", "Resampling Physics & The 22.05 kHz Defect", "Sinc interpolation, anti-aliasing filters, Bolbosh 22.05kHz audit", "src/data_processing/resampling.py", "Stage 25 Resampling Bug"),
        (5, "module_02_speech_and_audio", "lecture_05_fourier_analysis_and_filterbanks", "Short-Time Fourier Analysis & Filterbanks", "STFT, windowing functions, Mel scale, log-mel filterbanks", "src/data_processing/features.py", "Stage 4 Acoustic Features"),
        (6, "module_03_kashmiri_language_and_data", "lecture_06_kashmiri_phonology_and_script", "Kashmiri Phonology & Perso-Arabic Script", "Vowel inventory, matras, diacritics, right-to-left orthography", "src/normalization/kashmiri_char_map.py", "Stage 5 Lexicon"),
        (7, "module_03_kashmiri_language_and_data", "lecture_07_text_normalization", "Text Normalization & Unicode Standardization", "NFD/NFC normalization, zero-width characters, ligature stripping", "src/normalization/normalizer.py", "Stage 26 Normalization v3"),
        (8, "module_03_kashmiri_language_and_data", "lecture_08_speech_corpora_collection", "Speech Corpora Collection & Curation", "Field recordings, SNR thresholding, clipping detection, silence trimming", "src/data_collection/curator.py", "Stage 6 Data Gathering"),
        (9, "module_03_kashmiri_language_and_data", "lecture_09_dataset_manifests_and_leakage", "Dataset Manifests & Split Contamination", "Zero-speaker overlap splits, manifest hashing, data contamination audits", "src/datasets/manifest_validator.py", "Stage 25 Contamination Audit"),
        (10, "module_04_deep_learning_for_asr", "lecture_10_neural_networks_and_backprop", "Deep Learning Foundations: Convolutions & Residuals", "Backpropagation through time, 1D convolutions, layer normalization", "src/asr/layers.py", "Stage 7 Model Architecture"),
        (11, "module_04_deep_learning_for_asr", "lecture_11_temporal_convolutions_and_receptive_fields", "Temporal Convolutions & Receptive Fields", "Strided convolutions, receptive field math, Wav2Vec2 7-layer encoder", "src/asr/wav2vec2_cnn.py", "Stage 8 Encoder Slicing"),
        (12, "module_04_deep_learning_for_asr", "lecture_12_self_attention_and_transformers", "Self-Attention & Multi-Head Transformers", "Scaled dot-product attention, positional embeddings, transformer blocks", "src/asr/transformer_encoder.py", "Stage 9 Transformer Tuning"),
        (13, "module_04_deep_learning_for_asr", "lecture_13_self_supervised_learning_xlsr", "Self-Supervised Learning & XLS-R Foundation", "Masked latent speech modeling, contrastive loss, cross-lingual transfer", "src/asr/xlsr_backbone.py", "Stage 10 XLS-R Checkpoints"),
        (14, "module_05_ctc_and_acoustic_models", "lecture_14_the_alignment_problem", "The Alignment Problem in Speech Recognition", "Monotonic alignment, blank token semantics, collapse operations", "src/decoding/alignment.py", "Stage 11 Alignment"),
        (15, "module_05_ctc_and_acoustic_models", "lecture_15_ctc_loss_mathematics", "Connectionist Temporal Classification: Math & DP", "Forward-backward probabilities, log-sum-exp, exact analytical gradient", "src/asr/ctc_loss.py", "Stage 12 CTC Derivation"),
        (16, "module_05_ctc_and_acoustic_models", "lecture_16_kashmiri_ctc_acoustic_model", "The Kashmiri CTC Acoustic Model Architecture", "Linear projection head, blank index 0, character vocabulary mapping", "src/asr/model.py", "Stage 13 Acoustic Model"),
        (17, "module_05_ctc_and_acoustic_models", "lecture_17_pytorch_training_loop", "Building an End-to-End PyTorch Training Loop", "Dynamic padding collators, gradient accumulation, AMP autocast", "src/asr/trainer.py", "Stage 14 Training Harness"),
        (18, "module_06_training_and_optimization", "lecture_18_finetuning_dynamics_and_schedules", "Fine-Tuning Dynamics, Schedules & Warmup", "Frozen CNN feature extractor, linear warmup, cosine learning rate", "src/asr/optimization.py", "Stage 15 Warmup Schedules"),
        (19, "module_06_training_and_optimization", "lecture_19_acoustic_scaling_b01_to_b02c", "Acoustic Scaling: Model B-01 to B-02C", "Model capacity scaling, 300M vs 1B parameters, Kashmiri performance", "src/asr/model_registry.py", "Stages 16-18 Scaling"),
        (20, "module_06_training_and_optimization", "lecture_20_stability_gradient_clipping_oom", "Training Stability, Gradient Clamping & OOMs", "Gradient explosion prevention, batch sizing, mixed precision scaling", "src/asr/stability.py", "Stage 19 Stability Audits"),
        (21, "module_07_decoding_and_language_models", "lecture_21_greedy_and_beam_search", "Greedy Decoding & CTC Prefix Beam Search", "Collapsed paths, blank vs non-blank merging, beam pruning heuristics", "src/decoding/beam_search.py", "Stage 20 Decoder v1"),
        (22, "module_07_decoding_and_language_models", "lecture_22_ngram_language_models_kenlm", "N-gram Language Modeling & Kneser-Ney Smoothing", "Higher-order n-grams, continuation probability, KenLM binary format", "src/language_models/kenlm_wrapper.py", "Stage 21 LM Training"),
        (23, "module_07_decoding_and_language_models", "lecture_23_acoustic_temperature_and_shallow_fusion", "Acoustic Temperature & Shallow Fusion", "Softmax calibration, alpha/beta grid search, acoustic score weighting", "src/decoding/shallow_fusion.py", "Stage 22 Decoder Tuning"),
        (24, "module_08_benchmarking_and_evaluation", "lecture_24_levenshtein_distance_wer_cer", "Levenshtein Distance, Word & Character Error Rate", "Dynamic programming alignment, substitution/deletion/insertion accounting", "src/evaluation/metrics.py", "Stage 23 Evaluation Metrics"),
        (25, "module_08_benchmarking_and_evaluation", "lecture_25_scientific_benchmarking_methodology", "Scientific Benchmarking Methodology", "Protocol standardization, reference text normalization, disjoint sets", "src/evaluation/benchmark.py", "Stage 27 Protocol"),
        (26, "module_08_benchmarking_and_evaluation", "lecture_26_statistical_significance_bootstrap", "Statistical Significance & Bootstrap Tests", "Non-parametric bootstrap confidence intervals, paired Wilcoxon test", "src/evaluation/statistics.py", "Stage 27 Significance"),
        (27, "module_09_research_experiments", "lecture_27_stages_01_to_09_historical_reconstruction", "Historical Reconstruction: Stages 1 to 9", "Initial baseline experiments, phonetic mapping, vocabulary discovery", "stages/stage_01/", "Stages 1-9 Baselines"),
        (28, "module_09_research_experiments", "lecture_28_the_mwer_negative_result", "The MWER Sequence Optimization Illusion", "Minimum Word Error Rate loss, degradation on unseen domains, audit", "stages/stage_15_mwer/", "Stage 15 MWER Failure"),
        (29, "module_09_research_experiments", "lecture_29_e22_01_expansion_and_breakthrough", "Dataset Expansion & The E22-01 Breakthrough", "Multi-dialectal collection, 106k utterances, sub-25% WER milestone", "configs/asr_e22_01_production.yaml", "Stage 22 Model E22-01"),
        (30, "module_09_research_experiments", "lecture_30_auditing_s20r_s22c_s22e_checkpoints", "Auditing S20R, S22C, and S22E Checkpoints", "Logit parity verification, checkpoint weight hashing, discrepancy fixes", "docs/research/MODEL_REGISTRY.md", "Stage 26-27 Model Audit"),
        (31, "module_10_advanced_asr_research", "lecture_31_neural_rescoring_and_oracle_analysis", "Neural LM Rescoring & Oracle Error Ceilings", "N-best hypothesis extraction, Transformer rescoring, oracle error bound", "src/decoding/neural_rescorer.py", "Stage 24 Rescoring"),
        (32, "module_10_advanced_asr_research", "lecture_32_code_switching_and_dialectal_adaptation", "Code-Switching & Dialectal Adaptation", "Kashmiri-Urdu-English intra-sentential switching, fine-tuning adaptation", "src/translation/code_switch.py", "Stage 24 Adaptation"),
        (33, "module_11_production_systems", "lecture_33_production_streaming_and_chunking", "Production Streaming & Method C2 Overlap-Add", "Long-form audio segmentation, context overlap, stitching seam errors", "src/asr/inference_chunker.py", "Stage 26 Method C2"),
        (34, "module_11_production_systems", "lecture_34_mlops_parity_testing_reproducibility", "MLOps, Parity Testing & Reproducibility", "Continuous integration for speech, numeric regression, golden hashes", "tests/test_comprehensive_system_certification.py", "Stage 27 Parity Harness"),
        (35, "module_12_future_research", "lecture_35_capstone_synthesis", "Capstone Synthesis: The Future of Kashmiri Speech AI", "End-to-end integration, multimodal extensions, speech-to-speech horizons", "src/asr/pipeline.py", "EduCasheer Final Synthesis")
    ]

    rows = []
    for num, mod, ldir, title, concept, code, stage in lectures_meta:
        l_num_str = f"Lecture {num:02d}"
        dir_link = f"[{l_num_str}](file:///home/musa/Kashmiri%20Project/teaching/modules/{mod}/{ldir}/README.md)"
        notes_link = f"[Notes](file:///home/musa/Kashmiri%20Project/teaching/modules/{mod}/{ldir}/lecture_notes.md)"
        math_link = f"[Math](file:///home/musa/Kashmiri%20Project/teaching/modules/{mod}/{ldir}/mathematical_foundations.md)"
        code_link = f"[Code](file:///home/musa/Kashmiri%20Project/teaching/modules/{mod}/{ldir}/code_walkthrough.md)"
        lab_link = f"[Lab](file:///home/musa/Kashmiri%20Project/teaching/modules/{mod}/{ldir}/practical_lab.md)"
        links_str = f"{notes_link} \| {math_link} \| {code_link} \| {lab_link}"
        code_ref_str = f"`{code}`"
        rows.append(f"| {num:02d} | {dir_link} | **{title}** | {concept} | {code_ref_str} | {stage} | {links_str} |")

    table_body = "\n".join(rows)

    content = f"""# EduCasheer Kashmiri ASR: Master Lecture Index

This index provides a complete, hyperlinked directory of all **35 Lectures** across the 12 curriculum modules. Every lecture links directly to its standard sub-documents (`lecture_notes.md`, `mathematical_foundations.md`, `code_walkthrough.md`, `practical_lab.md`).

---

| # | Lecture | Title | Core Concept | Production Code | Stage / Milestone | Quick Links |
|---|---|---|---|---|---|---|
{table_body}

---

## Key Milestone Lectures

- **Lecture 04**: *Resampling Physics & The 22.05 kHz Defect* — The critical audio defect where cubic/linear resampling of Bolbosh speech degraded acoustic representations.
- **Lecture 09**: *Dataset Manifests & Split Contamination* — The empirical audit identifying overlapping speakers across train and test sets.
- **Lecture 11**: *Temporal Convolutions & Receptive Fields* — Mathematical proof of the $R=320$ downsampling factor in the 7-layer CNN encoder.
- **Lecture 15**: *Connectionist Temporal Classification (CTC)* — Full derivation of the forward-backward dynamic programming algorithm and log-domain loss.
- **Lecture 19**: *Acoustic Scaling: Model B-01 to B-02C* — Analysis of scaling XLS-R parameters and the empirical transition from 35.8% to 25.8% WER.
- **Lecture 21**: *Greedy & Prefix Beam Search* — Prefix-tree decoding and language model score integration.
- **Lecture 24**: *Levenshtein Distance, Word & Character Error Rate* — Dynamic programming edit-distance formulation and certified metric calculation.
- **Lecture 28**: *The MWER Negative Result* — The scientific breakdown of why sequence-level loss failed on unseen out-of-domain distributions.
- **Lecture 29**: *Dataset Expansion & The E22-01 Breakthrough* — The decisive training expansion to 106,487 utterances achieving certified 24.12% WER.
- **Lecture 33**: *Production Streaming & Method C2 Chunking* — The industrial overlap-add sliding window algorithm for arbitrarily long audio streams.
- **Lecture 35**: *Capstone Synthesis* — The complete end-to-end integration and defense of the Kashmiri speech recognition pipeline.
"""
    (TEACHING_DIR / "LECTURE_INDEX.md").write_text(content, encoding="utf-8")
    print("Generated teaching/LECTURE_INDEX.md")

def generate_research_map():
    content = """# Research-to-Lecture Mapping Guide: Stages 1 to 27 & Historical Models

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
"""
    (TEACHING_DIR / "RESEARCH_TO_LECTURE_MAP.md").write_text(content, encoding="utf-8")
    print("Generated teaching/RESEARCH_TO_LECTURE_MAP.md")

if __name__ == "__main__":
    generate_master_guide()
    generate_syllabus()
    generate_lecture_index()
    generate_research_map()
    print("All Navigational Guides Generated Successfully!")
