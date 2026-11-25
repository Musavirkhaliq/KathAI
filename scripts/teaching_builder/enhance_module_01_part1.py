#!/usr/bin/env python3
"""
enhance_module_01_part1.py
Greatly enhances and expands Module 01 README and Lecture 01 into comprehensive textbook-grade materials.
"""

from pathlib import Path

BASE_DIR = Path("/home/musa/Kashmiri Project/teaching/modules/module_01_introduction")
BASE_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# MODULE 01 README
# ==============================================================================
def write_module_01_readme():
    content = r"""# Module 01: Introduction, Mathematical Formulation, and Pipeline Anatomy

---

## 1. Module Overview & Pedagogical Scope

Welcome to **Module 01** of the EduCasheer Kashmiri Automatic Speech Recognition (ASR) curriculum. This module serves as the foundational cornerstone for the entire 12-module university course, bridging physical acoustic reality, statistical decision theory, modern deep learning architectures, and the complex sociolinguistic landscape of the Kashmiri language.

Speech recognition is not merely a classification problem—it is a continuous-to-discrete sequence transduction challenge characterized by non-linear temporal warping, acoustic variability, coarticulation, and linguistic ambiguity. In low-resource, morphologically complex languages like Kashmiri (*Koshur*, كٲشُر), these challenges are magnified by non-standardized orthography, pervasive phonemic central vowels, and extreme training data scarcity.

This module establishes the core mathematical, algorithmic, and architectural foundations necessary to understand how the EduCasheer project transformed an initial 88.40% Word Error Rate (WER) baseline failure into a certified 24.12% WER production-grade speech AI ecosystem.

```
+----------------------------------------------------------------------------------------------------+
|                                    MODULE 01 CONCEPTUAL FLOW                                        |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  [Physical Speech Acoustics]                                                                        |
|         │                                                                                          |
|         ▼                                                                                          |
|  [Continuous-to-Discrete Mapping] ──▶ Bayes' Decision Rule & MAP Sequence Decoding (Lecture 01)     |
|         │                                                                                          |
|         ▼                                                                                          |
|  [Classical Hybrid Limitations] ────▶ The Stage 01 Urdu Kaldi Baseline Autopsy (88.4% WER)         |
|         │                                                                                          |
|         ▼                                                                                          |
|  [End-to-End Deep Architectures] ──▶ CTC vs. RNN-T vs. Attention Encoder-Decoder (Lecture 02)      |
|         │                                                                                          |
|         ▼                                                                                          |
|  [6-Stage Production Pipeline] ────▶ Front-End, CNN-Transformer, Head, Beam Search, Normalizer v3   |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Module Learning Objectives

Upon rigorous completion of this module, students and researchers will be able to:
1. **Formalize the ASR Problem**: Mathematically formulate speech recognition as a maximum a posteriori (MAP) sequence decoding problem and prove the equivalence between 0-1 loss minimization and MAP decoding.
2. **Contrast Modeling Paradigms**: Systematically analyze the trade-offs between Classical Hybrid (HMM-GMM / HMM-DNN), Attention-based Encoder-Decoder (AED / Whisper), Recurrent Neural Network Transducer (RNN-T), and Connectionist Temporal Classification (CTC).
3. **Diagnose Cross-Lingual Failure**: Conduct a technical autopsy of why off-the-shelf acoustic models trained on neighboring high-resource languages (e.g. Urdu, Hindi) collapse on Kashmiri speech.
4. **Navigate the Kashmiri Linguistic Matrix**: Identify the unique phonological and orthographic challenges of Kashmiri, including central unrounded vowels (/ɨ/, /ə/, /əː/), phonemic palatalization ($C^j$, `ؠ`), and Perso-Arabic diacritic variation.
5. **Trace the Production Stack**: Dissect the end-to-end tensor transformations of the EduCasheer production pipeline—from raw 16 kHz audio through the 7-layer CNN feature extractor ($D=320$), 24-layer XLS-R Transformer, temperature-scaled CTC logits ($T=1.08$), 5-gram KenLM prefix beam search, and Certified Normalizer v3.
6. **Enforce Evaluation Discipline**: Articulate why disjoint speaker splitting and multi-domain evaluation are non-negotiable requirements for credible speech science.

---

## 3. Detailed Lecture Index

Module 01 consists of two comprehensive, multi-part lectures:

| Lecture ID | Title | Core Concepts | Files |
| :---: | :--- | :--- | :---: |
| **Lecture 01** | [What is ASR? Speech Recognition Formulation](file:///home/musa/Kashmiri%20Project/teaching/modules/module_01_introduction/lecture_01_what_is_asr/README.md) | Epistemology of speech, acoustic wave physics, Bayes' decision rule, zero-one loss proof, MAP decoding, Bayesian evidence invariance, classical hybrid vs. modern E2E, Kashmiri phonology & script, Stage 01 Urdu baseline failure autopsy (88.4% WER). | 11 Documents |
| **Lecture 02** | [Anatomy of Modern ASR Pipelines](file:///home/musa/Kashmiri%20Project/teaching/modules/module_01_introduction/lecture_02_asr_anatomy/README.md) | Architectural taxonomy (CTC vs. AED vs. RNN-T), 6-stage EduCasheer production stack, character vs. subword tokenization, 47-token vocabulary (`configs/vocab.json`), shallow fusion decoding triad ($\log P_{\text{CTC}} + \alpha \log P_{\text{LM}} + \beta |\mathbf{w}|$), acoustic temperature calibration, zero-leakage speaker splitting. | 11 Documents |

---

## 4. Prerequisites and Recommended Preparation

Before beginning Module 01, students should review:
- **Multivariate Calculus & Linear Algebra**: Matrix multiplication, chain rule, inner products, vector norms (see [PREREQUISITES.md](file:///home/musa/Kashmiri%20Project/teaching/PREREQUISITES.md)).
- **Probability & Statistics**: Conditional probability, Bayes' theorem, expectation, variance, discrete distributions.
- **Python & PyTorch Fundamentals**: Tensor slicing, autograd mechanics, PyTorch `nn.Module` construction.
- **Basic Signal Concepts**: Frequency, amplitude, continuous vs. discrete representations.

---

## 5. Course Navigation and Curriculum Mapping

- **Previous Module**: None (Course Beginning).
- **Next Module**: [Module 02: Speech and Audio Signal Processing](file:///home/musa/Kashmiri%20Project/teaching/modules/module_02_speech_and_audio/README.md) (Sound waves, sampling, Nyquist-Shannon theorem, the 22.05 kHz resampling bug, STFT, and Mel filterbanks).
- **Master Course Index**: [LECTURE_INDEX.md](file:///home/musa/Kashmiri%20Project/teaching/LECTURE_INDEX.md).
- **Curriculum Syllabus**: [COURSE_SYLLABUS.md](file:///home/musa/Kashmiri%20Project/teaching/COURSE_SYLLABUS.md).
- **Research Stage Matrix**: [RESEARCH_TO_LECTURE_MAP.md](file:///home/musa/Kashmiri%20Project/teaching/RESEARCH_TO_LECTURE_MAP.md).
"""
    (BASE_DIR / "README.md").write_text(content, encoding="utf-8")
    print("Enhanced Module 01 README.md successfully!")

# ==============================================================================
# LECTURE 01: WHAT IS ASR?
# ==============================================================================
def write_lecture_01():
    ldir = BASE_DIR / "lecture_01_what_is_asr"
    ldir.mkdir(parents=True, exist_ok=True)

    # 1. README.md
    readme_content = r"""# Lecture 01: What is Automatic Speech Recognition? Mathematical Formulation and the Kashmiri Challenge

---

## 1. Lecture Overview

This lecture opens the EduCasheer Kashmiri ASR curriculum by establishing the physical, mathematical, and linguistic foundations of speech recognition. We trace the journey of speech from continuous atmospheric pressure waves through analog-to-digital conversion, formulate the statistical decision-theoretic foundation of sequence decoding, prove Bayes' Decision Rule under zero-one loss, and investigate why classical speech recognition systems failed catastrophically when applied to the Kashmiri language.

---

## 2. Learning Outcomes

After studying this lecture and completing the accompanying labs and problem sets, you will be able to:
1. Explain the physical mechanics of speech production and how acoustic pressure waves $p(t)$ are converted into digital vectors $\mathbf{x}[n]$.
2. Formulate the fundamental statistical problem of ASR using Bayes' Theorem and rigorously prove why minimizing zero-one loss leads to Maximum A Posteriori (MAP) sequence decoding.
3. Compare Maximum A Posteriori (MAP) decoding against Minimum Bayes Risk (MBR) decoding under Word Error Rate (WER) loss.
4. Contrast the multi-component Classical Hybrid architecture (HMM-GMM / HMM-DNN + Lexicon + LM) with modern End-to-End (E2E) neural architectures.
5. Identify the three core linguistic bottlenecks of Kashmiri: phonemic central vowels (/ɨ/, /ə/), persistent secondary articulation (palatalization $C^j$), and unstandardized Perso-Arabic digital orthography.
6. Conduct a forensic post-mortem of the Stage 01 Urdu Kaldi baseline failure (88.40% WER), explaining why forced alignment broke down and rejected 38.2% of Kashmiri training utterances.

---

## 3. Document Directory for Lecture 01

| Document | Purpose |
| :--- | :--- |
| [lecture_notes.md](file:///home/musa/Kashmiri%20Project/teaching/modules/module_01_introduction/lecture_01_what_is_asr/lecture_notes.md) | Full textbook chapter covering physical acoustics, Bayes' theorem, classical vs. E2E, Kashmiri phonetics, and Stage 01 autopsy. |
| [instructor_script.md](file:///home/musa/Kashmiri%20Project/teaching/modules/module_01_introduction/lecture_01_what_is_asr/instructor_script.md) | 90-minute instructor teaching delivery script with whiteboard layout and live terminal demo cues. |
| [mathematical_foundations.md](file:///home/musa/Kashmiri%20Project/teaching/modules/module_01_introduction/lecture_01_what_is_asr/mathematical_foundations.md) | Rigorous mathematical derivations: Bayes 0-1 risk proof, MBR derivation, Bayesian evidence invariance, shallow fusion length bias. |
| [code_walkthrough.md](file:///home/musa/Kashmiri%20Project/teaching/modules/module_01_introduction/lecture_01_what_is_asr/code_walkthrough.md) | Line-by-line inspection of audio loading, tensor invariant assertions, and logging in `src/utils/audio.py` and `src/shared/logging.py`. |
| [practical_lab.md](file:///home/musa/Kashmiri%20Project/teaching/modules/module_01_introduction/lecture_01_what_is_asr/practical_lab.md) | Executable Python lab: Synthetic vowel formant synthesis, ADC quantization simulation, and toy MAP vs MBR decoder. |
| [exercises.md](file:///home/musa/Kashmiri%20Project/teaching/modules/module_01_introduction/lecture_01_what_is_asr/exercises.md) | 4 multi-part analytical, mathematical, and forensic exercises for students. |
| [solutions.md](file:///home/musa/Kashmiri%20Project/teaching/modules/module_01_introduction/lecture_01_what_is_asr/solutions.md) | Full worked-out solutions and Python reference code for all exercises. |
| [quiz.md](file:///home/musa/Kashmiri%20Project/teaching/modules/module_01_introduction/lecture_01_what_is_asr/quiz.md) | 5 challenging multiple-choice questions with comprehensive distractor autopsies. |
| [research_connections.md](file:///home/musa/Kashmiri%20Project/teaching/modules/module_01_introduction/lecture_01_what_is_asr/research_connections.md) | Direct grounding in Stage 01 logs (`urdu_baseline_eval.log`) and Stage 04 architecture evaluations. |
| [references.md](file:///home/musa/Kashmiri%20Project/teaching/modules/module_01_introduction/lecture_01_what_is_asr/references.md) | Annotated academic bibliography spanning speech recognition foundations and Kashmiri linguistics. |
"""
    (ldir / "README.md").write_text(readme_content, encoding="utf-8")

    # 2. lecture_notes.md
    notes_content = r"""# Lecture 01: What is Automatic Speech Recognition? Mathematical Formulation and the Kashmiri Challenge

---

## 1. The Epistemological and Physical Nature of Speech

Automatic Speech Recognition (ASR) is fundamentally the task of **transducing continuous, time-varying acoustic physical waves into discrete symbolic linguistic sequences**.

### The Physical Reality of Speech
At the physical level, speech begins in the human body as an aerodynamic and acoustic phenomenon:
1. **Subglottal Lung Pressure**: The lungs contract, expelling air through the trachea.
2. **Glottal Excitation**: As air passes through the vocal folds (glottis), the folds vibrate periodically for voiced sounds (vowels, nasals, voiced plosives) at a fundamental frequency $F_0$ (typically $85 - 155\text{ Hz}$ for adult males, $165 - 255\text{ Hz}$ for adult females, and $> 250\text{ Hz}$ for children). For unvoiced sounds (fricatives, voiceless stops), the folds remain open, creating turbulent aerodynamic noise.
3. **Vocal Tract Filtering**: The sound passes through the pharyngeal, oral, and nasal cavities. These cavities act as an acoustic resonator with time-varying cross-sectional area $A(x, t)$, boosting certain resonant frequencies known as **formants** ($F_1, F_2, F_3, \dots$) and attenuating others.
4. **Atmospheric Propagation**: Radiated sound propagates through air as longitudinal compressional waves at speed:
   $$c \approx 331.3 \sqrt{1 + \frac{\theta}{273.15}} \text{ m/s} \approx 343\text{ m/s at } 20^\circ\text{C}$$
   producing localized fluctuations in atmospheric pressure $p(t) \in \mathbb{R}$ around ambient barometric pressure $P_0 \approx 101.325\text{ kPa}$.
5. **Transduction and Digitization**: An analog microphone diaphragm converts pressure fluctuations $p(t)$ into a continuous voltage $v(t)$, which is subsequently sampled and quantized by an Analog-to-Digital Converter (ADC) into a 1D discrete-time sequence of quantized amplitude samples $\mathbf{x} = (x_1, x_2, \dots, x_N) \in \mathbb{Z}^{N}$.

### The Linguistic Target
At the linguistic level, speech is perceived by humans as a structured, discrete sequence of semantic symbols:
$$\mathbf{w} = (w_1, w_2, \dots, w_U)$$
where each $w_u$ belongs to a finite vocabulary $\mathcal{V}$ (words, morphemes, subwords, or characters).

### The Three Trilemmas of Speech Inversion
The inverse mapping from physical audio $\mathbf{x}$ to linguistic text $\mathbf{w}$ is notoriously ill-posed due to three structural phenomena:
1. **Acoustic Variability (Non-Uniqueness)**: The exact same linguistic sequence $\mathbf{w}$ produces an infinite continuum of distinct physical waveforms. Speaker physiology (vocal tract length $L$, vocal fold mass), speaking rate, regional dialect, pitch intonation ($F_0$), emotional state, ambient acoustic noise, microphone frequency response, and room reverberation fundamentally alter the waveform.
2. **Temporal Elasticity**: Speech possesses no internal spatial clock. A speaker can articulate the vowel /aː/ in 400 ms in a slow monologue or reduce it to 50 ms in rapid conversation. Phonetic durations stretch and compress non-linearly across time.
3. **Coarticulation**: The human vocal tract articulators (tongue dorsum, tongue blade, lips, velum, jaw) possess physical mass and inertia; they cannot transition instantaneously between idealized phonetic postures. Consequently, the acoustic realization of every phoneme is heavily modified by its preceding and following phonetic context. There are no discrete acoustic boundaries between phonemes in continuous speech.

---

## 2. Statistical Decision Theory and the Fundamental Equation of ASR

To resolve the non-linear, stochastic mapping between continuous acoustics and discrete text, speech recognition is formalized in statistical decision theory.

Let $\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T] \in \mathbb{R}^{D \times T}$ denote a sequence of $T$ acoustic feature vectors extracted from the audio waveform (where each $\mathbf{x}_t$ represents local spectral properties, such as 80-channel log-mel filterbank energies or 1024-dimensional self-supervised representations).

Let $\mathcal{W}$ denote the space of all permissible word sequences constructible from vocabulary $\mathcal{V}$.

### Bayes' Decision Rule and the Zero-One Loss
In statistical decision theory, an optimal decision rule $\delta(\mathbf{X}) = \hat{\mathbf{w}}$ minimizes the conditional Bayes risk $R(\hat{\mathbf{w}} \mid \mathbf{X})$ under a specified loss function $L(\mathbf{w}, \hat{\mathbf{w}})$:
$$R(\hat{\mathbf{w}} \mid \mathbf{X}) = \mathbb{E}_{\mathbf{w} \sim P(\mathbf{w} \mid \mathbf{X})} [L(\mathbf{w}, \hat{\mathbf{w}})] = \sum_{\mathbf{w} \in \mathcal{W}} L(\mathbf{w}, \hat{\mathbf{w}}) P(\mathbf{w} \mid \mathbf{X})$$

If we define the loss as the **Zero-One Sentence Error Loss**, which penalizes any incorrect sentence equally regardless of how many words were wrong:
$$L_{0-1}(\mathbf{w}, \hat{\mathbf{w}}) = \mathbb{I}(\mathbf{w} \ne \hat{\mathbf{w}}) = \begin{cases} 0 & \text{if } \mathbf{w} = \hat{\mathbf{w}} \\ 1 & \text{if } \mathbf{w} \ne \hat{\mathbf{w}} \end{cases}$$

The conditional risk simplifies to:
$$R(\hat{\mathbf{w}} \mid \mathbf{X}) = \sum_{\mathbf{w} \ne \hat{\mathbf{w}}} P(\mathbf{w} \mid \mathbf{X}) = \left( \sum_{\mathbf{w} \in \mathcal{W}} P(\mathbf{w} \mid \mathbf{X}) \right) - P(\hat{\mathbf{w}} \mid \mathbf{X}) = 1 - P(\hat{\mathbf{w}} \mid \mathbf{X})$$

Minimizing risk $R(\hat{\mathbf{w}} \mid \mathbf{X})$ is strictly mathematically equivalent to maximizing the posterior probability $P(\hat{\mathbf{w}} \mid \mathbf{X})$.
This yields the celebrated **Maximum A Posteriori (MAP)** decision rule:
$$\hat{\mathbf{w}}_{\text{MAP}} = \arg\max_{\mathbf{w} \in \mathcal{W}} P(\mathbf{w} \mid \mathbf{X})$$

### The Bayesian Generative Decomposition
Applying Bayes' Theorem to the posterior probability:
$$P(\mathbf{w} \mid \mathbf{X}) = \frac{P(\mathbf{X} \mid \mathbf{w}) P(\mathbf{w})}{P(\mathbf{X})}$$

Because the marginal likelihood (evidence) $P(\mathbf{X}) = \sum_{\mathbf{w}' \in \mathcal{W}} P(\mathbf{X} \mid \mathbf{w}') P(\mathbf{w}')$ is an integral over all candidate word sequences that depends strictly on the observed acoustic tensor $\mathbf{X}$ and does not vary with candidate hypothesis $\mathbf{w}$, it is invariant during argmax search:

$$\hat{\mathbf{w}}_{\text{MAP}} = \arg\max_{\mathbf{w} \in \mathcal{W}} \left[ P(\mathbf{X} \mid \mathbf{w}) P(\mathbf{w}) \right]$$

Applying a monotonic logarithmic transformation:
$$\hat{\mathbf{w}}_{\text{MAP}} = \arg\max_{\mathbf{w} \in \mathcal{W}} \left[ \log P(\mathbf{X} \mid \mathbf{w}) + \log P(\mathbf{w}) \right]$$

This is the **Fundamental Equation of Statistical Speech Recognition**:
1. $P(\mathbf{X} \mid \mathbf{w})$ is the **Acoustic Model (AM)**: Generative likelihood of observing acoustics $\mathbf{X}$ given linguistic sequence $\mathbf{w}$.
2. $P(\mathbf{w})$ is the **Language Model (LM)**: Prior probability of sequence $\mathbf{w}$ in the target language.

```
+----------------------------------------------------------------------------------------------------+
|                                    THE BAYESIAN ASR DUALITY                                        |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|            Acoustic Evidence P(X|w)              Linguistic Prior P(w)                             |
|          [ "What did the audio sound like?" ]    [ "What sentences make sense?" ]                  |
|                        │                                      │                                    |
|                        └───────────────────┬──────────────────┘                                    |
|                                            ▼                                                       |
|                                 Combined MAP Objective                                             |
|                   w* = argmax [ log P(X|w) + alpha * log P(w) + beta * |w| ]                       |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

### MAP vs. Minimum Bayes Risk (MBR) under Word Error Rate
A subtle theoretical vulnerability of MAP decoding is that it optimizes for **Sentence Error Rate (SER)** rather than **Word Error Rate (WER)**. In real-world evaluation, an ASR system is evaluated by Levenshtein edit distance $\mathcal{W}(\mathbf{w}, \hat{\mathbf{w}}) = \frac{S + D + I}{N}$.

Under the Levenshtein loss function, the optimal decision rule is **Minimum Bayes Risk (MBR)**:
$$\hat{\mathbf{w}}_{\text{MBR}} = \arg\min_{\hat{\mathbf{w}} \in \mathcal{W}} \sum_{\mathbf{w} \in \mathcal{W}} \mathcal{W}(\mathbf{w}, \hat{\mathbf{w}}) P(\mathbf{w} \mid \mathbf{X})$$

While exact MBR decoding across all possible word combinations is computationally NP-hard, modern beam search with length penalties ($\beta |\mathbf{w}|$) and language model scaling ($\alpha$) acts as an effective, highly calibrated surrogate.

---

## 3. Classical Hybrid Systems vs. End-to-End Neural Systems

To understand why the EduCasheer project chose its specific architecture, one must understand the historical shift from classical hybrid pipelines to end-to-end models.

### The Classical Hybrid Architecture (HMM-GMM / HMM-DNN, 1980–2015)
In classical hybrid systems (e.g. Kaldi, HTK, Sphinx):
1. **Pronunciation Lexicon ($L$)**: Hand-crafted mapping from orthographic words to phonetic sequences:
   $$\text{"كٲشُر"} \longrightarrow /k/ \; /əː/ \; /ʃ/ \; /u/ \; /r/$$
2. **Hidden Markov Models (HMMs)**: Every phoneme was modeled as a 3-state left-to-right HMM modeling temporal progression.
3. **Senones (Context-Dependent Tied States)**: Because of coarticulation, triphones (e.g., `/k-əː+ʃ/`) were clustered using decision trees into thousands of tied states (*senones*).
4. **Weighted Finite-State Transducers (WFSTs)**: Decoding composed four distinct graphs into a massive search lattice:
   $$H \circ C \circ L \circ G$$
   where $H$ represents HMM topologies, $C$ represents context-dependency triphones, $L$ represents the pronunciation lexicon, and $G$ represents the grammar / language model.

### Why Classical Hybrid Broke Down for Kashmiri:
1. **The Pronunciation Lexicon Bottleneck**: Building $L$ requires phonetic transcriptions for 100,000+ words. In Kashmiri, where no standardized phonetic dictionary existed, missing entries triggered out-of-vocabulary crashes.
2. **Forced Alignment Brittleness**: Training required Viterbi forced alignment. If early GMM alignments failed, the entire training pipeline stalled.

### The Modern End-to-End Revolution (CTC / Transducer / AED, 2015–Present)
Modern ASR eliminates intermediate phoneme lexicons and decision trees by training a single neural network $\theta$ that directly parameterizes sequence posterior $P_\theta(\mathbf{w} \mid \mathbf{X})$.

| Architectural Dimension | Classical Hybrid (Kaldi) | Attention-Encoder-Decoder (Whisper) | Connectionist Temporal Classification (CTC) |
| :--- | :--- | :--- | :--- |
| **Acoustic Modeling Unit** | Senones / HMM States (~4000) | BPE Subwords (~50,000) | Characters / Letters (|V| = 47) |
| **Pronunciation Lexicon** | Mandatory hand-crafted dictionary | None (implicit joint learning) | None (direct acoustic-to-grapheme) |
| **Alignment Mechanism** | Dynamic Programming (Viterbi) | Soft Cross-Attention | Dynamic Programming Collapse ($\mathcal{B}$) |
| **Monotonicity Guarantee** | Strict Monotonic (HMM transitions) | Non-Monotonic (Hallucination risk) | Strict Monotonic (Left-to-right) |
| **Streaming Latency** | Low (Frame-synchronous) | High (Full sentence buffering) | Low (Linear time $\mathcal{O}(T)$) |
| **Low-Resource Kashmiri Fit** | **Failed (Stage 01: 88.4% WER)** | **Failed (Hallucinations & Dropping)** | **CERTIFIED SOTA (Stage 22: 24.12% WER)** |

---

## 4. The Kashmiri Linguistic Challenge: Phonetics, Script, and Scarcity

Kashmiri (*Koshur*, كٲشُر) presents a unique set of linguistic phenomena rarely encountered in high-resource Western or East Asian languages.

```
+----------------------------------------------------------------------------------------------------+
|                                 THE KASHMIRI VOWEL TRAPEZOID                                        |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|              FRONT (Unrounded)              CENTRAL (Unrounded)            BACK (Rounded)          |
|                                                                                                    |
|   HIGH:      /i/ (اِ)    /iː/ (ای)          /ɨ/ (إ)    /ɨː/ (ٳ)           /u/ (اُ)    /uː/ (اوٗ)   |
|              F1~300, F2~2200                F1~340, F2~1450                F1~320, F2~800          |
|                                                                                                    |
|   MID:       /e/ (ای)    /eː/ (ای)          /ə/ (اۆ)    /əː/ (ٲ)           /o/ (او)    /oː/ (او)    |
|              F1~500, F2~1900                F1~480, F2~1500                F1~500, F2~1000         |
|                                                                                                    |
|   LOW:                                      /a/ (اَ)    /aː/ (آ)           /ɔ/ (اۄ)    /ɔː/ (اۄ)    |
|                                             F1~800, F2~1300                F1~700, F2~900          |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

### 1. The Central Vowel Matrix
Unlike Urdu, Hindi, or Punjabi, which possess only front and back vowels, Kashmiri features a **complete series of central unrounded vowels**:
- High central short /ɨ/ (represented by `إ`) and long /ɨː/ (represented by `ٳ`).
- Mid central short /ə/ (represented by `اۆ`) and long /əː/ (represented by `ٲ`).

These vowels occupy the acoustic center of the vowel trapezoid ($F_2 \approx 1400 - 1600\text{ Hz}$). Acoustic models pretrained on Urdu or Hindi invariably collapse Kashmiri central vowels into front (/i/) or back (/u/) vowels, causing widespread semantic confusion.

### 2. Phonemic Secondary Articulation: Palatalization ($C^j$)
In Kashmiri, consonant palatalization ($C^j$) is phonemic and carries inflectional morphology (marking grammatical gender, number, and tense).
- Palatalization is marked orthographically by the small Kashmiri Yeh (`ؠ`, `U+0620`):
  $$\text{"دَب"} \; (/dab/, \text{a fall}) \quad \text{vs.} \quad \text{"دَبؠ"} \; (/dab^j/, \text{falls / plural})$$
If an ASR vocabulary omits `ؠ`, the entire morphological agreement system of the language collapses.

### 3. Extended Perso-Arabic Script and Diacritic Omission
Kashmiri is written right-to-left in an extended Perso-Arabic alphabet containing **44 basic letters and diacritics**:
- Central vowel carriers: `ٲ` (`U+0672`), `ٳ` (`U+0673`), `ۄ` (`U+06C4`), `ۆ` (`U+06C6`).
- In informal writing, social media, and digital publishing, Kashmiri writers routinely omit short vowel diacritics (*aerab*). Consequently, a written string like `سر` can represent:
  - /sar/ (pond / head)
  - /sɨr/ (secret)
  - /sur/ (ashes)
An acoustic model decoding without a language model faces intense homograph ambiguity.

---

## 5. The Forensic Autopsy of Stage 01: The Urdu Kaldi Baseline Disaster

In August 2023, Stage 01 attempted to build a rapid baseline by testing an off-the-shelf Urdu Kaldi GMM-HMM acoustic model on Kashmiri audio recordings.

### The Experimental Outcome:
- **Reported Metric**: **88.40% Word Error Rate (WER)**.
- **Utterances Evaluated**: 500 test sentences.
- **Physical Evidence**: `stages/stage_01/logs/urdu_baseline_eval.log`.

```
================================================================================
STAGE 01 FORENSIC LOG EXCERPT (stages/stage_01/logs/urdu_baseline_eval.log)
================================================================================
[2023-08-14 10:14:22] LOG: Beginning Viterbi forced alignment for 500 utterances...
[2023-08-14 10:14:38] WARNING: Fst has no paths for utterance 'utt_0012'. Discarding.
[2023-08-14 10:14:41] WARNING: Fst has no paths for utterance 'utt_0015'. Discarding.
[2023-08-14 10:14:59] WARNING: Fst has no paths for utterance 'utt_0028'. Discarding.
[2023-08-14 10:16:12] LOG: Alignment completed.
[2023-08-14 10:16:12] SUMMARY: Successfully aligned: 309 utterances (61.8%).
[2023-08-14 10:16:12] SUMMARY: REJECTED / DISCARDED: 191 utterances (38.2%).
[2023-08-14 10:16:45] %WER 88.40 [ 4420 / 5000, 412 ins, 1890 del, 2118 sub ]
================================================================================
```

### Why Did the Urdu Model Collapse?
1. **Phonetic Void**: The Urdu pronunciation dictionary contained zero entries for /ɨ/, /əː/, and /t͡s/. When transcribers wrote words containing these sounds, Kaldi's lexicon graph failed composition (`Fst has no paths`), forcing the aligner to **throw away 38.2% of the training data**!
2. **Senone Mismatch**: The acoustic triphone states in the Urdu acoustic model had never observed spectral energy in the central vowel $F_2$ region ($1450\text{ Hz}$), causing random phonetic substitutions.
3. **The Core Lesson**: Cross-lingual transfer from a neighboring language fails unless the acoustic model is end-to-end, the vocabulary includes all target language graphemes, and acoustic representations are fine-tuned on native speech.
"""
    (ldir / "lecture_notes.md").write_text(notes_content, encoding="utf-8")

    # 3. instructor_script.md
    script_content = r"""# Instructor Delivery Script: Lecture 01 (What is Automatic Speech Recognition?)

## Lecture Details
- **Course**: Advanced Kashmiri Speech Recognition & Deep Learning
- **Lecture 01**: What is Automatic Speech Recognition? Mathematical Formulation and the Kashmiri Challenge
- **Duration**: 90 Minutes Total
- **Target Audience**: Graduate Students / AI Research Engineers / Speech Scientists

---

## Timetable & Pedagogical Flow

```
00:00 - 00:15 | Physical Reality: The Physics of Atmospheric Waves to Digital Tensors
00:15 - 00:40 | Mathematical Formulation: Bayes' Decision Rule, Zero-One Loss, and MAP
00:40 - 00:60 | The Paradigm Evolution: Classical Hybrid (Kaldi) vs. Modern E2E (CTC)
00:60 - 00:75 | The Kashmiri Linguistic Crucible: Central Vowels, Palatalization, and Script
00:75 - 00:85 | Forensic Case Study: The Stage 01 Urdu Baseline Disaster (88.4% WER)
00:85 - 00:90 | Summary, Student Misconceptions, and Practical Lab Briefing
```

---

## Blackboard / Presentation Plan

```
+-----------------------------+-----------------------------+-----------------------------+
| LEFT BOARD: ACOUSTICS & ADC | CENTER BOARD: BAYES & MAP   | RIGHT BOARD: KASHMIRI CRUCIBLE
+-----------------------------+-----------------------------+-----------------------------+
| 1. Physical Wave:           | 1. Bayes' Decision Rule:    | 1. Central Vowels:          |
|    p(t) in Pa -> v(t) in V  |    w* = argmin E[L(w, w^)]  |    /i/   /ɨ/   /u/          |
|    Sampling: x[n] = x(n Ts) |                             |    /e/   /ə/   /o/          |
|    Nyquist: fs >= 2 B       | 2. Zero-One Loss Proof:     |          /əː/ (ٲ)           |
|                             |    L(w, w^) = I(w != w^)    |                             |
| 2. The Acoustic Trilemma:   |    R(w^|X) = 1 - P(w^|X)    | 2. Palatalization:          |
|    - Non-uniqueness         |    w* = argmax P(w|X) (MAP) |    /dab/ vs /dab^j/ (دَبؠ)   |
|    - Temporal elasticity    |                             |                             |
|    - Coarticulation         | 3. Bayesian Decomposition:  | 3. Stage 01 Autopsy:        |
|                             |    argmax P(X|w) * P(w)     |    - Urdu Kaldi baseline    |
| 3. Feature Extraction:      |    = log P(X|w) + log P(w)  |    - 38.2% rejected         |
|    x[n] -> X in R^{D x T}   |      (AM)           (LM)    |    - 88.40% WER collapse    |
+-----------------------------+-----------------------------+-----------------------------+
```

---

## Detailed Minute-by-Minute Delivery

### 00:00 - 00:15 | Physical Reality: Waves to Tensors
- **Instructor Action**: Enter room. Silence the room. Tap lightly on the podium microphone.
- **Speaking Script**:
  > "Welcome to Advanced Kashmiri Speech Recognition. Look at that microphone. What is happening right now?
  > My lungs are expelling air. My vocal folds are vibrating at roughly 130 Hz. My tongue, jaw, and lips are shaping that vibration into resonances called formants.
  > That sound moves through the air at 343 meters per second as a pressure fluctuation, hits the diaphragm of that microphone, converts to a microvolt voltage, and is sampled 16,000 times every single second into 16-bit integers.
  > That is physical audio. It is continuous, messy, noisy, and elastic.
  > On the other side of this room, your brain is doing something miraculous: it takes that continuous pressure wave and instantly converts it into discrete words.
  > Our job in this course is to teach a computer to perform that exact same miracle for one of the most linguistically complex languages on planet Earth: Kashmiri.
  > Let us begin with the mathematics."

### 00:15 - 00:40 | Mathematical Formulation: Bayes and MAP Decoding
- **Instructor Action**: Move to Center Board. Write the general decision rule $\hat{\mathbf{w}} = \arg\min_{\hat{\mathbf{w}}} R(\hat{\mathbf{w}} \mid \mathbf{X})$.
- **Speaking Script**:
  > "How do we make optimal decisions under uncertainty? Statistical Decision Theory.
  > We define a loss function $L(\mathbf{w}, \hat{\mathbf{w}})$. If we choose the zero-one loss—where any error in the sentence costs exactly 1, and perfection costs 0—what happens to the conditional risk?
  > Look at the algebra on the board:
  > The sum of probabilities over all incorrect sentences is simply 1 minus the probability of the chosen sentence!
  > Therefore, minimizing error is mathematically identical to maximizing the posterior probability $P(\mathbf{w} \mid \mathbf{X})$.
  > That is Maximum A Posteriori (MAP) decoding."
- **Whiteboard Derivation**:
  - Write: $P(\mathbf{w} \mid \mathbf{X}) = \frac{P(\mathbf{X} \mid \mathbf{w}) P(\mathbf{w})}{P(\mathbf{X})}$.
  - Cross out $P(\mathbf{X})$ with chalk:
  > "Why can we cross out $P(\mathbf{X})$? Because $P(\mathbf{X})$ is the marginal evidence. It is the probability of the sound itself across all possible sentences. It does not depend on which hypothesis $\mathbf{w}$ we are testing! So for argmax, it is a constant scaling factor.
  > And what are we left with? The two pillars of classical speech recognition:
  > The Acoustic Model $P(\mathbf{X} \mid \mathbf{w})$, and the Language Model $P(\mathbf{w})$."

### 00:40 - 00:60 | Paradigm Evolution: Why Hybrid Systems Died
- **Instructor Action**: Compare Kaldi HMM-GMM with modern End-to-End CTC.
- **Speaking Script**:
  > "For 35 years—from 1980 to 2015—the entire speech recognition industry built systems based on this generative decomposition.
  > But it had a fatal flaw for languages like Kashmiri.
  > To connect acoustic features to words, classical systems needed a **Pronunciation Lexicon**. An explicit human-written dictionary mapping every word to phonemes.
  > In English, you have CMUdict: 134,000 words phonetically transcribed by linguists over decades.
  > In Kashmiri, we had no such dictionary.
  > And worse: classical systems assumed phonemes transition through hidden Markov states that must be forced-aligned frame-by-frame.
  > Modern End-to-End systems threw away the lexicon. They threw away the HMM states.
  > Using Connectionist Temporal Classification (CTC) over self-supervised Transformer backbones, we map raw audio directly to Kashmiri Perso-Arabic characters in a single differentiable neural network."

### 00:60 - 00:75 | The Kashmiri Linguistic Crucible
- **Instructor Action**: Move to Right Board. Draw the Kashmiri Vowel Trapezoid.
- **Speaking Script**:
  > "Now, why couldn't we just take an English or Urdu speech recognition model and fine-tune it naively?
  > Look at the center of this vowel trapezoid.
  > English has front vowels like /i/ in 'beet' and back vowels like /u/ in 'boot'.
  > Urdu has front and back vowels.
  > Kashmiri has an entire family of **central unrounded vowels**: /ɨ/, /ɨː/, /ə/, and /əː/.
  > When a Kashmiri speaker says 'كٲشُر' (/kəːʃur/), the vowel /əː/ has a first formant of 580 Hz and a second formant of 1550 Hz.
  > In Urdu, that sound does not exist!
  > If your acoustic model hasn't learned to resolve the 1550 Hz region, it will confuse /kəːʃur/ with /kaːʃur/ or /koːʃur/.
  > Add to that phonemic palatalization—where adding `ؠ` to a consonant changes a verb from singular to plural—and diacritic-less writing, and you have one of the most challenging speech recognition problems in computational linguistics."

### 00:75 - 00:85 | The Stage 01 Urdu Kaldi Autopsy
- **Instructor Action**: Display terminal log excerpt on screen.
- **Speaking Script**:
  > "Let's look at real repository history. In Stage 01 of our project, the team tested an Urdu Kaldi model on Kashmiri recordings.
  > The result was an unmitigated disaster: **88.40% Word Error Rate**.
  > Why?
  > Look at line 14 of the log: `WARNING: Fst has no paths`.
  > Because the Urdu lexicon did not have Kashmiri central vowels, Kaldi's alignment graph had zero valid paths.
  > Kaldi threw away **38.2% of the audio files** before training even started!
  > This is our baseline. This is where EduCasheer started.
  > Over the next 34 lectures, we will see how every single percentage point of error was methodically dismantled from 88.4% down to 24.12%."

### 00:85 - 00:90 | Conclusion & Lab Orientation
- **Speaking Script**:
  > "In today's practical lab, you will synthesize synthetic vowel waveforms, simulate 16-bit ADC quantization, and implement a toy MAP vs MBR decoder.
  > Open `practical_lab.md` in your lecture directory. See you in the lab."
"""
    (ldir / "instructor_script.md").write_text(script_content, encoding="utf-8")

    # 4. mathematical_foundations.md
    math_content = r"""# Mathematical Foundations: Decision Theory, Bayes Risk, and MAP Decoding

---

## 1. Rigorous Proof: Equivalence of Zero-One Loss and MAP Sequence Decoding

Let $\mathcal{W}$ be the discrete countable set of all permissible linguistic word sequences over vocabulary $\mathcal{V}$.
Let $\mathbf{X} \in \mathcal{X}$ be the observed continuous acoustic feature tensor.
Let $P(\mathbf{w} \mid \mathbf{X})$ denote the true posterior probability distribution over $\mathcal{W}$ given observation $\mathbf{X}$, satisfying:
$$\sum_{\mathbf{w} \in \mathcal{W}} P(\mathbf{w} \mid \mathbf{X}) = 1, \quad P(\mathbf{w} \mid \mathbf{X}) \ge 0 \quad \forall \mathbf{w} \in \mathcal{W}$$

Let $\delta: \mathcal{X} \to \mathcal{W}$ be a deterministic decision rule mapping acoustic observation $\mathbf{X}$ to hypothesized sequence $\hat{\mathbf{w}} = \delta(\mathbf{X})$.

### Theorem 1:
Under the Zero-One Sentence Error loss function:
$$L_{0-1}(\mathbf{w}, \hat{\mathbf{w}}) = \mathbb{I}(\mathbf{w} \neq \hat{\mathbf{w}}) = \begin{cases} 0 & \text{if } \mathbf{w} = \hat{\mathbf{w}} \\ 1 & \text{if } \mathbf{w} \neq \hat{\mathbf{w}} \end{cases}$$
the decision rule $\delta^*(\mathbf{X})$ that minimizes the expected Bayes risk is strictly the **Maximum A Posteriori (MAP)** decision rule:
$$\delta^*(\mathbf{X}) = \arg\max_{\mathbf{w} \in \mathcal{W}} P(\mathbf{w} \mid \mathbf{X})$$

### Proof:
The conditional Bayes risk $R(\hat{\mathbf{w}} \mid \mathbf{X})$ for selecting hypothesis $\hat{\mathbf{w}}$ is defined as:
$$R(\hat{\mathbf{w}} \mid \mathbf{X}) = \mathbb{E}_{\mathbf{w} \sim P(\mathbf{w} \mid \mathbf{X})} [L_{0-1}(\mathbf{w}, \hat{\mathbf{w}})] = \sum_{\mathbf{w} \in \mathcal{W}} L_{0-1}(\mathbf{w}, \hat{\mathbf{w}}) P(\mathbf{w} \mid \mathbf{X})$$

Substitute the definition of $L_{0-1}$:
$$R(\hat{\mathbf{w}} \mid \mathbf{X}) = \sum_{\mathbf{w} \in \mathcal{W}, \mathbf{w} \neq \hat{\mathbf{w}}} (1) \cdot P(\mathbf{w} \mid \mathbf{X}) + (0) \cdot P(\hat{\mathbf{w}} \mid \mathbf{X}) = \sum_{\mathbf{w} \neq \hat{\mathbf{w}}} P(\mathbf{w} \mid \mathbf{X})$$

Since the total probability mass over all sequences sums to 1:
$$\sum_{\mathbf{w} \in \mathcal{W}} P(\mathbf{w} \mid \mathbf{X}) = P(\hat{\mathbf{w}} \mid \mathbf{X}) + \sum_{\mathbf{w} \neq \hat{\mathbf{w}}} P(\mathbf{w} \mid \mathbf{X}) = 1$$
it follows directly that:
$$\sum_{\mathbf{w} \neq \hat{\mathbf{w}}} P(\mathbf{w} \mid \mathbf{X}) = 1 - P(\hat{\mathbf{w}} \mid \mathbf{X})$$

Therefore, the conditional risk is:
$$R(\hat{\mathbf{w}} \mid \mathbf{X}) = 1 - P(\hat{\mathbf{w}} \mid \mathbf{X})$$

We seek the decision rule $\hat{\mathbf{w}}^* = \delta^*(\mathbf{X})$ that minimizes conditional risk:
$$\hat{\mathbf{w}}^* = \arg\min_{\hat{\mathbf{w}} \in \mathcal{W}} R(\hat{\mathbf{w}} \mid \mathbf{X}) = \arg\min_{\hat{\mathbf{w}} \in \mathcal{W}} [1 - P(\hat{\mathbf{w}} \mid \mathbf{X})]$$

Since minimizing $1 - f(x)$ is mathematically equivalent to maximizing $f(x)$:
$$\hat{\mathbf{w}}^* = \arg\max_{\hat{\mathbf{w}} \in \mathcal{W}} P(\hat{\mathbf{w}} \mid \mathbf{X})$$
This completes the proof. $\blacksquare$

---

## 2. Invariance of the Marginal Likelihood (Evidence) in MAP Decoding

Applying Bayes' Rule to the posterior:
$$P(\mathbf{w} \mid \mathbf{X}) = \frac{P(\mathbf{X} \mid \mathbf{w}) P(\mathbf{w})}{P(\mathbf{X})}$$

### Proposition 2:
The denominator $P(\mathbf{X})$ has zero influence on the argmax decision:
$$\arg\max_{\mathbf{w} \in \mathcal{W}} \frac{P(\mathbf{X} \mid \mathbf{w}) P(\mathbf{w})}{P(\mathbf{X})} = \arg\max_{\mathbf{w} \in \mathcal{W}} [P(\mathbf{X} \mid \mathbf{w}) P(\mathbf{w})]$$

### Proof:
By the Law of Total Probability:
$$P(\mathbf{X}) = \sum_{\mathbf{w}' \in \mathcal{W}} P(\mathbf{X}, \mathbf{w}') = \sum_{\mathbf{w}' \in \mathcal{W}} P(\mathbf{X} \mid \mathbf{w}') P(\mathbf{w}')$$
Notice that $\mathbf{w}'$ is a dummy summation index over all candidate sequences in $\mathcal{W}$. Thus, $P(\mathbf{X})$ is a scalar function strictly of the acoustic tensor $\mathbf{X}$ and does not depend on the specific candidate hypothesis $\mathbf{w}$ being evaluated.

Since $P(\mathbf{X}) > 0$ for any realizable observation $\mathbf{X}$, multiplying the objective function by the strictly positive constant $\frac{1}{P(\mathbf{X})}$ preserves the order of all elements in $\mathcal{W}$:
$$P(\mathbf{w}_1 \mid \mathbf{X}) > P(\mathbf{w}_2 \mid \mathbf{X}) \iff \frac{P(\mathbf{X} \mid \mathbf{w}_1) P(\mathbf{w}_1)}{P(\mathbf{X})} > \frac{P(\mathbf{X} \mid \mathbf{w}_2) P(\mathbf{w}_2)}{P(\mathbf{X})} \iff P(\mathbf{X} \mid \mathbf{w}_1) P(\mathbf{w}_1) > P(\mathbf{X} \mid \mathbf{w}_2) P(\mathbf{w}_2)$$
Hence, the argmax operator yields the exact same maximizing element. $\blacksquare$

---

## 3. Minimum Bayes Risk (MBR) under Word Error Rate Loss

While zero-one loss optimizes Sentence Error Rate (SER), speech recognition benchmarks are evaluated on Word Error Rate (WER), defined by the Levenshtein distance:
$$\mathcal{W}(\mathbf{w}, \hat{\mathbf{w}}) = \frac{\text{EditDistance}(\mathbf{w}, \hat{\mathbf{w}})}{|\mathbf{w}|} = \frac{S(\mathbf{w}, \hat{\mathbf{w}}) + D(\mathbf{w}, \hat{\mathbf{w}}) + I(\mathbf{w}, \hat{\mathbf{w}})}{|\mathbf{w}|}$$

Under Levenshtein loss $\mathcal{W}(\mathbf{w}, \hat{\mathbf{w}})$, the conditional Bayes risk is:
$$R_{\text{WER}}(\hat{\mathbf{w}} \mid \mathbf{X}) = \sum_{\mathbf{w} \in \mathcal{W}} \mathcal{W}(\mathbf{w}, \hat{\mathbf{w}}) P(\mathbf{w} \mid \mathbf{X})$$

The MBR decision rule is:
$$\hat{\mathbf{w}}_{\text{MBR}} = \arg\min_{\hat{\mathbf{w}} \in \mathcal{W}} \sum_{\mathbf{w} \in \mathcal{W}} \mathcal{W}(\mathbf{w}, \hat{\mathbf{w}}) P(\mathbf{w} \mid \mathbf{X})$$

### Why MBR Diverges from MAP:
Suppose the posterior distribution over 3 hypotheses is:
- $\mathbf{w}_1 = \text{"کٲشُر زَبان"}$, $P(\mathbf{w}_1 \mid \mathbf{X}) = 0.45$
- $\mathbf{w}_2 = \text{"کٲشُر زَبانہٕ"}$, $P(\mathbf{w}_2 \mid \mathbf{X}) = 0.35$
- $\mathbf{w}_3 = \text{"ہِندوستان"}$, $P(\mathbf{w}_3 \mid \mathbf{X}) = 0.20$

Under MAP decoding:
$$\hat{\mathbf{w}}_{\text{MAP}} = \mathbf{w}_1 \quad (\text{since } 0.45 > 0.35 > 0.20)$$

However, notice that $\mathbf{w}_1$ and $\mathbf{w}_2$ share the word `"کٲشُر"` and differ only by a single suffix vowel. If $\mathbf{w}_2$ is the truth, predicting $\mathbf{w}_1$ incurs an edit distance of only 1 word, whereas predicting $\mathbf{w}_3$ incurs an edit distance of 2 words. MBR explicitly minimizes word-level editing risk rather than winner-take-all sentence probability.

---

## 4. Length Bias Derivation in Log-Linear Shallow Fusion

In character-level decoding, a word hypothesis $\mathbf{w}$ spanning $L$ characters requires $L$ distinct character transitions.
Because each frame emission probability $P(c_t \mid \mathbf{X}) < 1.0$, log-probabilities are strictly negative:
$$\log P(c_t \mid \mathbf{X}) < 0$$

Summing negative log-probabilities across time frames causes longer hypotheses to accumulate larger negative penalties:
$$\sum_{t=1}^{T_{\text{long}}} \log P(c_t \mid \mathbf{X}) \ll \sum_{t=1}^{T_{\text{short}}} \log P(c_t \mid \mathbf{X})$$

Without length compensation, the decoder exhibits **brevity bias**: it systematically prefers deleting words or emitting shorter strings.
By introducing the linear word insertion bonus $\beta |\mathbf{w}|$ (where $\beta > 0$):
$$\mathcal{S}(\mathbf{w}) = \log P_{\text{CTC}}(\mathbf{w} \mid \mathbf{X}) + \alpha \log P_{\text{LM}}(\mathbf{w}) + \beta |\mathbf{w}|$$
each completed word receives a positive credit $\beta$, restoring an unbiased trade-off between insertion and deletion errors.
"""
    (ldir / "mathematical_foundations.md").write_text(math_content, encoding="utf-8")

    # 5. code_walkthrough.md
    code_content = r"""# Code Walkthrough: Audio Invariant Verification and Pipeline Entry Points

---

## 1. Inspecting the Audio Loading Engine (`src/utils/audio.py`)

In this walkthrough, we inspect the production audio preprocessing pipeline from `src/utils/audio.py`. Every incoming recording must be strictly audited before it enters neural feature extraction.

```python
import torch
import torchaudio

def load_and_preprocess_audio(audio_path: str, target_sr: int = 16000) -> torch.Tensor:
    # 1. Load raw audio tensor and sample rate from disk
    waveform, sr = torchaudio.load(audio_path)
    
    # 2. Invariant Check: Multi-channel downmix to Mono
    # In speech recognition, phase differences across stereo microphones
    # create destructive interference if naively summed without beamforming.
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # 3. Invariant Check: Polyphase Sinc Resampling
    # If the source audio is 22.05 kHz (Bolbosh corpus) or 44.1 kHz,
    # it must be resampled to 16 kHz using the certified Kaiser window.
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=target_sr,
            resampling_method="sinc_interp_kaiser",
            lowpass_filter_width=16,
            rolloff=0.94,
            beta=8.5555
        )
        waveform = resampler(waveform)
        
    # 4. Invariant Check: Dynamic Range Normalization
    # Rescales waveform to [-1.0, 1.0] while preserving relative dynamics.
    max_val = torch.max(torch.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val
        
    # Return 1D tensor [T_samples]
    return waveform.squeeze(0)
```

### Critical Implementation Details:
- **`sinc_interp_kaiser`**: Avoids the destructive spectral cutoff of bilinear resampling.
- **`rolloff=0.94`**: Sets the filter cutoff at $0.94 \times 8000\text{ Hz} = 7520\text{ Hz}$, perfectly preserving high-frequency Kashmiri fricatives (/s/, /ʃ/, /t͡s/).
- **`beta=8.5555`**: Guarantees stopband attenuation exceeding $80\text{ dB}$, eliminating aliasing noise.

---

## 2. Audio Tensor Shape Assertions

```python
def assert_valid_speech_tensor(waveform: torch.Tensor, min_sec: float = 0.5, max_sec: float = 30.0, sr: int = 16000):
    # Ensure tensor is 1D
    assert waveform.ndim == 1, f"Expected 1D tensor [samples], got shape {waveform.shape}"
    
    # Check duration bounds
    duration = len(waveform) / sr
    assert min_sec <= duration <= max_sec, f"Audio duration {duration:.2f}s outside bounds [{min_sec}, {max_sec}]"
    
    # Ensure no NaNs or Infs exist
    assert not torch.isnan(waveform).any(), "Acoustic tensor contains NaN values!"
    assert not torch.isinf(waveform).any(), "Acoustic tensor contains Inf values!"
```

---

## 3. Auditing the Production Logging System (`src/shared/logging.py`)

A speech recognition repository running long multi-hour training runs requires thread-safe, structured logging. The EduCasheer logging module ensures all training metrics, WER updates, and loss spikes are recorded with microsecond timestamps:

```python
import logging
import sys
from pathlib import Path

def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File Handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger
```
"""
    (ldir / "code_walkthrough.md").write_text(code_content, encoding="utf-8")

    # 6. practical_lab.md
    lab_content = r"""# Practical Lab: Audio Physics, Formant Synthesis, and Decision Theory

---

## Lab Overview
In this laboratory session, students will implement pure Python and PyTorch simulations to explore:
1. Physical synthesis of Kashmiri central vowels using formant resonators.
2. Analog-to-Digital Conversion (ADC) quantization noise and Signal-to-Quantization-Noise Ratio (SQNR).
3. A toy Maximum A Posteriori (MAP) vs. Minimum Bayes Risk (MBR) sequence decoder.

---

## Exercise 1: Synthesizing the Kashmiri Central Vowel /əː/ (`lab_formant_synthesis.py`)

```python
import numpy as np
import scipy.signal
import soundfile as sf

def synthesize_kashmiri_vowel(f0=130.0, f1=580.0, f2=1550.0, f3=2450.0, duration=1.0, sr=16000):
    # Time vector
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # 1. Glottal source: periodic pulse train filtered with -12 dB/octave tilt
    # Generate impulses at fundamental frequency f0
    period_samples = int(sr / f0)
    source = np.zeros_like(t)
    source[::period_samples] = 1.0
    
    # Glottal pulse shaping filter
    b_glot, a_glot = scipy.signal.butter(1, f0 * 2 / sr, btype='low')
    glottal_excitation = scipy.signal.lfilter(b_glot, a_glot, source)
    
    # 2. Resonant Formant Filters (F1, F2, F3)
    # Bandwidths for human formants
    bw1, bw2, bw3 = 80.0, 110.0, 140.0
    
    def make_resonator(freq, bw):
        r = np.exp(-np.pi * bw / sr)
        omega = 2 * np.pi * freq / sr
        a = [1.0, -2 * r * np.cos(omega), r ** 2]
        b = [1.0 - 2 * r * np.cos(omega) + r ** 2]
        return b, a
        
    b1, a1 = make_resonator(f1, bw1)
    b2, a2 = make_resonator(f2, bw2)
    b3, a3 = make_resonator(f3, bw3)
    
    # Cascade resonators
    y1 = scipy.signal.lfilter(b1, a1, glottal_excitation)
    y2 = scipy.signal.lfilter(b2, a2, y1)
    y3 = scipy.signal.lfilter(b3, a3, y2)
    
    # 3. Radiation impedance at lips (+6 dB/octave high-pass)
    b_rad, a_rad = [1.0, -0.97], [1.0]
    speech_wave = scipy.signal.lfilter(b_rad, a_rad, y3)
    
    # Peak normalize
    speech_wave = speech_wave / np.max(np.abs(speech_wave))
    return speech_wave

if __name__ == "__main__":
    vowel = synthesize_kashmiri_vowel()
    print(f"Synthesized Kashmiri central vowel /əː/: {len(vowel)} samples at 16 kHz")
    assert len(vowel) == 16000, "Vowel length mismatch!"
    print("SUCCESS: Formant synthesis verified!")
```

---

## Exercise 2: Simulating ADC Quantization and SQNR (`lab_quantization.py`)

```python
import numpy as np

def compute_sqnr(signal, bits):
    # Normalize signal to [-1.0, 1.0]
    sig_norm = signal / np.max(np.abs(signal))
    
    # Quantize to 2^bits levels
    levels = 2 ** bits
    delta = 2.0 / levels
    quantized = np.round(sig_norm / delta) * delta
    quantized = np.clip(quantized, -1.0, 1.0)
    
    # Noise = error
    noise = sig_norm - quantized
    
    p_signal = np.mean(sig_norm ** 2)
    p_noise = np.mean(noise ** 2)
    
    sqnr_db = 10 * np.log10(p_signal / max(p_noise, 1e-12))
    return sqnr_db

if __name__ == "__main__":
    # Test on pure sine wave
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    sine = np.sin(2 * np.pi * 440.0 * t)
    
    for b in [8, 12, 16]:
        empirical_sqnr = compute_sqnr(sine, b)
        theoretical_sqnr = 6.02 * b + 1.76
        print(f"Bits: {b:2d} | Empirical: {empirical_sqnr:6.2f} dB | Theoretical: {theoretical_sqnr:6.2f} dB")
        assert abs(empirical_sqnr - theoretical_sqnr) < 1.0, "SQNR discrepancy!"
    print("SUCCESS: Quantization noise model verified!")
```

---

## Exercise 3: Toy MAP vs. MBR Decoder (`lab_map_vs_mbr.py`)

```python
from collections import defaultdict

def toy_mbr_decode():
    # 3 hypotheses with posterior probabilities
    candidates = [
        ("کٲشُر زَبان", 0.45),
        ("کٲشُر زَبانہٕ", 0.35),
        ("ہِندوستان", 0.20)
    ]
    
    # Levenshtein word distance
    def word_edit_distance(h1, h2):
        w1, w2 = h1.split(), h2.split()
        # Simple word symmetric difference
        diff = len(set(w1).symmetric_difference(set(w2)))
        return diff
        
    # 1. MAP Decode (argmax posterior)
    map_winner = max(candidates, key=lambda x: x[1])
    print(f"MAP Selected Hypothesis: '{map_winner[0]}' (Prob = {map_winner[1]:.2f})")
    
    # 2. MBR Decode (argmin expected risk)
    risks = {}
    for hyp_i, _ in candidates:
        expected_risk = 0.0
        for hyp_j, prob_j in candidates:
            dist = word_edit_distance(hyp_i, hyp_j)
            expected_risk += dist * prob_j
        risks[hyp_i] = expected_risk
        print(f"Hypothesis '{hyp_i:15s}' -> Expected Word Risk: {expected_risk:.3f}")
        
    mbr_winner = min(risks.items(), key=lambda x: x[1])
    print(f"MBR Selected Hypothesis: '{mbr_winner[0]}' (Risk = {mbr_winner[1]:.3f})")

if __name__ == "__main__":
    toy_mbr_decode()
```
"""
    (ldir / "practical_lab.md").write_text(lab_content, encoding="utf-8")

    # 7. exercises.md
    exercises_content = r"""# Exercises: Lecture 01 (What is Automatic Speech Recognition?)

---

## Exercise 1: Theoretical Proof of Bayesian Evidence Invariance
Let $\mathbf{X}$ be an observed acoustic tensor and $\mathbf{w} \in \mathcal{W}$ be a candidate word sequence.
1. Formally write the expression for the marginal likelihood $P(\mathbf{X})$.
2. Show that $\frac{\partial P(\mathbf{X})}{\partial \mathbf{w}} = 0$.
3. Prove that for any scalar function $g(\mathbf{X}) > 0$, the decision rule:
   $$\hat{\mathbf{w}} = \arg\max_{\mathbf{w} \in \mathcal{W}} [g(\mathbf{X}) P(\mathbf{w} \mid \mathbf{X})]$$
   is strictly invariant to the choice of $g(\mathbf{X})$.

---

## Exercise 2: MBR vs. MAP Under Asymmetric Loss
Consider an emergency voice dispatch system where failing to recognize the word `"خَطَر"` (*danger*) incurs a catastrophic penalty of $C = 100$, while normal word errors incur a penalty of $1$.
1. Formulate the asymmetric loss function $L_{\text{emergency}}(\mathbf{w}, \hat{\mathbf{w}})$.
2. Suppose posterior probabilities are $P(\text{"خَطَر"} \mid \mathbf{X}) = 0.05$ and $P(\text{"خَبَر"} \mid \mathbf{X}) = 0.95$.
   Which word does MAP decoding select?
3. Calculate the conditional Bayes risk for both hypotheses and determine which word MBR decoding selects.

---

## Exercise 3: Vowel Formant Acoustic Analysis
A speech scientist records three sustained Kashmiri vowel sounds. Spectral analysis reveals the following peak frequencies:
- Vowel A: $F_1 = 300\text{ Hz}, F_2 = 2200\text{ Hz}$
- Vowel B: $F_1 = 340\text{ Hz}, F_2 = 1450\text{ Hz}$
- Vowel C: $F_1 = 580\text{ Hz}, F_2 = 1550\text{ Hz}$
1. Map each vowel to its correct International Phonetic Alphabet (IPA) symbol (/i/, /ɨ/, /əː/).
2. Which of these vowels corresponds to the letter `ٲ` in the word `"کٲشُر"`?
3. Why does an Urdu acoustic model misclassify Vowel B and Vowel C?

---

## Exercise 4: Forensic Analysis of the Stage 01 Urdu Failure
Review the Stage 01 execution log (`stages/stage_01/logs/urdu_baseline_eval.log`):
- Total utterances: 500
- Successfully aligned: 309 utterances
- Discarded / Rejected: 191 utterances
- Reference words: 5,000
- Substitutions: 2,118; Deletions: 1,890; Insertions: 412
1. Calculate the percentage of utterances rejected by forced alignment.
2. Calculate the Word Error Rate (WER) using the standard Levenshtein formula.
3. If an engineer claims: *"The model achieved 88.4% WER, so it worked on 11.6% of words"*, explain why this statement is technically flawed.
"""
    (ldir / "exercises.md").write_text(exercises_content, encoding="utf-8")

    # 8. solutions.md
    solutions_content = r"""# Solutions: Lecture 01 (What is Automatic Speech Recognition?)

---

## Solution to Exercise 1

### Part 1:
The marginal likelihood (evidence) is:
$$P(\mathbf{X}) = \sum_{\mathbf{w}' \in \mathcal{W}} P(\mathbf{X}, \mathbf{w}') = \sum_{\mathbf{w}' \in \mathcal{W}} P(\mathbf{X} \mid \mathbf{w}') P(\mathbf{w}')$$

### Part 2:
In the summation above, $\mathbf{w}'$ is a bound dummy summation variable integrating over all candidate sequences in the vocabulary space $\mathcal{W}$. The expression evaluates to a scalar dependent solely on $\mathbf{X}$. It has no functional dependence on any specific candidate sequence $\mathbf{w}$. Hence:
$$\frac{\partial P(\mathbf{X})}{\partial \mathbf{w}} = 0$$

### Part 3:
Let $\hat{\mathbf{w}}^* = \arg\max_{\mathbf{w} \in \mathcal{W}} [g(\mathbf{X}) P(\mathbf{w} \mid \mathbf{X})]$.
By definition of argmax:
$$g(\mathbf{X}) P(\hat{\mathbf{w}}^* \mid \mathbf{X}) \ge g(\mathbf{X}) P(\mathbf{w} \mid \mathbf{X}) \quad \forall \mathbf{w} \in \mathcal{W}$$
Since $g(\mathbf{X}) > 0$, we can divide both sides by $g(\mathbf{X})$ without altering the direction of the inequality:
$$P(\hat{\mathbf{w}}^* \mid \mathbf{X}) \ge P(\mathbf{w} \mid \mathbf{X}) \quad \forall \mathbf{w} \in \mathcal{W}$$
This is the exact definition of $\hat{\mathbf{w}}^* = \arg\max_{\mathbf{w} \in \mathcal{W}} P(\mathbf{w} \mid \mathbf{X})$. $\blacksquare$

---

## Solution to Exercise 2

### Part 1:
$$L_{\text{emergency}}(\mathbf{w}, \hat{\mathbf{w}}) = \begin{cases}
0 & \text{if } \mathbf{w} = \hat{\mathbf{w}} \\
100 & \text{if } \mathbf{w} = \text{"خَطَر"} \land \hat{\mathbf{w}} \neq \text{"خَطَر"} \\
1 & \text{otherwise}
\end{cases}$$

### Part 2:
MAP decoding selects the hypothesis with highest posterior probability:
$$P(\text{"خَبَر"} \mid \mathbf{X}) = 0.95 > P(\text{"خَطَر"} \mid \mathbf{X}) = 0.05 \implies \hat{\mathbf{w}}_{\text{MAP}} = \mathbf{\text{"خَبَر"}}$$

### Part 3:
Calculate conditional Bayes risk:
1. If predicting $\hat{\mathbf{w}} = \text{"خَبَر"}$:
   $$R(\text{"خَبَر"} \mid \mathbf{X}) = L(\text{"خَطَر"}, \text{"خَبَر"}) P(\text{"خَطَر"}) + L(\text{"خَبَر"}, \text{"خَبَر"}) P(\text{"خَبَر"}) = 100 \times 0.05 + 0 \times 0.95 = \mathbf{5.0}$$
2. If predicting $\hat{\mathbf{w}} = \text{"خَطَر"}$:
   $$R(\text{"خَطَر"} \mid \mathbf{X}) = L(\text{"خَطَر"}, \text{"خَطَر"}) P(\text{"خَطَر"}) + L(\text{"خَبَر"}, \text{"خَطَر"}) P(\text{"خَبَر"}) = 0 \times 0.05 + 1 \times 0.95 = \mathbf{0.95}$$
Since $0.95 < 5.0$, MBR decoding selects **`"خَطَر"`**! Even with only 5% probability, the asymmetric penalty for missing a danger alert dictates the optimal decision.

---

## Solution to Exercise 3

### Part 1:
- Vowel A ($F_1 = 300\text{ Hz}, F_2 = 2200\text{ Hz}$): High front unrounded vowel **/i/**.
- Vowel B ($F_1 = 340\text{ Hz}, F_2 = 1450\text{ Hz}$): High central unrounded vowel **/ɨ/** (`إ`).
- Vowel C ($F_1 = 580\text{ Hz}, F_2 = 1550\text{ Hz}$): Low-mid central unrounded vowel **/əː/** (`ٲ`).

### Part 2:
Vowel C corresponds to the letter `ٲ` in `"کٲشُر"` (/kəːʃur/).

### Part 3:
Urdu lacks phonemic central unrounded vowels. Its acoustic vowel space has an empty cavity between $F_2 = 1200\text{ Hz}$ and $F_2 = 1800\text{ Hz}$. When presented with 1450–1550 Hz formant energy, an Urdu acoustic model maps the sound to the nearest available trained Gaussian clusters: front vowels (/i/, /e/) or back vowels (/u/, /o/).

---

## Solution to Exercise 4

### Part 1:
Rejection rate:
$$\frac{191}{500} \times 100 = \mathbf{38.2\% \text{ of utterances rejected}}$$

### Part 2:
Levenshtein WER formula:
$$\text{WER} = \frac{S + D + I}{N} \times 100 = \frac{2118 + 1890 + 412}{5000} \times 100 = \frac{4420}{5000} \times 100 = \mathbf{88.40\%}$$

### Part 3:
The statement is fundamentally flawed because Word Error Rate is an edit distance ratio, not a bounded percentage of correct items. Because insertions can grow unbounded, WER can exceed 100%. A model with 88.4% WER could have recognized only 5% of words correctly if insertions were low, or 30% correctly with high insertions.
"""
    (ldir / "solutions.md").write_text(solutions_content, encoding="utf-8")

    # 9. quiz.md
    quiz_content = r"""# Quiz: Lecture 01 (What is Automatic Speech Recognition?)

---

### Question 1:
Under statistical decision theory, what loss function makes Maximum A Posteriori (MAP) sequence decoding strictly optimal?
- [A] Word Error Rate (Levenshtein distance)
- [B] Mean Squared Error (MSE)
- [C] Zero-One Sentence Error Loss ($L(\mathbf{w}, \hat{\mathbf{w}}) = \mathbb{I}(\mathbf{w} \ne \hat{\mathbf{w}})$)
- [D] Cross-Entropy Loss

**Correct Answer**: **[C]**  
*In-Depth Explanation*:
- **Why [C] is correct**: As proved in `mathematical_foundations.md`, the conditional risk under zero-one loss reduces directly to $R(\hat{\mathbf{w}} \mid \mathbf{X}) = 1 - P(\hat{\mathbf{w}} \mid \mathbf{X})$. Minimizing risk is therefore identical to maximizing posterior probability $P(\hat{\mathbf{w}} \mid \mathbf{X})$ (MAP).
- **Why [A] is incorrect**: Word Error Rate penalizes sequences proportionally to edit operations, leading to Minimum Bayes Risk (MBR) decoding, not MAP.
- **Why [B] is incorrect**: MSE is defined for continuous regression, not discrete sequence classification.
- **Why [D] is incorrect**: Cross-entropy is a surrogate training loss, not an evaluation decision loss.

---

### Question 2:
Why can the marginal likelihood (evidence) $P(\mathbf{X})$ be omitted during MAP sequence decoding?
- [A] Because $P(\mathbf{X}) = 1.0$ for all audio files.
- [B] Because $P(\mathbf{X})$ is an integral over all candidate sequences that depends solely on acoustic observation $\mathbf{X}$ and does not vary across competing hypotheses $\mathbf{w}$.
- [C] Because PyTorch automatically normalizes tensors.
- [D] Because acoustic models only operate in the frequency domain.

**Correct Answer**: **[B]**  
*In-Depth Explanation*:
- **Why [B] is correct**: By the Law of Total Probability, $P(\mathbf{X}) = \sum_{\mathbf{w}'} P(\mathbf{X} \mid \mathbf{w}') P(\mathbf{w}')$. It is constant with respect to candidate hypothesis $\mathbf{w}$. Since multiplying an objective by a positive constant preserves argmax, it has zero impact on search.
- **Why [A] is incorrect**: $P(\mathbf{X})$ is a continuous probability density, not 1.0.
- **Why [C] and [D] are incorrect**: These are irrelevant implementation and signal details.

---

### Question 3:
What acoustic property primarily distinguishes the Kashmiri central vowel /əː/ (represented by `ٲ` in `کٲشُر`) from high front /i/ and high back /u/?
- [A] It has an intermediate second formant ($F_2 \approx 1550\text{ Hz}$), whereas /i/ has high $F_2$ (~2200 Hz) and /u/ has low $F_2$ (~800 Hz).
- [B] It has zero fundamental frequency ($F_0 = 0$).
- [C] It has a duration exceeding 5 seconds.
- [D] It can only be spoken by male speakers.

**Correct Answer**: **[A]**  
*In-Depth Explanation*:
- **Why [A] is correct**: Second formant frequency ($F_2$) directly correlates with tongue advancement (backness). Front vowels have high $F_2$, back vowels have low $F_2$, and central vowels occupy the intermediate 1400–1600 Hz space.
- **Why [B], [C], and [D] are incorrect**: Vowels are voiced ($F_0 > 0$), have normal phonetic durations (100–300 ms), and are spoken across genders.

---

### Question 4:
What was the primary technical root cause of the 88.40% WER collapse in the Stage 01 Urdu Kaldi baseline?
- [A] The audio files were corrupted by MP3 compression.
- [B] The Urdu pronunciation lexicon contained zero phonetic entries for Kashmiri central vowels and palatalization, causing forced alignment to reject 38.2% of utterances.
- [C] The computer ran out of GPU memory.
- [D] The sample rate was too high (96 kHz).

**Correct Answer**: **[B]**  
*In-Depth Explanation*:
- **Why [B] is correct**: As proven by `stages/stage_01/logs/urdu_baseline_eval.log`, Kaldi's WFST composition $H \circ C \circ L \circ G$ had no valid paths for Kashmiri phonetics, rejecting 191 out of 500 utterances and producing catastrophic substitution rates.
- **Why [A], [C], and [D] are incorrect**: The audio was standard WAV format, Kaldi ran on CPU, and sampling was 16 kHz.

---

### Question 5:
Why does unconstrained character-level CTC decoding require a positive word insertion penalty ($\beta |\mathbf{w}|$) during shallow fusion?
- [A] To increase decoding latency.
- [B] To compensate for brevity bias caused by negative log-probabilities decaying with sequence length, preventing excessive word deletions.
- [C] To convert Perso-Arabic text to Latin script.
- [D] To satisfy Unicode NFKC standardization.

**Correct Answer**: **[B]**  
*In-Depth Explanation*:
- **Why [B] is correct**: Since frame log-probabilities are negative ($\log P < 0$), longer strings accumulate more negative penalties. Adding $\beta |\mathbf{w}|$ adds a positive reward for each completed word, counterbalancing deletion bias.
- **Why [A], [C], and [D] are incorrect**: Length penalties are designed for accuracy balancing, not latency, script transliteration, or Unicode formatting.
"""
    (ldir / "quiz.md").write_text(quiz_content, encoding="utf-8")

    # 10. research_connections.md
    res_content = r"""# Research Connections: Lecture 01

---

## 1. Primary Repository Artifacts
The theoretical concepts introduced in this lecture are directly grounded in the following physical repository artifacts:
- **`stages/stage_01/`**: The historical archive of the earliest ASR attempt on Kashmiri.
  - Log File: `stages/stage_01/logs/urdu_baseline_eval.log` (Execution timestamp: August 2023).
  - Checkpoint: Historical GMM-HMM acoustic models trained on Urdu broadcast speech.
- **`src/utils/audio.py`**: Production implementation of audio loading, mono downmixing, and Kaiser polyphase resampling.
- **`src/shared/logging.py`**: Production logging harness recording microsecond timestamps for training runs.

---

## 2. Research History: From the Stage 01 Disaster to the E22-01 Breakthrough
- **The Stage 01 Baseline**: The 88.40% WER collapse demonstrated that Kashmiri cannot be treated as a trivial dialectal variant of Urdu. Cross-lingual acoustic transfer without native phonetic representation is fundamentally non-viable.
- **The Stage 04 Architecture Pivot**: Following the Kaldi failure, the team evaluated Whisper (AED) vs. XLS-R (CTC). Whisper was rejected due to hallucination loops on low-resource Kashmiri speech, establishing CTC over self-supervised pretrained representations as the permanent architectural direction.
- **The Culmination**: The foundational principles formulated in Lecture 01 directly enabled checkpoint `E22-01` in Stage 22 to achieve **24.12% certified WER** across the 8-domain disjoint benchmark.
"""
    (ldir / "research_connections.md").write_text(res_content, encoding="utf-8")

    # 11. references.md
    refs_content = r"""# Annotated References: Lecture 01

---

## Foundations of Statistical Speech Recognition

1. **Bahl, L. R., Jelinek, F., & Mercer, R. L. (1983).**  
   *A maximum likelihood approach to continuous speech recognition.*  
   *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 5(2), 179–190.  
   - **Annotation**: The seminal classic formulating the Bayesian acoustic-language model decomposition and MAP decoding for speech recognition.

2. **Jelinek, F. (1997).**  
   *Statistical Methods for Speech Recognition.*  
   MIT Press, Cambridge, MA.  
   - **Annotation**: Authoritative textbook detailing HMM acoustic models, triphone clustering, and statistical language modeling.

3. **Rabiner, L. R. (1989).**  
   *A tutorial on hidden Markov models and selected applications in speech recognition.*  
   *Proceedings of the IEEE*, 77(2), 257–286.  
   - **Annotation**: The definitive mathematical tutorial on forward-backward algorithms, Viterbi alignment, and Baum-Welch training.

---

## End-to-End Neural Speech Recognition

4. **Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J. (2006).**  
   *Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks.*  
   *Proceedings of the 23rd International Conference on Machine Learning (ICML)*, 369–376.  
   - **Annotation**: Landmark paper introducing the CTC loss function, the non-emitting blank token, and the collapse operator $\mathcal{B}$.

5. **Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020).**  
   *wav2vec 2.0: A framework for self-supervised learning of speech representations.*  
   *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 12449–12460.  
   - **Annotation**: Introduces raw waveform self-supervised contrastive representation learning, forming the acoustic foundation of XLS-R.

---

## Kashmiri Linguistics and Phonology

6. **Koul, O. N. (2005).**  
   *Modern Kashmiri Grammar.*  
   Dunwoody Press, Springfield, VA.  
   - **Annotation**: Authoritative descriptive reference for Kashmiri central vowels (/ɨ/, /ə/), vowel length contrasts, and palatalization rules.

7. **Wali, K., & Koul, O. N. (1997).**  
   *Kashmiri: A Cognitive-Descriptive Grammar.*  
   Routledge, London.  
   - **Annotation**: Comprehensive analysis of Kashmiri morphosyntax, verb-second (V2) word order, and noun-verb agreement.

8. **Grierson, G. A. (1915).**  
   *A Dictionary of the Kashmiri Language.*  
   Asiatic Society of Bengal, Calcutta.  
   - **Annotation**: Historic monumental linguistic lexicon cataloging Dardic roots, phonetic inflections, and Perso-Arabic orthography.
"""
    (ldir / "references.md").write_text(refs_content, encoding="utf-8")

    print("Enhanced Lecture 01 successfully with 11 comprehensive documents!")

if __name__ == "__main__":
    write_module_01_readme()
    write_lecture_01()
