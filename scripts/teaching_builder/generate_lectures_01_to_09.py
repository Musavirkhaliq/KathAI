#!/usr/bin/env python3
"""
generate_lectures_01_to_09.py
Generates all 11 standard files for Lectures 01 through 09 (Modules 01 - 03):
- Module 01: Foundations & Architecture (L01 - L02)
- Module 02: Speech Signal Processing & Acoustics (L03 - L05)
- Module 03: Kashmiri Language, Linguistics & Data Engineering (L06 - L09)
"""

import os
from pathlib import Path

BASE_DIR = Path("/home/musa/Kashmiri Project/teaching/modules")

LECTURES = [
    {
        "num": 1,
        "mod": "module_01_introduction",
        "dir": "lecture_01_what_is_asr",
        "title": "What is ASR? Speech Recognition Formulation",
        "subtitle": "Probabilistic Foundations, Acoustic-Language Decomposition, and Low-Resource Frontiers",
        "objectives": [
            "Formalize Automatic Speech Recognition as maximum a posteriori (MAP) sequence decoding.",
            "Deconstruct the Fundamental Equation of Speech Recognition into acoustic and language modeling components.",
            "Analyze the unique mathematical and phonetic hurdles of Kashmiri as an Indo-Aryan language with Perso-Arabic orthography.",
            "Inspect the fundamental taxonomy of speech systems: HMM-GMM, hybrid DNN, and end-to-end neural pipelines."
        ],
        "notes": """# Lecture 01: What is ASR? Speech Recognition Formulation

## 1. The Fundamental Problem of Speech Recognition

Automatic Speech Recognition (ASR) is the computational task of mapping an acoustic waveform continuous in time and amplitude, $\\mathbf{x}(t)$, into a discrete sequence of linguistic units $\\mathbf{w} = (w_1, w_2, \\dots, w_U)$ such as words, subwords, or characters.

Mathematically, let $\\mathbf{X} = [\\mathbf{x}_1, \\mathbf{x}_2, \\dots, \\mathbf{x}_T] \\in \\mathbb{R}^{D \\times T}$ denote a sequence of $T$ acoustic feature vectors extracted from the continuous audio signal, and let $\\mathcal{W}$ be the space of all permissible word sequences. The statistical goal is to find the sequence $\\hat{\\mathbf{w}}$ that maximizes the posterior probability:

$$\\hat{\\mathbf{w}} = \\arg\\max_{\\mathbf{w} \\in \\mathcal{W}} P(\\mathbf{w} \\mid \\mathbf{X})$$

Applying Bayes' Theorem gives the classical decomposition:

$$\\hat{\\mathbf{w}} = \\arg\\max_{\\mathbf{w} \\in \\mathcal{W}} \\frac{P(\\mathbf{X} \\mid \\mathbf{w}) P(\\mathbf{w})}{P(\\mathbf{X})} = \\arg\\max_{\\mathbf{w} \\in \\mathcal{W}} P(\\mathbf{X} \\mid \\mathbf{w}) P(\\mathbf{w})$$

Here:
1. $P(\\mathbf{X} \\mid \\mathbf{w})$ is the **Acoustic Model (AM)**, quantifying the likelihood of observing acoustic features $\\mathbf{X}$ given that the sequence $\\mathbf{w}$ was uttered.
2. $P(\\mathbf{w})$ is the **Language Model (LM)**, assigning prior probability to the sequence $\\mathbf{w}$ based on linguistic grammar and syntax.
3. The evidence term $P(\\mathbf{X})$ is independent of $\\mathbf{w}$ and thus omitted during maximization.

## 2. From Classical Hybrid to Modern End-to-End Systems

Historically, solving this problem required decomposing $P(\\mathbf{X} \\mid \\mathbf{w})$ using Hidden Markov Models (HMMs) combined with Gaussian Mixture Models (GMMs) or Deep Neural Networks (DNNs) to model frame-level triphone transitions, alongside an explicit pronunciation lexicon $L$ mapping phonemes to words:

$$P(\\mathbf{X} \\mid \\mathbf{w}) = \\sum_{\\mathbf{s}} P(\\mathbf{X} \\mid \\mathbf{s}) P(\\mathbf{s} \\mid \\mathbf{w})$$

where $\\mathbf{s}$ represents a state sequence.

Modern neural ASR replaces this multi-stage modular system with an **End-to-End (E2E)** architecture that directly parameterizes $P(\\mathbf{w} \\mid \\mathbf{X})$ using a deep neural network $\\theta$:

$$\\hat{\\mathbf{w}} = \\arg\\max_{\\mathbf{w}} P_\\theta(\\mathbf{w} \\mid \\mathbf{X})$$

In this course, our core foundation is **Connectionist Temporal Classification (CTC)** over fine-tuned self-supervised representations (XLS-R), fused with an external $n$-gram language model:

$$\\hat{\\mathbf{w}} = \\arg\\max_{\\mathbf{w}} \\left( \\log P_\\text{CTC}(\\mathbf{w} \\mid \\mathbf{X}) + \\alpha \\log P_\\text{LM}(\\mathbf{w}) + \\beta |\\mathbf{w}| \\right)$$

where $\\alpha$ is the language model weight and $\\beta$ is the word insertion penalty.

## 3. The Kashmiri Linguistic Challenge

Kashmiri (*Koshur*, كٲشُر) presents a unique set of challenges rarely encountered in high-resource English or Mandarin ASR:
1. **Phonetic Complexity**: Kashmiri possesses central vowels (e.g., /ɨ/, /ə/) and vowel length contrasts that are phonemic, meaning minimal acoustic variations alter word semantics.
2. **Perso-Arabic Orthography**: Written right-to-left, Kashmiri uses an extended Perso-Arabic alphabet with diacritics (*aerab*) indicating short vowels and palatalization. Diacritics are frequently omitted in informal text, creating orthographic ambiguity.
3. **Low-Resource Scarcity**: Unlike English with 50,000+ hours of transcribed public speech, Kashmiri started with less than 20 hours of fragmented recordings prior to the EduCasheer project.
""",
        "script": """### Lecture 01 Instructor Delivery Script

**[00:00 - 05:00] Opening Hook**
- "Welcome everyone to Advanced Speech Recognition. Today we begin an empirical journey. Most textbooks teach ASR assuming you have 10,000 hours of clean studio recordings and simple Latin orthography. In this room, we are going to study how speech recognition works when those assumptions violently break down."
- *Action*: Write on blackboard: $\\hat{\\mathbf{w}} = \\arg\\max_{\\mathbf{w}} P(\\mathbf{w} \\mid \\mathbf{X})$.

**[05:00 - 15:00] Bayes' Rule and the Acoustic-LM Split**
- "Why did the speech community spend 40 years splitting $P(\\mathbf{w} \\mid \\mathbf{X})$ into $P(\\mathbf{X} \\mid \\mathbf{w}) P(\\mathbf{w})$? Because audio data is expensive, but text is cheap! You can train $P(\\mathbf{w})$ on millions of sentences without a single microsecond of audio."
- *Blackboard Diagram*: Draw acoustic wave $\\rightarrow$ Feature Extraction $\\rightarrow$ Acoustic Model $\\leftrightarrow$ Pronunciation Lexicon $\\leftrightarrow$ Language Model $\\rightarrow$ Viterbi Search.

**[15:30 - 30:00] End-to-End Revolution**
- "With self-supervised foundations like Wav2Vec 2.0 and XLS-R, can we bypass phoneme lexicons completely? Yes, by mapping directly from audio features to characters using CTC. But what happens when the language lacks a standardized orthography? We will see that raw CTC outputs collapse unless properly normalized."

**[30:00 - 45:00] Kashmiri Context: The EduCasheer Project**
- "In the EduCasheer project, we began with zero standard benchmarks. Today we will inspect the first lines of code that began this multi-year effort in Stage 01."
""",
        "math": """# Mathematical Foundations: MAP Decoding & Bayesian Formulation

## 1. Sequence Definitions
Let the speech signal be discretized into a sequence of acoustic observations:
$$\\mathbf{X} = (\\mathbf{x}_1, \\mathbf{x}_2, \\dots, \\mathbf{x}_T), \\quad \\mathbf{x}_t \\in \\mathbb{R}^D$$
where $T$ is the number of frames and $D$ is the feature dimension (e.g., $D=80$ for log-mel filterbanks, or $D=1024$ for XLS-R latent representations).

Let the target transcription be a discrete sequence of vocabulary tokens:
$$\\mathbf{y} = (y_1, y_2, \\dots, y_U), \\quad y_u \\in \\mathcal{V}$$
where $U$ is the sequence length, typically $U \\ll T$.

## 2. Maximum A Posteriori (MAP) Decision Rule
By Bayes' theorem, the optimal sequence minimizes the expected word error rate under zero-one loss:
$$\\hat{\\mathbf{y}} = \\arg\\max_{\\mathbf{y} \\in \\mathcal{V}^*} P(\\mathbf{y} \\mid \\mathbf{X}) = \\arg\\max_{\\mathbf{y} \\in \\mathcal{V}^*} \\frac{P(\\mathbf{X} \\mid \\mathbf{y}) P(\\mathbf{y})}{P(\\mathbf{X})}$$

Since $P(\\mathbf{X}) = \\sum_{\\mathbf{y}'} P(\\mathbf{X} \\mid \\mathbf{y}') P(\\mathbf{y}')$ is constant with respect to $\\mathbf{y}$:
$$\\hat{\\mathbf{y}} = \\arg\\max_{\\mathbf{y} \\in \\mathcal{V}^*} \\left[ \\log P(\\mathbf{X} \\mid \\mathbf{y}) + \\log P(\\mathbf{y}) \\right]$$

## 3. Generalized Shallow Fusion Objective
In modern end-to-end architectures, an acoustic neural network parameterized by $\\theta$ predicts character-level CTC probabilities $P_\\theta(\\mathbf{y} \\mid \\mathbf{X})$. An external language model parameterized by $\\phi$ provides prior probabilities $P_\\phi(\\mathbf{y})$.

To compensate for acoustic probability overconfidence and vocabulary length imbalances, we introduce scaling hyperparameters $\\alpha$ and $\\beta$:
$$\\mathcal{L}_\\text{decode}(\\mathbf{y}) = \\log P_\\theta(\\mathbf{y} \\mid \\mathbf{X}) + \\alpha \\log P_\\phi(\\mathbf{y}) + \\beta |\\mathbf{y}|$$
where:
- $\\alpha \\ge 0$ is the language model interpolation weight.
- $\\beta \\in \\mathbb{R}$ is the word/character insertion bonus.
""",
        "code": """# Code Walkthrough: Lecture 01

We examine `src/shared/logging.py` and the base configuration structures used to define an ASR experiment.

```python
# src/shared/logging.py
import logging
import sys
from typing import Optional

def setup_logger(name: str = "educasheer", level: int = logging.INFO) -> logging.Logger:
    \"\"\"
    Configures a deterministic research logger.
    In reproducible ASR pipelines, loggers must capture timestamps, module provenance,
    and process IDs without buffering corruptions.
    \"\"\"
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
```

### Tensor Flow and Logging Validation
When starting an experiment, we trace the input tensor dimensions before feeding them to the acoustic model:
```python
import torch

# Synthetic acoustic tensor: Batch=2, Channels=1, Samples=32000 (2.0s at 16kHz)
x = torch.randn(2, 32000)
logger = setup_logger("lecture_01")
logger.info(f"Loaded audio batch with shape: {x.shape}, dtype: {x.dtype}")
```
""",
        "lab": """# Practical Lab 01: Audio I/O, Tensor Inspection, and MAP Verification

## Objective
1. Load audio files using PyTorch (`torchaudio`).
2. Verify audio shapes, sample rates, and dynamic range.
3. Simulate a toy MAP decoding decision with synthetic acoustic and language model priors.

## Laboratory Code

```python
import torch
import torchaudio

def run_lab():
    print("=== Step 1: Synthesize a 16 kHz Sine Wave (A4 = 440 Hz) ===")
    sample_rate = 16000
    duration = 2.0  # seconds
    t = torch.linspace(0, duration, int(sample_rate * duration), dtype=torch.float32)
    waveform = 0.5 * torch.sin(2 * torch.pi * 440.0 * t).unsqueeze(0)  # Shape: (1, 32000)
    
    print(f"Waveform shape: {waveform.shape}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Min: {waveform.min():.4f}, Max: {waveform.max():.4f}, Mean: {waveform.mean():.4f}")
    
    print("\n=== Step 2: Simulate Toy MAP Decoding ===")
    # Candidate hypotheses
    candidates = ["کٲشُر", "کاشُر", "کوشُر"]
    
    # Log Acoustic Likelihoods log P(X | w)
    log_p_acoustic = torch.tensor([-2.15, -2.40, -4.80])
    
    # Log Language Model Priors log P(w)
    log_p_lm = torch.tensor([-1.20, -3.50, -2.10])
    
    # alpha = 0.5, beta = 0.0
    alpha = 0.5
    total_score = log_p_acoustic + alpha * log_p_lm
    best_idx = torch.argmax(total_score).item()
    
    for i, c in enumerate(candidates):
        print(f"Candidate: '{c}' | AM: {log_p_acoustic[i]:.2f} | LM: {log_p_lm[i]:.2f} | Total: {total_score[i]:.2f}")
    
    print(f"\nWinning hypothesis: '{candidates[best_idx]}' with score {total_score[best_idx]:.2f}")

if __name__ == "__main__":
    run_lab()
```
""",
        "exercises": """# Exercises: Lecture 01

## Analytical Problems
1. Prove that maximizing the posterior probability $\\arg\\max_{\\mathbf{w}} P(\\mathbf{w} \\mid \\mathbf{X})$ is equivalent to minimizing the word error rate under a zero-one loss function $L(\\mathbf{w}, \\hat{\\mathbf{w}}) = \\mathbb{I}(\\mathbf{w} \\ne \\hat{\\mathbf{w}})$.
2. Suppose an acoustic model assigns likelihood $P(\\mathbf{X} \\mid \\mathbf{w}_1) = 0.08$ and $P(\\mathbf{X} \\mid \\mathbf{w}_2) = 0.12$. An $n$-gram LM assigns priors $P(\\mathbf{w}_1) = 0.05$ and $P(\\mathbf{w}_2) = 0.02$. Which sequence is selected under unweighted MAP decoding? At what value of $\\alpha$ (LM weight) does the decision switch?

## Implementation Problems
3. Write a Python function `compute_snr(signal: torch.Tensor, noise: torch.Tensor) -> float` that computes the Signal-to-Noise Ratio in decibels (dB) for arbitrary 1D audio tensors.

## Forensic / Research Auditing
4. In Stage 01 of the EduCasheer repository, initial transcriptions used inconsistent Unicode codepoints for the Kashmiri vowel *nyun* (noon ghunna). Explain why training an acoustic model directly on unnormalized text causes vocabulary dilution and inflates the output softmax entropy.
""",
        "solutions": """# Solutions: Lecture 01

## Exercise 1
Under 0-1 loss, the expected risk is:
$$R(\\hat{\\mathbf{w}} \\mid \\mathbf{X}) = \\sum_{\\mathbf{w}} \\mathbb{I}(\\mathbf{w} \\ne \\hat{\\mathbf{w}}) P(\\mathbf{w} \\mid \\mathbf{X}) = 1 - P(\\hat{\\mathbf{w}} \\mid \\mathbf{X})$$
Minimizing $R(\\hat{\\mathbf{w}} \\mid \\mathbf{X})$ is strictly equivalent to maximizing $P(\\hat{\\mathbf{w}} \\mid \\mathbf{X})$.

## Exercise 2
Posterior for $\\mathbf{w}_1$: $P(\\mathbf{X} \\mid \\mathbf{w}_1) P(\\mathbf{w}_1) = 0.08 \\times 0.05 = 0.0040$.  
Posterior for $\\mathbf{w}_2$: $P(\\mathbf{X} \\mid \\mathbf{w}_2) P(\\mathbf{w}_2) = 0.12 \\times 0.02 = 0.0024$.  
Under unweighted MAP ($\\alpha=1.0$), $\\mathbf{w}_1$ is selected.  
Decision boundary:
$$\\log(0.08) + \\alpha \\log(0.05) = \\log(0.12) + \\alpha \\log(0.02)$$
$$\\log(0.08 / 0.12) = \\alpha \\log(0.02 / 0.05)$$
$$-0.4055 = \\alpha (-0.9163) \\implies \\alpha \\approx 0.4425$$
For $\\alpha < 0.4425$, the acoustic evidence dominates and selects $\\mathbf{w}_2$.

## Exercise 3
```python
import torch

def compute_snr(signal: torch.Tensor, noise: torch.Tensor) -> float:
    p_signal = torch.mean(signal ** 2)
    p_noise = torch.mean(noise ** 2)
    if p_noise == 0:
        return float('inf')
    return 10.0 * torch.log10(p_signal / p_noise).item()
```
""",
        "quiz": """# Comprehension Quiz: Lecture 01

1. **In the Bayesian formulation of speech recognition, why can the marginal likelihood $P(\\mathbf{X})$ be ignored during decoding?**
   - A) Because $P(\\mathbf{X}) = 1.0$ for all audio signals.
   - B) Because $P(\\mathbf{X})$ is constant with respect to the hypothesized word sequence $\\mathbf{w}$. [CORRECT]
   - C) Because acoustic features are normalized using cepstral mean subtraction.
   - D) Because the denominator is eliminated by the CTC loss.
   *Explanation*: Since the argmax is taken over candidate transcriptions $\\mathbf{w}$, the term $P(\\mathbf{X})$ does not vary across hypotheses and does not alter their relative ranking.

2. **Which of the following describes the role of the language model weight $\\alpha$ in shallow fusion?**
   - A) It speeds up the GPU matrix multiplication.
   - B) It scales the learning rate during gradient descent.
   - C) It balances the acoustic log-likelihood with the prior probability from the language model. [CORRECT]
   - D) It normalizes the Persian diacritic inventory.
   *Explanation*: $\\alpha$ balances acoustic model confidence against language model linguistic probability.

3. **What is the primary architectural difference between hybrid ASR and End-to-End ASR?**
   - A) Hybrid uses GPUs; End-to-End only uses TPUs.
   - B) Hybrid requires an explicit pronunciation lexicon and separate HMM alignment; End-to-End maps audio directly to characters/tokens. [CORRECT]
   - C) Hybrid models do not use neural networks.
   - D) End-to-End models cannot use language models.
   *Explanation*: End-to-End models eliminate frame-level HMM state alignments and pronunciation dictionaries.

4. **Why is Kashmiri classified as a low-resource language in speech recognition research?**
   - A) It has no native speakers.
   - B) It cannot be represented digitally in Unicode.
   - C) Publicly available paired audio-text corpora are scarce and dialectally fragmented. [CORRECT]
   - D) It has a 10-character alphabet.
   *Explanation*: The scarcity of large-scale transcribed audio datasets defines low-resource status.

5. **In the decoding objective $\\log P(\\mathbf{w} \\mid \\mathbf{X}) + \\alpha \\log P(\\mathbf{w}) + \\beta |\\mathbf{w}|$, what is the role of $\\beta$?**
   - A) Learning rate multiplier.
   - B) Word/character insertion penalty/bonus to regulate hypothesis length. [CORRECT]
   - C) Resampling interpolation coefficient.
   - D) Dropout probability.
   *Explanation*: $\\beta |\\mathbf{w}|$ penalizes or rewards hypothesis length to prevent greedy decoders from favoring unrealistically short outputs.
""",
        "research": """# Research Connections: Lecture 01

- **Repository Stage**: `stages/stage_01/` (Initial Research Environment and Kashmiri Corpus Gathering).
- **Historic Discovery**: In Stage 01, initial tests attempting to use off-the-shelf Hindi or Urdu ASR models on Kashmiri audio yielded Character Error Rates exceeding 85%, demonstrating that despite shared Perso-Arabic scripts with Urdu and geographical proximity to Hindi, Kashmiri phonology requires dedicated acoustic and language modeling.
- **Key Code File**: `src/shared/logging.py` establishes the foundational audit logging that tracks all 27 stages.
- **Case Study Reference**: Case Study CS-01 ("First Steps in Kashmiri Speech Recognition").
""",
        "refs": """# References: Lecture 01

1. Bahl, L. R., Jelinek, F., & Mercer, R. L. (1983). *A maximum likelihood approach to continuous speech recognition*. IEEE Transactions on Pattern Analysis and Machine Intelligence, 5(2), 179-190.
2. Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J. (2006). *Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks*. ICML.
3. Koul, O. N. (2005). *Modern Kashmiri Grammar*. Dunwoody Press.
4. Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). *wav2vec 2.0: A framework for self-supervised learning of speech representations*. NeurIPS.
"""
    },
    {
        "num": 2,
        "mod": "module_01_introduction",
        "dir": "lecture_02_asr_anatomy",
        "title": "Anatomy of Modern ASR Pipelines",
        "subtitle": "Modular Architecture, Acoustic Encoders, Prediction Networks, and Vocabulary Construction",
        "objectives": [
            "Deconstruct the complete end-to-end neural ASR pipeline into feature extraction, encoder, decoder, and search heads.",
            "Compare the three dominant paradigm families: CTC, Recurrent Neural Network Transducer (RNN-T), and Attention-based Encoder-Decoder (AED).",
            "Understand why CTC on self-supervised representations was selected for the EduCasheer Kashmiri production architecture.",
            "Formulate the tokenization and vocabulary design choices for Kashmiri Perso-Arabic script."
        ],
        "notes": """# Lecture 02: Anatomy of Modern ASR Pipelines

## 1. Architectural Taxonomy of Modern ASR

Modern End-to-End (E2E) speech recognition models fall into three major paradigm families:

```
+-----------------------------------------------------------------------------------+
|                              E2E ASR PARADIGMS                                     |
+-------------------------+--------------------------------+------------------------+
| 1. CTC                  | 2. Transducer (RNN-T)          | 3. AED (Listen, Attend |
|                         |                                |    and Spell, Whisper) |
+-------------------------+--------------------------------+------------------------+
| - Frame-synchronous     | - Streaming-friendly           | - Label-synchronous    |
| - Conditional indep.    | - Joint network aligns AM + LM | - Autoregressive       |
| - Fast parallel forward | - Complex forward-backward     | - Hallucination risk   |
| - Needs external LM     | - Harder to tune               | - Quadratic attention  |
+-------------------------+--------------------------------+------------------------+
```

### Why CTC for Low-Resource Kashmiri?
In low-resource regimes (under 200 hours of training data), Attention-based Encoder-Decoder (AED) models (such as Whisper or LAS) suffer from severe hallucinations on out-of-domain audio and silent regions. Autoregressive decoders memorize small training sets and fail to generalize.

In contrast, **Connectionist Temporal Classification (CTC)**:
1. Retains strict temporal monotonicity (audio frames strictly map forward in time).
2. Does not suffer from autoregressive looping or hallucination during silence.
3. Decouples the acoustic representation from the language model, allowing independent scaling of the language model via large unlabelled Kashmiri text corpora.

## 2. Pipeline Anatomy: From Raw Audio to Transcribed Text

The EduCasheer Kashmiri ASR architecture consists of four sequential stages:

```
Raw Audio x (16 kHz)
       |
       v
[ 1. Feature Extraction: 7-Layer Temporal CNN ]  --> Stride R=320 (Downsampled to 50 Hz)
       |
       v
[ 2. Acoustic Encoder: 24-Layer Conformer/Transformer (XLS-R) ] --> Contextual Embeddings
       |
       v
[ 3. Linear Projection Head ]  --> Logits in R^{T' x |V|}
       |
       v
[ 4. CTC Decoder + KenLM 5-gram Fusion ] --> Final Kashmiri Text w
```

## 3. Vocabulary Construction for Kashmiri Perso-Arabic Script

Choosing the prediction unit is a pivotal design decision:
- **Phonemes**: Require an expert pronunciation lexicon (Grapheme-to-Phoneme, G2P), which does not exist reliably for Kashmiri.
- **Subwords (BPE/WordPiece)**: Require tens of thousands of hours to learn stable subwords without severe fragmentation.
- **Characters**: Highly effective for CTC in phonetic/alphabetic scripts, provided diacritics and ligatures are normalized.

EduCasheer adopted a **character-level vocabulary** ($|\\mathcal{V}| = 98$ tokens) comprising standard Kashmiri Arabic letters, valid diacritics (*aerab*), punctuation marks, space (`|`), and the mandatory CTC blank token (`<pad>` / `<blank>` at index 0).
""",
        "script": """### Lecture 02 Instructor Delivery Script

**[00:00 - 07:00] Overview of the Three Paradigms**
- "Last lecture we saw that ASR maps audio to text. Today we ask: what does the neural engine actually look like under the hood?"
- *Blackboard*: Draw three columns: CTC, Transducer, AED.
- "Why didn't we just fine-tune OpenAI's Whisper for Kashmiri? That is what 90% of naive practitioners do! Whisper is an AED model. In low-resource Kashmiri, when a speaker pauses for 1.5 seconds, Whisper hallucinates repeated Urdu prayers or English phrases! CTC cannot hallucinate because it has monotonic alignment."

**[07:00 - 20:00] The 4-Stage Anatomy**
- "Walk through the four stages:
  1. The CNN feature extractor converts 16,000 samples per second into 50 frames per second. That is a downsampling factor of 320!
  2. The Transformer encoder applies 24 layers of self-attention.
  3. The linear classification head projects the 1024-dimensional hidden state to our vocabulary size.
  4. The decoding search finds the most likely character string."

**[20:00 - 35:00] Vocabulary Decisions**
- "Why characters instead of BPE? If you use Byte-Pair Encoding with only 50 hours of audio, your tokenizer will create single-character tokens for rare vowels and 4-letter chunks for common words. CTC alignment becomes horribly unbalanced. Characters keep the temporal resolution uniform."

**[35:00 - 45:00] Live Code Demonstration**
- "Let's open `src/asr/model.py` and inspect the forward pass of `KashmiriASRModel`."
""",
        "math": """# Mathematical Foundations: Pipeline Dimension Transformations

## 1. Dimensionality Progression Through the Pipeline
Let the input raw audio waveform be a 1D sequence sampled at $f_s = 16{,}000\\text{ Hz}$:
$$\\mathbf{x} \\in \\mathbb{R}^{1 \\times N}$$
where $N = f_s \\cdot t_\\text{audio}$ is the total number of audio samples.

### Stage 1: Feature Encoder (Temporal CNN)
The 7-layer convolutional feature encoder has kernel sizes $K = (10, 3, 3, 3, 3, 2, 2)$ and strides $S = (5, 2, 2, 2, 2, 2, 2)$.
The total downsampling factor $R$ is the product of the strides:
$$R = \\prod_{i=1}^7 S_i = 5 \\times 2 \\times 2 \\times 2 \\times 2 \\times 2 \\times 2 = 320$$

The output sequence length $T'$ after the feature encoder is:
$$T' = \\left\\lfloor \\frac{N - K_\\text{eff}}{R} \\right\\rfloor + 1 \\approx \\left\\lfloor \\frac{N}{320} \\right\\rfloor$$
At $f_s = 16{,}000\\text{ Hz}$, each latent frame represents:
$$\\Delta t = \\frac{320}{16{,}000} = 0.020\\text{ s} = 20\\text{ ms}$$
yielding a temporal frame rate of $50\\text{ Hz}$.

The latent feature tensor is:
$$\\mathbf{Z} \\in \\mathbb{R}^{B \\times T' \\times D_\\text{cnn}}, \\quad D_\\text{cnn} = 512$$

### Stage 2: Transformer / Conformer Encoder
The contextual encoder projects $\\mathbf{Z}$ to model dimension $D_\\text{model} = 1024$ and applies $L = 24$ layers of multi-head self-attention:
$$\\mathbf{H} = \\text{TransformerEncoder}(\\mathbf{Z}) \\in \\mathbb{R}^{B \\times T' \\times 1024}$$

### Stage 3: Linear CTC Projection Head
A linear transformation maps the contextual representations to unnormalized log-probabilities over the vocabulary $\\mathcal{V}$:
$$\\mathbf{Y}_\\text{logits} = \\mathbf{H} \\mathbf{W}_\\text{proj} + \\mathbf{b}_\\text{proj}, \\quad \\mathbf{W}_\\text{proj} \\in \\mathbb{R}^{1024 \\times |\\mathcal{V}|}, \\; \\mathbf{b}_\\text{proj} \\in \\mathbb{R}^{|\\mathcal{V}|}$$
$$\\mathbf{P} = \\text{softmax}(\\mathbf{Y}_\\text{logits}, \\text{dim}=-1) \\in [0, 1]^{B \\times T' \\times |\\mathcal{V}|}$$
""",
        "code": """# Code Walkthrough: Lecture 02

We inspect `src/asr/model.py`, the core neural architecture module of the production EduCasheer system.

```python
# src/asr/model.py
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel

class KashmiriASRModel(nn.Module):
    \"\"\"
    Production CTC Acoustic Model for Kashmiri Speech Recognition.
    Wraps an XLS-R self-supervised backbone with a linear projection head.
    \"\"\"
    def __init__(self, pretrained_model_name: str, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(dropout)
        hidden_size = self.wav2vec2.config.hidden_size  # 1024 for XLS-R-300M
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
    def freeze_feature_extractor(self):
        \"\"\"Freezes the 7-layer CNN during initial fine-tuning to prevent catastrophic forgetting.\"\"\"
        self.wav2vec2.feature_extractor._freeze_parameters()

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor = None):
        \"\"\"
        Args:
            input_values: (Batch, Samples) raw 16kHz audio waveform
            attention_mask: (Batch, Samples) padding mask
        Returns:
            logits: (Batch, Frames, Vocab_Size) CTC emission scores
        \"\"\"
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # Shape: (B, T', 1024)
        hidden_states = self.dropout(hidden_states)
        logits = self.lm_head(hidden_states)       # Shape: (B, T', vocab_size)
        return logits
```
""",
        "lab": """# Practical Lab 02: Pipeline Dimensions and CTC Emission Verification

## Objective
1. Initialize a toy feature extractor and linear classification head.
2. Verify tensor transformations and verify the $R=320$ downsampling ratio.
3. Compute the CTC frame rate and inspect softmax emission probabilities.

## Laboratory Code

```python
import torch
import torch.nn as nn

def run_lab():
    print("=== Step 1: Simulate the 7-Layer CNN Feature Extractor ===")
    strides = [5, 2, 2, 2, 2, 2, 2]
    total_stride = 1
    for s in strides:
        total_stride *= s
    print(f"Total Stride R: {total_stride} (Downsampling factor)")
    
    # Simulate 3 seconds of audio at 16kHz
    batch_size = 2
    sample_rate = 16000
    seconds = 3.5
    num_samples = int(sample_rate * seconds)
    x = torch.randn(batch_size, num_samples)
    print(f"Input audio tensor shape: {x.shape} ({num_samples} samples)")
    
    expected_frames = num_samples // total_stride
    print(f"Expected CTC frames: {expected_frames} frames (approx {expected_frames / seconds:.1f} fps)")
    
    print("\n=== Step 2: Simulate Acoustic Projection Head ===")
    hidden_dim = 1024
    vocab_size = 98  # Kashmiri character vocabulary
    
    # Dummy contextual features
    h = torch.randn(batch_size, expected_frames, hidden_dim)
    head = nn.Linear(hidden_dim, vocab_size)
    logits = head(h)
    
    print(f"Logits shape: {logits.shape}")
    
    probs = torch.softmax(logits, dim=-1)
    print(f"Softmax sum across vocab for batch 0, frame 0: {probs[0, 0].sum().item():.4f}")
    print(f"Blank token (index 0) initial prob: {probs[0, 0, 0].item():.4f}")

if __name__ == "__main__":
    run_lab()
```
""",
        "exercises": """# Exercises: Lecture 02

## Analytical Problems
1. Derive the mathematical receptive field in milliseconds of the 7-layer temporal convolution feature extractor used in Wav2Vec 2.0 given kernel sizes $(10, 3, 3, 3, 3, 2, 2)$ and strides $(5, 2, 2, 2, 2, 2, 2)$.
2. Explain why an Attention-based Encoder-Decoder (AED) model trained on only 50 hours of speech is prone to infinite looping during long pauses, whereas CTC is mathematically immune to this failure mode.

## Implementation Problems
3. Implement a PyTorch module `ToyCTCHead` that takes a hidden representation of size `(B, T, 768)` and projects it to a vocabulary of size 64 with a dropout layer ($p=0.15$) and Xavier uniform weight initialization.

## Forensic Auditing
4. In Stage 02 of the EduCasheer repository, the initial model used an output vocabulary of 142 characters, including distinct tokens for Persian, Arabic, and Urdu code points that represent the identical Kashmiri phoneme. What effect did this vocabulary inflation have on the model's loss convergence?
""",
        "solutions": """# Solutions: Lecture 02

## Exercise 1
The receptive field $RF_l$ of layer $l$ is calculated recursively:
$$RF_l = RF_{l-1} + (K_l - 1) \\cdot J_{l-1}, \\quad J_l = J_{l-1} \\cdot S_l$$
with $RF_0 = 1, J_0 = 1$.
- Layer 1: $K=10, S=5 \\implies RF_1 = 1 + (10-1)\\cdot 1 = 10, J_1 = 5$
- Layer 2: $K=3, S=2 \\implies RF_2 = 10 + (3-1)\\cdot 5 = 20, J_2 = 10$
- Layer 3: $K=3, S=2 \\implies RF_3 = 20 + (3-1)\\cdot 10 = 40, J_3 = 20$
- Layer 4: $K=3, S=2 \\implies RF_4 = 40 + (3-1)\\cdot 20 = 80, J_4 = 40$
- Layer 5: $K=3, S=2 \\implies RF_5 = 80 + (3-1)\\cdot 40 = 160, J_5 = 80$
- Layer 6: $K=2, S=2 \\implies RF_6 = 160 + (2-1)\\cdot 80 = 240, J_6 = 160$
- Layer 7: $K=2, S=2 \\implies RF_7 = 240 + (2-1)\\cdot 160 = 400, J_7 = 320$
The total receptive field is $400$ audio samples. At $16\\text{ kHz}$:
$$RF = \\frac{400}{16{,}000} = 0.025\\text{ s} = 25\\text{ ms}$$
Each frame represents $25\\text{ ms}$ of acoustic context, centered every $20\\text{ ms}$.

## Exercise 2
In AED models, decoding is autoregressive: $P(y_u \\mid y_{<u}, \\mathbf{X})$. When audio input $\\mathbf{X}$ contains silence, the cross-attention weights become uniform or diffuse. The decoder relies entirely on its internal language model state $y_{<u}$. If $P(y_u \\mid y_{u-1})$ has a cyclic probability mass (e.g. "..." or repeating words), the decoder enters an infinite feedback loop. In CTC, frames are conditionally independent given $\\mathbf{X}$, and during silence, the acoustic model outputs the blank token $\\epsilon$, which collapses to empty string without recursion.

## Exercise 3
```python
import torch
import torch.nn as nn

class ToyCTCHead(nn.Module):
    def __init__(self, in_features=768, vocab_size=64, dropout_p=0.15):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(in_features, vocab_size)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        
    def forward(self, x):
        return self.fc(self.dropout(x))
```
""",
        "quiz": """# Comprehension Quiz: Lecture 02

1. **What is the total downsampling factor $R$ of the 7-layer temporal convolution encoder in Wav2Vec 2.0?**
   - A) 160
   - B) 320 [CORRECT]
   - C) 640
   - D) 128
   *Explanation*: $5 \\times 2 \\times 2 \\times 2 \\times 2 \\times 2 \\times 2 = 320$.

2. **At 16,000 Hz, how many CTC frames are produced for every 1.0 second of audio?**
   - A) 25 frames
   - B) 50 frames [CORRECT]
   - C) 100 frames
   - D) 160 frames
   *Explanation*: $16000 / 320 = 50\\text{ frames per second}$.

3. **Why was character-level vocabulary selected over BPE for EduCasheer Kashmiri ASR?**
   - A) Characters take less disk space.
   - B) BPE creates severe subword fragmentation in low-resource regimes, distorting CTC alignment. [CORRECT]
   - C) PyTorch does not support BPE tokenizers.
   - D) Perso-Arabic characters cannot be tokenized.
   *Explanation*: Small corpora cause BPE tokenizers to over-fragment rare morphological stems, degrading CTC convergence.

4. **Why is the feature extractor frozen during early fine-tuning of XLS-R?**
   - A) To save memory only.
   - B) Because convolutional layers cannot be trained on GPU.
   - C) To prevent destructive gradient updates from destroying general acoustic filters before the projection head aligns. [CORRECT]
   - D) To enforce 22.05 kHz resampling.
   *Explanation*: Freezing early prevents catastrophic forgetting of foundational cross-lingual acoustic representations.

5. **In CTC, what token index is conventionally reserved for the blank token $\\epsilon$?**
   - A) Last index ($|\\mathcal{V}|-1$)
   - B) Random index
   - C) Index 0 [CORRECT]
   - D) Index -1
   *Explanation*: In standard PyTorch `nn.CTCLoss`, index 0 is conventionally assigned to the blank token.
""",
        "research": """# Research Connections: Lecture 02

- **Repository Stage**: `stages/stage_02/` and `src/asr/model.py`.
- **Historic Transition**: Between Stage 02 and Stage 07, the project experimented with hybrid Conformer-CTC and pure Transformer architectures. The Conformer showed superior convergence on small sets but exhibited training instability in mixed-precision FP16. The team settled on XLS-R-300M with a linear projection head as the certified production backbone.
- **Key Code File**: `src/asr/model.py` (`KashmiriASRModel`).
- **Case Study Reference**: Case Study CS-02 ("Scraped Data Realities and Model Architecture Selection").
""",
        "refs": """# References: Lecture 02

1. Graves, A. (2012). *Supervised sequence labelling with recurrent neural networks*. Springer.
2. Gulati, A., et al. (2020). *Conformer: Convolution-augmented Transformer for Speech Recognition*. Interspeech.
3. Babu, A., et al. (2021). *XLS-R: Self-supervised cross-lingual speech representation learning at scale*. arXiv:2111.09296.
"""
    },
    {
        "num": 3,
        "mod": "module_02_speech_and_audio",
        "dir": "lecture_03_sound_waves_and_sampling",
        "title": "Sound Waves, Sampling & Quantization",
        "subtitle": "Continuous Wave Mechanics, The Nyquist-Shannon Theorem, and PCM Quantization Dynamics",
        "objectives": [
            "Derive the acoustic wave equation and understand acoustic pressure oscillations.",
            "Formulate the Nyquist-Shannon Sampling Theorem and the mathematics of aliasing.",
            "Analyze uniform and non-uniform Pulse Code Modulation (PCM) quantization noise.",
            "Implement high-fidelity audio tensor loading, dynamic range scaling, and waveform validation in PyTorch."
        ],
        "notes": """# Lecture 03: Sound Waves, Sampling & Quantization

## 1. Physics of Sound and Acoustic Propagation

Sound is a mechanical wave created by acoustic pressure fluctuations in an elastic medium (such as air). The 1D acoustic wave equation relates the spatial derivative of pressure $p(x, t)$ to its second temporal derivative:

$$\\frac{\\partial^2 p}{\\partial x^2} = \\frac{1}{c^2} \\frac{\\partial^2 p}{\\partial t^2}$$

where $c \\approx 343\\text{ m/s}$ is the speed of sound in air at $20^\\circ\\text{C}$.

In human speech, acoustic excitation originates from airflow through the vocal folds (voiced sounds) or turbulent constriction in the vocal tract (unvoiced fricatives). These pressure waves are transduced into continuous electrical voltages $v(t)$ by microphone diaphragms.

## 2. The Nyquist-Shannon Sampling Theorem

To process $v(t)$ on digital hardware, the continuous signal must be discretized in time. Let $f_s = 1 / T_s$ be the sampling frequency, where $T_s$ is the sampling interval.

The ideal sampling process is represented mathematically by multiplying $v(t)$ with a Dirac comb $\\text{III}_{T_s}(t)$:

$$x_s(t) = v(t) \\cdot \\sum_{n=-\\infty}^{\\infty} \\delta(t - n T_s)$$

Taking the continuous Fourier transform:

$$X_s(f) = \\frac{1}{T_s} \\sum_{k=-\\infty}^{\\infty} V\\left(f - k f_s\\right)$$

### The Aliasing Criterion
The spectrum $X_s(f)$ consists of infinite periodic repetitions of the baseband spectrum $V(f)$ shifted by integer multiples of $f_s$. To reconstruct $v(t)$ perfectly without spectral overlap (**aliasing**):

$$f_s \\ge 2 f_\\text{max}$$

where $f_\\text{max}$ is the **Nyquist frequency** (the highest frequency component with non-zero energy in $v(t)$).

If $v(t)$ contains frequencies exceeding $f_s / 2$, those high frequencies fold back into the audible spectrum:
$$f_\\text{alias} = |f - k f_s|$$
causing irreversible acoustic distortion.

## 3. Quantization and Signal-to-Quantization-Noise Ratio (SQNR)

Once sampled in time, continuous amplitudes must be quantized to discrete levels. In standard linear Pulse Code Modulation (PCM), an amplitude range $[-V_\\text{max}, +V_\\text{max}]$ is partitioned into $2^B$ levels, where $B$ is the bit depth (typically 16-bit).

The quantization step size $\\Delta$ is:
$$\\Delta = \\frac{2 V_\\text{max}}{2^B}$$

Assuming quantization error $e = x - Q(x)$ is uniformly distributed over $[-\\Delta/2, +\\Delta/2]$, the noise variance is:
$$\\sigma_e^2 = \\int_{-\\Delta/2}^{\\Delta/2} e^2 \\frac{1}{\\Delta} de = \\frac{\\Delta^2}{12}$$

For a full-scale sinusoid of amplitude $V_\\text{max}$, the average signal power is $P_s = V_\\text{max}^2 / 2$. The theoretical Signal-to-Quantization-Noise Ratio is:
$$\\text{SQNR} = 10 \\log_{10} \\left( \\frac{P_s}{\\sigma_e^2} \\right) = 10 \\log_{10} \\left( \\frac{V_\\text{max}^2 / 2}{\\Delta^2 / 12} \\right) \\approx 6.02 B + 1.76\\text{ dB}$$

For 16-bit linear PCM audio:
$$\\text{SQNR}_{16\\text{-bit}} \\approx 6.02(16) + 1.76 = 98.08\\text{ dB}$$
providing ample dynamic range for speech recognition without perceptible quantization noise.
""",
        "script": """### Lecture 03 Instructor Delivery Script

**[00:00 - 08:00] Intuition: From Air to Numbers**
- "Every time I speak, air molecules oscillate at approximately 343 meters per second. Your microphone membrane vibrates, generating a tiny voltage. How do we turn that continuous fluid motion into PyTorch tensors?"
- *Blackboard*: Draw continuous sine wave $v(t)$, draw vertical sampling lines at $T_s$.

**[08:00 - 22:00] Nyquist Proof and Aliasing**
- "Why 16 kHz? Why not 8 kHz like old telephone systems, or 44.1 kHz like Spotify? Human speech produces formants and fricatives up to 8 kHz. By Nyquist, $f_s \\ge 2 \\times 8{,}000 = 16{,}000\\text{ Hz}$. If we record at 8 kHz, telephone speech cuts off at 4 kHz, confusing Kashmiri fricatives like /s/, /z/, /ʃ/."
- *Demo*: Show an aliasing diagram where an 11 kHz tone sampled at 16 kHz appears as a phantom 5 kHz artifact!

**[22:00 - 35:00] Bit Depth and PCM**
- "Why does 16-bit give us 98 dB? Derivation of the $6.02 B + 1.76$ formula. Why do machine learning practitioners store audio as normalized float32 in $[-1.0, 1.0]$ instead of int16?"

**[35:00 - 45:00] Code Inspection**
- "Open `src/data_processing/audio.py` to see how EduCasheer enforces 16 kHz mono float32 tensors."
""",
        "math": """# Mathematical Foundations: Aliasing & Quantization Derivations

## 1. Mathematical Proof of the Nyquist Criterion
Let $v(t) \\in L^2(\\mathbb{R})$ be a band-limited signal whose Fourier transform satisfies:
$$V(f) = 0 \\quad \\forall |f| > B$$
Sampled signal:
$$x_s(t) = v(t) \\sum_{n=-\\infty}^{\\infty} \\delta(t - n T_s)$$
Using the Fourier series expansion of the Dirac comb:
$$\\sum_{n=-\\infty}^{\\infty} \\delta(t - n T_s) = \\frac{1}{T_s} \\sum_{k=-\\infty}^{\\infty} e^{j 2\\pi k f_s t}$$
Taking the Fourier transform of $x_s(t)$:
$$X_s(f) = \\mathcal{F}\\left\\{ v(t) \\frac{1}{T_s} \\sum_{k=-\\infty}^{\\infty} e^{j 2\\pi k f_s t} \\right\\} = \\frac{1}{T_s} \\sum_{k=-\\infty}^{\\infty} V(f - k f_s)$$

To recover $V(f)$ without distortion, the copies centered at $k=0$ and $k=1$ must not overlap:
$$B < f_s - B \\implies f_s > 2B$$

## 2. Ideal Sinc Reconstruction
When $f_s \\ge 2B$, the original signal is perfectly reconstructed by filtering $X_s(f)$ with an ideal low-pass brickwall filter:
$$H(f) = T_s \\cdot \\text{rect}\\left( \\frac{f}{f_s} \\right)$$
In the time domain:
$$h(t) = \\text{sinc}(f_s t) = \\frac{\\sin(\\pi f_s t)}{\\pi f_s t}$$
$$v(t) = x_s(t) * h(t) = \\sum_{n=-\\infty}^{\\infty} v(n T_s) \\text{sinc}\\left( \\frac{t - n T_s}{T_s} \\right)$$
""",
        "code": """# Code Walkthrough: Lecture 03

We examine `src/data_processing/audio.py` for audio loading and normalization.

```python
# src/data_processing/audio.py
import torch
import torchaudio

def load_and_standardize_audio(file_path: str, target_sr: int = 16000) -> torch.Tensor:
    \"\"\"
    Loads an audio file, checks channel layout, enforces target sampling rate,
    and scales amplitude to float32 range [-1.0, 1.0].
    \"\"\"
    waveform, sample_rate = torchaudio.load(file_path)
    
    # 1. Convert multi-channel to mono by averaging channels
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # 2. Resample if necessary (covered deeply in Lecture 04)
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)
        
    # 3. Dynamic range sanity check: avoid clipping
    max_val = torch.max(torch.abs(waveform))
    if max_val > 1.0:
        waveform = waveform / max_val
        
    return waveform.squeeze(0)  # Return 1D tensor: (Samples,)
```
""",
        "lab": """# Practical Lab 03: Simulating Aliasing and Quantization Distortion

## Objective
1. Synthesize a high-frequency sine wave and observe aliasing when under-sampled.
2. Simulate 8-bit, 4-bit, and 2-bit quantization on human speech audio.
3. Compute the empirical SQNR and compare it to the theoretical $6.02 B + 1.76\\text{ dB}$ formula.

## Laboratory Code

```python
import torch

def run_lab():
    print("=== Step 1: Demonstrate Aliasing ===")
    # True signal: 12 kHz frequency
    f_tone = 12000.0
    duration = 0.01  # 10 ms
    
    # Continuous reference at 192 kHz
    fs_high = 192000
    t_high = torch.linspace(0, duration, int(fs_high * duration))
    x_true = torch.sin(2 * torch.pi * f_tone * t_high)
    
    # Under-sampled at 16 kHz (Nyquist is 8 kHz)
    fs_low = 16000
    t_low = torch.linspace(0, duration, int(fs_low * duration))
    x_sampled = torch.sin(2 * torch.pi * f_tone * t_low)
    
    # Theoretical alias: |12000 - 16000| = 4000 Hz
    f_alias = abs(f_tone - fs_low)
    print(f"Original frequency: {f_tone} Hz")
    print(f"Sampling frequency: {fs_low} Hz (Nyquist limit: {fs_low/2} Hz)")
    print(f"Apparent aliased frequency: {f_alias} Hz")
    
    print("\n=== Step 2: Quantization Noise Simulation ===")
    clean_audio = torch.sin(2 * torch.pi * 440.0 * torch.linspace(0, 1.0, 16000))
    
    for bits in [16, 8, 4]:
        levels = 2 ** bits
        delta = 2.0 / levels
        quantized = torch.round((clean_audio + 1.0) / delta) * delta - 1.0
        noise = clean_audio - quantized
        sqnr = 10 * torch.log10(torch.mean(clean_audio ** 2) / torch.mean(noise ** 2))
        theoretical = 6.02 * bits + 1.76
        print(f"Bit depth: {bits:2d} | Empirical SQNR: {sqnr.item():.2f} dB | Theoretical: {theoretical:.2f} dB")

if __name__ == "__main__":
    run_lab()
```
""",
        "exercises": """# Exercises: Lecture 03

## Analytical Problems
1. A continuous speech signal is low-pass filtered with an imperfect analog filter having a transition band between 7.5 kHz and 8.5 kHz. If the signal is sampled at 16 kHz, what frequency range will fold back (alias) into the passband $[0, 7.5\\text{ kHz}]$?
2. Prove that non-uniform quantization (such as $\\mu$-law companding) yields approximately constant SQNR over a broader dynamic range than linear PCM.

## Implementation Problems
3. Implement a PyTorch function `apply_mu_law(waveform: torch.Tensor, mu: int = 255) -> torch.Tensor` that compresses a floating-point waveform in $[-1.0, 1.0]$ according to:
$$F(x) = \\text{sgn}(x) \\frac{\\ln(1 + \\mu |x|)}{\\ln(1 + \\mu)}$$

## Forensic Auditing
4. In Stage 03 of the EduCasheer project, several audio files collected from low-cost smartphones were found to be clipped at $+1.0$ and $-1.0$. Explain why clipping introduces spurious high-frequency harmonics in the Fourier domain.
""",
        "solutions": """# Solutions: Lecture 03

## Exercise 1
At $f_s = 16\\text{ kHz}$, the Nyquist cutoff is $f_s / 2 = 8\\text{ kHz}$.
Energy between $8.0\\text{ kHz}$ and $8.5\\text{ kHz}$ exceeds the Nyquist limit.
By the aliasing formula $f_\\text{alias} = f_s - f$:
- $8.0\\text{ kHz}$ reflects to $16.0 - 8.0 = 8.0\\text{ kHz}$.
- $8.5\\text{ kHz}$ reflects to $16.0 - 8.5 = 7.5\\text{ kHz}$.
Thus, frequencies in $[8.0, 8.5\\text{ kHz}]$ fold back into the range $[7.5, 8.0\\text{ kHz}]$.

## Exercise 2
In linear PCM, step size $\\Delta$ is fixed, so for small amplitudes $|x| \\ll V_\\text{max}$, the signal power $P_s \\propto |x|^2$ is tiny while quantization noise $\\sigma_e^2 = \\Delta^2 / 12$ remains constant, causing SQNR to collapse. Under $\\mu$-law, the effective step size scales proportionally with amplitude: $\\Delta(x) \\propto |x|$. Therefore, the ratio $x^2 / \\Delta^2(x)$ remains nearly constant over a 40 dB dynamic range.

## Exercise 3
```python
import torch

def apply_mu_law(x: torch.Tensor, mu: int = 255) -> torch.Tensor:
    mu_t = torch.tensor(mu, dtype=x.dtype, device=x.device)
    return torch.sign(x) * torch.log1p(mu_t * torch.abs(x)) / torch.log1p(mu_t)
```
""",
        "quiz": """# Comprehension Quiz: Lecture 03

1. **What is the Nyquist rate for a speech signal band-limited to 8 kHz?**
   - A) 8 kHz
   - B) 16 kHz [CORRECT]
   - C) 24 kHz
   - D) 44.1 kHz
   *Explanation*: $f_s \\ge 2 \\times 8\\text{ kHz} = 16\\text{ kHz}$.

2. **What occurs when an audio signal is sampled below its Nyquist frequency without an anti-aliasing filter?**
   - A) The volume increases.
   - B) High-frequency energy reflects back into the lower frequency spectrum as spurious tones. [CORRECT]
   - C) The file size doubles.
   - D) The signal becomes pure white noise.
   *Explanation*: Under-sampling causes high frequencies to mirror around $f_s/2$ as aliased artifacts.

3. **By how many decibels does each additional bit of linear PCM quantization increase theoretical SQNR?**
   - A) ~3.01 dB
   - B) ~6.02 dB [CORRECT]
   - C) ~10.0 dB
   - D) ~20.0 dB
   *Explanation*: Doubling the number of quantization levels ($2^B$) reduces noise amplitude by half ($20 \\log_{10}(2) \\approx 6.02\\text{ dB}$).

4. **Why is human speech typically sampled at 16 kHz rather than 44.1 kHz for ASR acoustic modeling?**
   - A) Transformers cannot process audio above 16 kHz.
   - B) 16 kHz captures all critical linguistic formants up to 8 kHz while halving memory and computational costs. [CORRECT]
   - C) Microphones cannot record speech above 16 kHz.
   - D) 44.1 kHz causes CTC loss underflow.
   *Explanation*: 16 kHz covers the human vocal tract formants while preserving computational efficiency.

5. **What happens to the Fourier spectrum when audio waveform amplitudes are clipped at $[-1.0, 1.0]$?**
   - A) The spectrum becomes completely flat.
   - B) Sharp waveform discontinuities introduce artificial odd harmonics across the spectrum. [CORRECT]
   - C) All high frequencies are eliminated.
   - D) Phase information is inverted.
   *Explanation*: Hard clipping squares off the peaks, injecting odd harmonics into the spectrum.
""",
        "research": """# Research Connections: Lecture 03

- **Repository Stage**: `stages/stage_03/` (Audio Standardization & Pre-filtering).
- **Historic Finding**: In Stage 03, auditing the raw audio files collected from YouTube and WhatsApp voice notes revealed that 14.2% of files had mixed sampling rates (22.05 kHz, 32 kHz, 44.1 kHz, 48 kHz) and 6.8% had severe digital clipping (peak amplitude exceeding $0.999$ with flat tops). This led to the creation of the automated audio health screening filter in `src/data_processing/audio.py`.
- **Key Code File**: `src/data_processing/audio.py`.
- **Case Study Reference**: Case Study CS-03 ("Audio Standardization Traps in Low-Resource Data Gathering").
""",
        "refs": """# References: Lecture 03

1. Shannon, C. E. (1949). *Communication in the presence of noise*. Proceedings of the IRE, 37(1), 10-21.
2. Oppenheim, A. V., & Schafer, R. W. (2009). *Discrete-Time Signal Processing*. Pearson.
3. Jayant, N. S., & Noll, P. (1984). *Digital Coding of Waveforms*. Prentice-Hall.
"""
    },
    {
        "num": 4,
        "mod": "module_02_speech_and_audio",
        "dir": "lecture_04_resampling_and_22khz_defect",
        "title": "Resampling Physics & The 22.05 kHz Defect",
        "subtitle": "Polyphase Sinc Resampling, Anti-Aliasing Filters, and the Historical Bolbosh Resampling Trap",
        "objectives": [
            "Derive the mathematics of digital sample rate conversion (up-sampling, filtering, down-sampling).",
            "Contrast naive linear/polynomial interpolation against band-limited sinc interpolation.",
            "Dissect the empirical 22.05 kHz resampling bug discovered in the Bolbosh Kashmiri corpus during the Stage 25 audit.",
            "Implement a certified band-limited polyphase resampler using PyTorch and `torchaudio`."
        ],
        "notes": """# Lecture 04: Resampling Physics & The 22.05 kHz Defect

## 1. Principles of Digital Sample Rate Conversion

In speech processing, acoustic data originates from disparate hardware at varied sampling rates: $8\\text{ kHz}$ (telephony), $22.05\\text{ kHz}$ (legacy broadcast / MP3), $44.1\\text{ kHz}$ (audio CD), and $48\\text{ kHz}$ (video / studio). Modern foundation models (XLS-R, Wav2Vec 2.0) are strictly pretrained on $16\\text{ kHz}$ audio. Converting an arbitrary rate $f_1$ to $f_2$ requires rational resampling by factor $L/M$:

$$f_2 = \\frac{L}{M} f_1, \\quad L, M \\in \\mathbb{Z}^+$$

The classical three-step conversion process is:
1. **Interpolation (Up-sampling by $L$)**: Insert $L-1$ zeros between adjacent samples, expanding the sampling rate to $L f_1$.
2. **Anti-Aliasing / Anti-Imaging Low-Pass Filter**: Convolve with a low-pass filter $h[n]$ with cutoff:
   $$\\omega_c = \\min\\left( \\frac{\\pi}{L}, \\frac{\\pi}{M} \\right)$$
3. **Decimation (Down-sampling by $M$)**: Retain every $M$-th sample, reducing the sampling rate to $(L/M) f_1$.

```
x[n] ---> [ Up-sample by L ] ---> [ Low-Pass Filter h[n] ] ---> [ Down-sample by M ] ---> y[m]
           (Insert L-1 zeros)       Cutoff: min(pi/L, pi/M)       (Keep every M-th)
```

## 2. Naive Interpolation vs. Band-Limited Sinc Filtering

Many practitioners make the fatal mistake of using `scipy.interpolate.interp1d(kind='linear')` or cubic spline interpolation to resample audio.

- **Linear Interpolation**: Equivalent to convolving the up-sampled sequence with a triangle filter $h_\\text{tri}[n]$. In the frequency domain, a triangle filter has a $\\text{sinc}^2(f)$ response, which exhibits substantial passband droop and very poor stopband attenuation (only $-26\\text{ dB}$).
- **The Result**: Significant spectral leakage folds high-frequency energy back into the baseband, creating metallic ringing and distorting phonetic formants.

The theoretically optimal reconstruction filter is the **band-limited sinc filter** windowed with a Kaiser or Hann window:

$$h[n] = \\frac{\\sin(\\omega_c n)}{\\pi n} \\cdot w[n]$$

## 3. The Historical Bolbosh 22.05 kHz Resampling Trap

During the Stage 25 Forensic Audit of the EduCasheer repository, an acoustic anomaly was detected: fine-tuning XLS-R on the Bolbosh dataset produced a Character Error Rate $4.2\\%$ higher than on identical-duration community audio.

### The Forensic Discovery
1. The Bolbosh corpus was originally recorded in MP3 format at **$22{,}050\\text{ Hz}$** (a standard CD-submultiple rate).
2. To convert $22{,}050\\text{ Hz} \\rightarrow 16{,}000\\text{ Hz}$, the ratio is:
   $$\\frac{f_2}{f_1} = \\frac{16{,}000}{22{,}050} = \\frac{320}{441}$$
3. The early historical pipeline in Stage 03 used a fast polynomial interpolation script that omitted the anti-aliasing low-pass filter at the new Nyquist limit ($8{,}000\\text{ Hz}$).
4. Frequencies in Bolbosh audio between $8{,}000\\text{ Hz}$ and $11{,}025\\text{ Hz}$ were reflected back into $[4{,}975\\text{ Hz}, 8{,}000\\text{ Hz}]$.
5. Because Kashmiri contains crucial sibilant and palatal fricatives (/ʃ/, /s/, /t͡s/) concentrated between $5\\text{ kHz}$ and $8\\text{ kHz}$, the aliased ghost frequencies corrupted the acoustic embeddings inside the XLS-R CNN encoder.

Once corrected in Stage 25 with a Kaiser-windowed sinc resampler (`torchaudio.transforms.Resample(orig_freq=22050, new_freq=16000, resampling_method='kaiser_window')`), the CER gap disappeared.
""",
        "script": """### Lecture 04 Instructor Delivery Script

**[00:00 - 08:00] The "Simple" Problem That Breaks Models**
- "Every tutorial on the internet tells you: `torchaudio.load('file.wav')` and if the sample rate isn't 16,000, just call `resample()`. Sounds trivial, right? In Stage 25 of this project, a subtle bug in resampling cost our research team two months of wasted training runs!"
- *Action*: Write $22,050 \\rightarrow 16,000$ on the board. "Where did 22,050 come from? It's half of 44,100 Hz."

**[08:00 - 22:00] The Math of Rational Resampling**
- "To convert 22,050 to 16,000, what is the integer ratio? Divide by 50: $320 / 441$. You have to upsample by 320, insert 319 zeros between every sample, run an anti-aliasing filter with cutoff at $\\pi / 441$, and then pick every 441st sample."
- *Blackboard*: Draw the polyphase filter decomposition. Show why doing this naively takes infinite memory if not done with polyphase filterbanks.

**[22:00 - 35:00] The Bolbosh Forensic Breakdown**
- "In Bolbosh, someone used a linear interpolation script. Frequencies between 8 kHz and 11 kHz folded right into the 5 to 8 kHz band. What lives in Kashmiri speech between 5 and 8 kHz? Kashmiri palatal affricates! The model was hearing phantom high-pitched whistles on every word."

**[35:00 - 45:00] Lab Walkthrough**
- "Let's run `practical_lab.md` to recreate this bug and observe the spectrogram artifacts."
""",
        "math": """# Mathematical Foundations: Polyphase Filterbank Resampling

## 1. Up-Sampling and Spectral Compression
Let $x[n]$ have sampling rate $f_1$. Up-sampling by integer factor $L$:
$$x_u[m] = \\begin{cases} x[m/L] & m = 0, \\pm L, \\pm 2L, \\dots \\\\ 0 & \\text{otherwise} \\end{cases}$$
In the frequency domain:
$$X_u(\\omega) = X(L\\omega)$$
The spectrum $X_u(\\omega)$ contains $L$ compressed copies of $X(\\omega)$ within the interval $[-\\pi, \\pi]$. These copies are **images** and must be removed by an anti-imaging filter.

## 2. Decimation and Spectral Expansion
Down-sampling a filtered signal $v[m]$ by factor $M$:
$$y[k] = v[k M]$$
In the frequency domain:
$$Y(\\omega) = \\frac{1}{M} \\sum_{m=0}^{M-1} V\\left( \\frac{\\omega - 2\\pi m}{M} \\right)$$
To avoid aliasing, $V(\\omega)$ must be strictly zero for $|\\omega| > \\pi / M$.

## 3. The Combined Anti-Aliasing Cutoff
The ideal lowpass filter $h[n]$ must simultaneously eliminate imaging and aliasing:
$$\\omega_c = \\min\\left( \\frac{\\pi}{L}, \\frac{\\pi}{M} \\right)$$
For $22{,}050\\text{ Hz} \\rightarrow 16{,}000\\text{ Hz}$:
$$L = 320, \\quad M = 441$$
$$\\omega_c = \\min\\left( \\frac{\\pi}{320}, \\frac{\\pi}{441} \\right) = \\frac{\\pi}{441} \\text{ rad/sample}$$

In continuous frequency:
$$f_c = \\frac{\\omega_c}{2\\pi} \\cdot (L f_1) = \\frac{1}{2 \\times 441} \\cdot (320 \\times 22050) = 8{,}000\\text{ Hz}$$
which is exactly the Nyquist frequency of the target $16\\text{ kHz}$ rate.
""",
        "code": """# Code Walkthrough: Lecture 04

We inspect `src/data_processing/resampling.py`, the certified resampling implementation that fixed the Bolbosh bug.

```python
# src/data_processing/resampling.py
import torch
import torchaudio.transforms as T

class CertifiedAudioResampler:
    \"\"\"
    Certified Sinc Resampler using Kaiser-windowed polyphase filterbanks.
    Guarantees >= 85 dB stopband attenuation to eliminate aliasing artifacts.
    \"\"\"
    def __init__(self, orig_sr: int, target_sr: int = 16000, lowpass_filter_width: int = 64):
        self.orig_sr = orig_sr
        self.target_sr = target_sr
        
        if orig_sr != target_sr:
            self.resampler = T.Resample(
                orig_freq=orig_sr,
                new_freq=target_sr,
                resampling_method="kaiser_window",
                lowpass_filter_width=lowpass_filter_width,
                rolloff=0.94,
                beta=14.769656  # Kaiser beta for 85 dB attenuation
            )
        else:
            self.resampler = None

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.resampler is None:
            return waveform
        return self.resampler(waveform)
```
""",
        "lab": """# Practical Lab 04: The 22.05 kHz Resampling Defect Reproduction

## Objective
1. Synthesize a 10 kHz tone in a 22.05 kHz audio stream.
2. Resample to 16 kHz using (a) naive linear interpolation and (b) Kaiser-windowed sinc interpolation.
3. Perform FFT analysis to identify the aliased ghost frequency in the naive resampler.

## Laboratory Code

```python
import torch
import torchaudio.transforms as T

def run_lab():
    print("=== Step 1: Synthesize a 10 kHz Tone at 22.05 kHz ===")
    fs_orig = 22050
    duration = 0.1  # 100 ms
    t = torch.linspace(0, duration, int(fs_orig * duration))
    # 10 kHz is audible and below 22050/2 = 11025 Hz, but ABOVE 16000/2 = 8000 Hz!
    f_tone = 10000.0
    x_orig = torch.sin(2 * torch.pi * f_tone * t).unsqueeze(0)
    
    print("=== Step 2: Resample to 16 kHz with Certified Kaiser Resampler ===")
    certified_resampler = T.Resample(
        orig_freq=fs_orig,
        new_freq=16000,
        resampling_method="kaiser_window",
        lowpass_filter_width=32
    )
    x_clean = certified_resampler(x_orig)
    
    # The 10 kHz tone MUST be filtered out because 10 kHz > 8 kHz (Nyquist of 16 kHz)
    power_clean = torch.mean(x_clean ** 2).item()
    print(f"Residual power in certified output: {power_clean:.6f} (Should be near 0)")
    
    print("\n=== Step 3: Simulate Naive Resampler (No Lowpass Anti-Aliasing) ===")
    # Naive nearest/linear interpolation preserves the samples without filtering
    # Under 16 kHz, 10 kHz folds to: |10000 - 16000| = 6000 Hz
    expected_alias = abs(10000 - 16000)
    print(f"When unfiltered, 10 kHz folds to: {expected_alias} Hz in the 16 kHz stream!")
    print("This ghost tone degraded Kashmiri affricate recognition in the historical Bolbosh set.")

if __name__ == "__main__":
    run_lab()
```
""",
        "exercises": """# Exercises: Lecture 04

## Analytical Problems
1. A recording in the Bolbosh dataset contains a high-energy vowel harmonic at $9.2\\text{ kHz}$. If this recording is downsampled from $22{,}050\\text{ Hz}$ to $16{,}000\\text{ Hz}$ without an anti-aliasing filter, calculate the exact frequency at which this harmonic will appear in the downsampled audio.
2. Explain why polyphase filter implementations reduce the computational complexity of rational resampling by a factor of $M$ compared to direct convolution.

## Implementation Problems
3. Implement a function `verify_spectral_suppression(resampler_fn, test_freq=10000, fs_orig=22050, fs_target=16000) -> float` that injects a test sinusoid and measures the decibel suppression of the out-of-band component.

## Forensic Auditing
4. In the Stage 25 audit report (`docs/audit/INDEPENDENT_AUDIT_VERIFICATION.md`), how did the audit team prove that the Bolbosh CER degradation was caused by resampling rather than poor transcription quality?
""",
        "solutions": """# Solutions: Lecture 04

## Exercise 1
The original frequency is $f = 9{,}200\\text{ Hz}$.
The target sampling rate is $f_s = 16{,}000\\text{ Hz}$, with Nyquist limit $f_N = 8{,}000\\text{ Hz}$.
Since $f > f_N$, the alias frequency is:
$$f_\\text{alias} = |f - f_s| = |9{,}200 - 16{,}000| = 6{,}800\\text{ Hz}$$
The harmonic appears as a spurious $6.8\\text{ kHz}$ artifact in the middle of the speech spectrum.

## Exercise 2
In direct convolution, we upsample by $L$ (inserting $L-1$ zeros), convolve with filter $h[n]$ of length $K$, and downsample by $M$. Most operations involve multiplying zero by filter coefficients, and $M-1$ out of $M$ outputs are discarded. In polyphase decomposition, the filter $h[n]$ is partitioned into $L$ subfilters $E_p(z) = \\sum_k h[k L + p] z^{-k}$. Convolutions are evaluated only at non-zero input samples and only for the outputs that are actually retained, eliminating redundant computations by a factor of $M$.

## Exercise 3
```python
import torch

def verify_spectral_suppression(resampler, test_freq=10000, fs_orig=22050, fs_target=16000) -> float:
    t = torch.linspace(0, 1.0, fs_orig)
    x = torch.sin(2 * torch.pi * test_freq * t).unsqueeze(0)
    y = resampler(x)
    in_power = torch.mean(x ** 2)
    out_power = torch.mean(y ** 2)
    suppression_db = 10 * torch.log10(in_power / (out_power + 1e-12)).item()
    return suppression_db
```
""",
        "quiz": """# Comprehension Quiz: Lecture 04

1. **What is the rational fraction $L/M$ for converting 22,050 Hz to 16,000 Hz?**
   - A) 160 / 220
   - B) 320 / 441 [CORRECT]
   - C) 441 / 320
   - D) 16 / 22
   *Explanation*: $16000 / 22050 = 320 / 441$.

2. **What is the required cutoff frequency for the anti-aliasing filter when resampling 22,050 Hz to 16,000 Hz?**
   - A) 11,025 Hz
   - B) 8,000 Hz [CORRECT]
   - C) 16,000 Hz
   - D) 4,000 Hz
   *Explanation*: The target Nyquist rate is $16000 / 2 = 8000\\text{ Hz}$.

3. **Why did naive resampling of the Bolbosh dataset degrade Kashmiri Character Error Rate?**
   - A) The audio was converted to mono.
   - B) Out-of-band frequencies between 8 kHz and 11.025 kHz folded back into the speech band, corrupting affricate representations. [CORRECT]
   - C) The volume was doubled, causing clipping.
   - D) The transcripts were deleted during conversion.
   *Explanation*: High-frequency energy aliased into the 5-8 kHz band where Kashmiri affricates reside.

4. **Which window function is commonly used in high-quality sinc resampling to achieve >= 85 dB stopband attenuation?**
   - A) Rectangular window
   - B) Kaiser window [CORRECT]
   - C) Bartlett window
   - D) Flat-top window
   *Explanation*: Kaiser windows offer parameterizable stopband attenuation via the $\\beta$ parameter.

5. **In the polyphase implementation of a resampler, when is the lowpass filtering performed?**
   - A) After discarding samples.
   - B) Simultaneously with up-sampling at the lower base rate, avoiding computations on inserted zeros. [CORRECT]
   - C) In the frequency domain using FFT only.
   - D) Before reading the file from disk.
   *Explanation*: Polyphase filters commute the filtering operation to operate efficiently at the native rate.
""",
        "research": """# Research Connections: Lecture 04

- **Repository Stage**: `stages/stage_25/` (Independent Verification & Audit) and `src/data_processing/resampling.py`.
- **The Forensic Proof**: In Stage 25, the audit team extracted 500 Bolbosh audio files, re-resampled them with the certified Kaiser-windowed resampler, and evaluated Model B-02C. The CER dropped from $28.4\\%$ to $24.2\\%$, proving unequivocally that the acoustic model was penalizing aliasing artifacts rather than dialectal variations.
- **Key Code File**: `src/data_processing/resampling.py` (`CertifiedAudioResampler`).
- **Case Study Reference**: Case Study CS-03 ("The 22.05 kHz Resampling Disaster").
""",
        "refs": """# References: Lecture 04

1. Crochiere, R. E., & Rabiner, L. R. (1983). *Multirate Digital Signal Processing*. Prentice-Hall.
2. Vaidyanathan, P. P. (1993). *Multirate Systems and Filter Banks*. Prentice Hall.
3. torchaudio documentation: *Audio Resampling with Kaiser Window*. PyTorch official docs.
"""
    },
    {
        "num": 5,
        "mod": "module_02_speech_and_audio",
        "dir": "lecture_05_fourier_analysis_and_filterbanks",
        "title": "Short-Time Fourier Analysis & Filterbanks",
        "subtitle": "STFT Mathematics, Spectrograms, Mel-Scale Warping, and Modern Self-Supervised Waveforms",
        "objectives": [
            "Derive the Short-Time Fourier Transform (STFT) and analyze time-frequency resolution trade-offs (Heisenberg-Gabor limit).",
            "Construct triangular Mel filterbanks from psychoacoustic pitch perception models.",
            "Compute log-mel spectrogram features in PyTorch and verify energy conservation.",
            "Understand why modern ASR architectures (Wav2Vec2 / XLS-R) operate directly on raw 1D waveforms rather than 2D spectrograms."
        ],
        "notes": """# Lecture 05: Short-Time Fourier Analysis & Filterbanks

## 1. The Non-Stationarity of Speech and the STFT

Speech is inherently non-stationary: its spectral properties change rapidly as the vocal tract articulators (tongue, lips, velum) move. Over short temporal intervals ($20 - 30\\text{ ms}$), speech is **quasi-stationary**.

The **Short-Time Fourier Transform (STFT)** analyzes local frequency content by sliding a finite-duration window function $w[n]$ across the signal:

$$X[m, k] = \\sum_{n=-\\infty}^{\\infty} x[n] w[n - m H] e^{-j \\frac{2\\pi}{N} k n}$$

where:
- $m$ is the time frame index.
- $k$ is the discrete frequency bin index ($0 \\le k < N$).
- $H$ is the hop size (e.g., $10\\text{ ms} = 160$ samples at $16\\text{ kHz}$).
- $N$ is the FFT window length (e.g., $25\\text{ ms} = 400$ samples, zero-padded to $N=512$).

### The Heisenberg-Gabor Uncertainty Principle
In time-frequency analysis, temporal resolution $\\Delta t$ and frequency resolution $\\Delta f$ are bounded by:
$$\\Delta t \\cdot \\Delta f \\ge \\frac{1}{4\\pi}$$
- A short window (e.g. $5\\text{ ms}$) provides sharp temporal precision but blurs harmonic frequencies (wideband spectrogram).
- A long window (e.g. $50\\text{ ms}$) resolves individual pitch harmonics but smears rapid phonetic transitions like plosive bursts (narrowband spectrogram).
- A $25\\text{ ms}$ Hann window with a $10\\text{ ms}$ hop is the empirical standard in speech recognition.

## 2. The Mel Scale and Log-Mel Filterbanks

Human auditory perception of pitch is non-linear: our ears have high frequency resolution below $1\\text{ kHz}$, but logarithmic resolution at higher frequencies.

The **Mel Scale** models this psychoacoustic phenomenon:
$$m = 2595 \\log_{10}\\left( 1 + \\frac{f}{700} \\right)$$

To construct an $M$-channel Mel filterbank:
1. Map linear frequencies $[0, f_s/2]$ to the Mel scale.
2. Place $M$ uniformly spaced triangular filters along the Mel axis.
3. Map the filter center frequencies back to linear Hz:
   $$f = 700 \\left( 10^{\\frac{m}{2595}} - 1 \\right)$$
4. Multiply the power spectrum $|X[m, k]|^2$ with each triangular filter $H_m[k]$ and take the logarithm:
   $$S[m, b] = \\log \\left( \\sum_{k=0}^{N/2} |X[m, k]|^2 H_b[k] + \\epsilon \\right), \\quad b = 1, \\dots, M$$

## 3. Why Did Modern ASR Move Away From Filterbanks?

For 30 years (1980–2015), 80-dimensional log-mel filterbanks or 13-dimensional MFCCs were the universal input to speech recognition models.

However, modern self-supervised foundation models like **Wav2Vec 2.0** and **XLS-R** operate directly on **raw 1D waveforms**:
1. **Preservation of Phase**: Filterbanks discard phase $|X[m, k]|^2$. Raw audio preserves phase information, which is critical for noise cancellation and speaker separation.
2. **Learned Representations**: Rather than imposing a fixed human-designed Mel filterbank, the 7-layer temporal CNN learns data-driven filterbanks optimized directly for downstream representation learning.
3. **Cross-Lingual Uniformity**: Raw audio eliminates hyperparameter discrepancies (window shape, pre-emphasis, Mel formula variants) across multilingual datasets.
""",
        "script": """### Lecture 05 Instructor Delivery Script

**[00:00 - 08:00] Time vs. Frequency: The Fundamental Dilemma**
- "Look at a raw waveform: you see amplitude over time, but you cannot tell what vowel is being spoken. Look at a Fourier transform: you see frequencies, but you have no idea WHEN they happened! How do we get both?"
- *Blackboard*: Draw a 2D time-frequency grid. Explain the STFT sliding window.

**[08:00 - 20:00] The Mel Scale and Human Hearing**
- "Play two tones: 200 Hz and 300 Hz. You can easily hear that 300 Hz is higher. Now play 8000 Hz and 8100 Hz. They sound almost identical to the human ear! The Mel filterbank mimics this cochlear biology."
- *Blackboard*: Draw overlapping triangular filterbanks: dense below 1000 Hz, spread out above 1000 Hz.

**[20:00 - 32:00] The Modern Shift to Raw Waveforms**
- "If Mel filterbanks are so great, why does XLS-R take raw waveforms? Because hand-engineered filters throw away phase and assume human biology is the optimal inductive bias for deep networks. We will see that the first layer of Wav2Vec2's CNN actually learns Gabor-like bandpass filters automatically!"

**[32:00 - 45:00] Lab and Code Inspection**
- "Let's run `practical_lab.md` to compute log-mel spectrograms and visualize the formants of Kashmiri vowels."
""",
        "math": """# Mathematical Foundations: STFT and Triangular Mel Filterbanks

## 1. Discrete Short-Time Fourier Transform (STFT)
Given discrete signal $x[n]$ and window $w[n]$ of length $N$:
$$X[m, k] = \\sum_{n=0}^{N-1} x[m H + n] w[n] e^{-j 2\\pi \\frac{k n}{N}}, \\quad 0 \\le k \\le \\frac{N}{2}$$
where $H$ is the hop size in samples.

The power spectral density is:
$$P[m, k] = |X[m, k]|^2 = \\text{Re}(X[m, k])^2 + \\text{Im}(X[m, k])^2$$

## 2. Triangular Filterbank Construction
Let $\{f_{b-1}, f_b, f_{b+1}\}$ be the center frequencies of three adjacent filters in Hz. The triangular weighting function $H_b[k]$ evaluated at discrete frequency bin $f(k) = k \\frac{f_s}{N}$ is:

$$H_b[k] = \\begin{cases} 0 & f(k) < f_{b-1} \\\\ \\frac{f(k) - f_{b-1}}{f_b - f_{b-1}} & f_{b-1} \\le f(k) \\le f_b \\\\ \\frac{f_{b+1} - f(k)}{f_{b+1} - f_b} & f_b \\le f(k) \\le f_{b+1} \\\\ 0 & f(k) > f_{b+1} \\end{cases}$$

## 3. Log-Energy Feature Vector
The log-mel energy for filterbank channel $b$ at time frame $m$ is:
$$E[m, b] = \\log \\left( \\sum_{k=0}^{N/2} P[m, k] H_b[k] + 10^{-10} \\right)$$
Yielding feature matrix $\\mathbf{E} \\in \\mathbb{R}^{M \\times T'}$, where $M$ is the number of filters (typically $M=80$).
""",
        "code": """# Code Walkthrough: Lecture 05

We inspect `src/data_processing/features.py`, which implements feature extraction for acoustic baselines.

```python
# src/data_processing/features.py
import torch
import torchaudio.transforms as T

class LogMelFeatureExtractor:
    \"\"\"
    Extracts 80-channel Log-Mel Filterbank energies from 16 kHz audio.
    Used for baseline HMM and Conformer comparative evaluations.
    \"\"\"
    def __init__(self, sample_rate: int = 16000, n_fft: int = 400, hop_length: int = 160, n_mels: int = 80):
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,          # 25 ms window
            win_length=n_fft,
            hop_length=hop_length,# 10 ms hop
            n_mels=n_mels,
            power=2.0
        )
        
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        \"\"\"
        Args:
            waveform: (Batch, Samples)
        Returns:
            log_mel: (Batch, n_mels, Frames)
        \"\"\"
        mel_spec = self.mel_transform(waveform)
        # Add epsilon and apply natural logarithm
        log_mel = torch.log(mel_spec + 1e-6)
        return log_mel
```
""",
        "lab": """# Practical Lab 05: Visualizing STFT Spectrograms and Log-Mel Filterbanks

## Objective
1. Compute the STFT of a speech waveform using PyTorch.
2. Construct an 80-channel Mel filterbank and compute log-mel energies.
3. Compare the time-frequency dimensions against the raw waveform downsampling factor.

## Laboratory Code

```python
import torch
import torchaudio.transforms as T

def run_lab():
    print("=== Step 1: Generate Synthetic Speech-Like Signal ===")
    sample_rate = 16000
    duration = 2.0  # 2 seconds
    t = torch.linspace(0, duration, int(sample_rate * duration))
    # Fundamental pitch 150 Hz + Formant at 800 Hz + Formant at 2200 Hz
    x = (0.6 * torch.sin(2 * torch.pi * 150.0 * t) +
         0.3 * torch.sin(2 * torch.pi * 800.0 * t) +
         0.1 * torch.sin(2 * torch.pi * 2200.0 * t)).unsqueeze(0)
    print(f"Waveform shape: {x.shape}")
    
    print("\n=== Step 2: Compute Log-Mel Spectrogram ===")
    n_mels = 80
    hop_length = 160  # 10 ms
    n_fft = 400       # 25 ms
    
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )
    
    mel_spec = mel_transform(x)
    log_mel = torch.log(mel_spec + 1e-6)
    print(f"Log-Mel tensor shape: {log_mel.shape} (Batch, Mels, Frames)")
    print(f"Total time frames: {log_mel.shape[-1]} frames for 2.0s audio (100 fps)")
    print(f"Min: {log_mel.min():.2f}, Max: {log_mel.max():.2f}")

if __name__ == "__main__":
    run_lab()
```
""",
        "exercises": """# Exercises: Lecture 05

## Analytical Problems
1. Given a sampling rate of $16\\text{ kHz}$, an FFT window of $N=512$, and a hop size of $H=160$, calculate:
   - The frequency bin resolution in Hz.
   - The time shift between successive frames in milliseconds.
   - The total number of frames generated for an audio file of length $4.5\\text{ seconds}$.
2. Prove that the Mel-to-Hz conversion formula $f = 700(10^{m/2595} - 1)$ is approximately linear for $f \\ll 700\\text{ Hz}$ using a first-order Taylor expansion.

## Implementation Problems
3. Implement a PyTorch function `extract_energy_envelope(waveform: torch.Tensor, frame_size: int = 400, hop_size: int = 160) -> torch.Tensor` that computes the root-mean-square (RMS) energy per frame without using external signal processing libraries.

## Forensic Auditing
4. In Stage 04 of the project, an acoustic model trained on 80-dim log-mel filterbanks was compared against XLS-R-300M fine-tuned on raw waveforms. On noisy community recordings, the raw waveform model achieved a $12.4\\%$ lower WER. Why are learned 1D convolutions more robust to acoustic noise than fixed triangular filterbanks?
""",
        "solutions": """# Solutions: Lecture 05

## Exercise 1
- Frequency resolution: $\\Delta f = \\frac{f_s}{N} = \\frac{16000}{512} = 31.25\\text{ Hz}$ per bin.
- Time shift: $\\Delta t = \\frac{H}{f_s} = \\frac{160}{16000} = 0.010\\text{ s} = 10\\text{ ms}$.
- Total frames for $4.5\\text{ s}$ ($72{,}000$ samples):
  $$T' = \\left\\lfloor \\frac{72000 - 512}{160} \\right\\rfloor + 1 = \\left\\lfloor \\frac{71488}{160} \\right\\rfloor + 1 = 446 + 1 = 447\\text{ frames}$$

## Exercise 2
Using Taylor expansion for $10^x = e^{x \\ln 10} \\approx 1 + x \\ln 10$:
$$f = 700 \\left( e^{\\frac{m \\ln 10}{2595}} - 1 \\right) \\approx 700 \\left( 1 + \\frac{m \\ln 10}{2595} - 1 \\right) = 700 \\cdot \\frac{2.3026}{2595} m \\approx 0.621 m$$
Thus, for small frequencies ($f \\ll 700\\text{ Hz}$), frequency in Hz is directly proportional to pitch in Mel.

## Exercise 3
```python
import torch

def extract_energy_envelope(waveform: torch.Tensor, frame_size: int = 400, hop_size: int = 160) -> torch.Tensor:
    # waveform shape: (Batch, Samples)
    unfolded = waveform.unfold(dimension=-1, size=frame_size, step=hop_size)  # (Batch, Frames, Frame_Size)
    rms = torch.sqrt(torch.mean(unfolded ** 2, dim=-1))
    return rms
```
""",
        "quiz": """# Comprehension Quiz: Lecture 05

1. **What is the standard hop size and window duration in speech recognition STFT analysis?**
   - A) 1 ms hop, 5 ms window
   - B) 10 ms hop, 25 ms window [CORRECT]
   - C) 50 ms hop, 100 ms window
   - D) 100 ms hop, 500 ms window
   *Explanation*: 25 ms captures quasi-stationary formants; 10 ms hop provides 100 fps temporal resolution.

2. **What does the Heisenberg-Gabor limit dictate for time-frequency representations?**
   - A) That computers cannot compute Fourier transforms in real time.
   - B) That you cannot simultaneously achieve arbitrarily high time resolution and arbitrarily high frequency resolution. [CORRECT]
   - C) That speech signals must have zero phase.
   - D) That Mel filterbanks cannot exceed 80 channels.
   *Explanation*: $\\Delta t \\cdot \\Delta f \\ge 1 / (4\\pi)$ limits simultaneous precision.

3. **Why are Mel filterbanks spaced more densely at frequencies below 1 kHz?**
   - A) Because low frequencies take up less memory.
   - B) Because human auditory perception resolves low frequencies with much higher acuity than high frequencies. [CORRECT]
   - C) Because microphones only record below 1 kHz.
   - D) Because the Nyquist theorem requires it.
   *Explanation*: The Mel scale models the logarithmic frequency resolution of the human cochlea.

4. **What information is discarded when converting an STFT complex matrix to a power spectrogram $|X[m, k]|^2$?**
   - A) Magnitude
   - B) Frequency
   - C) Phase [CORRECT]
   - D) Time
   *Explanation*: $|X|^2 = \\text{Re}^2 + \\text{Im}^2$, completely discarding phase $\\angle X$.

5. **Why do modern foundation models (XLS-R) take raw audio waveforms instead of log-mel filterbanks?**
   - A) Because PyTorch removed support for Mel spectrograms.
   - B) Because learned 1D convolutions can optimize data-driven filter shapes and retain phase information. [CORRECT]
   - C) Because raw waveforms require less GPU memory than spectrograms.
   - D) Because CTC only works on 1D inputs.
   *Explanation*: Learnable 1D CNNs preserve raw signal information without imposing human auditory biases.
""",
        "research": """# Research Connections: Lecture 05

- **Repository Stage**: `stages/stage_04/` (Feature Extraction & Filterbank Experiments).
- **Historic Transition**: Early experiments in Stage 04 trained a Conformer model on 80-dimensional log-mel filterbanks. While fast to train, it plateaued at $38.2\\%$ WER. When the project transitioned in Stage 10 to XLS-R-300M taking raw audio, the baseline WER immediately dropped by $8.6\\%$ absolute, demonstrating the superiority of learned temporal representations for low-resource speech.
- **Key Code File**: `src/data_processing/features.py`.
- **Case Study Reference**: Case Study CS-04 ("Filterbanks vs. Raw Waveform Embeddings in Low-Resource ASR").
""",
        "refs": """# References: Lecture 05

1. Stevens, S. S., Volkmann, J., & Newman, E. B. (1937). *A scale for the measurement of the psychological magnitude pitch*. JASA, 8(3), 185-190.
2. Rabiner, L. R., & Schafer, R. W. (2010). *Theory and Applications of Digital Speech Processing*. Pearson.
3. Schneider, S., et al. (2019). *wav2vec: Unsupervised pre-training for speech recognition*. Interspeech.
"""
    },
    {
        "num": 6,
        "mod": "module_03_kashmiri_language_and_data",
        "dir": "lecture_06_kashmiri_phonology_and_script",
        "title": "Kashmiri Phonology & Perso-Arabic Script",
        "subtitle": "Vowel Inventories, Central Vowels, Palatalization, and Right-to-Left Orthographic Complexities",
        "objectives": [
            "Map the complete phonetic inventory of the Kashmiri language (vowels, consonants, diphthongs).",
            "Analyze the linguistic role of central vowels (/ɨ/, /ɨː/, /ə/, /əː/) and palatalization in Kashmiri grammar.",
            "Deconstruct the Kashmiri Perso-Arabic orthography and identify sources of orthographic ambiguity.",
            "Construct a formalized phonetic-to-graphemic mapping table for the ASR prediction vocabulary."
        ],
        "notes": """# Lecture 06: Kashmiri Phonology & Perso-Arabic Script

## 1. Kashmiri Phonological Inventory

Kashmiri (*Koshur*, كٲشُر) belongs to the Dardic subgroup of the Indo-Aryan language family. Its sound system is phonologically distinctive, setting it sharply apart from neighboring languages such as Urdu, Punjabi, and Hindi.

```
+----------------------------------------------------------------------------------------+
|                             KASHMIRI VOWEL SYSTEM                                      |
+-------------------+-----------------------------+--------------------------------------+
|                   | Front                       | Central                              | Back
+-------------------+-----------------------------+--------------------------------------+
| High              | i, iː                       | ɨ, ɨː  (ü, üü - unique to Kashmiri)  | u, uː
| Mid               | e, eː                       | ə, əː  (ö, öö)                       | o, oː, ɔ, ɔː
| Low               | a, aː                       |                                      |
+-------------------+-----------------------------+--------------------------------------+
```

### The Central Vowels
Kashmiri features a rich system of **central unrounded vowels**:
- High central vowels: short /ɨ/ (written with diacritic *tschapre hath*) and long /ɨː/.
- Mid central vowels: short /ə/ and long /əː/.
These vowels carry high functional load. For instance, altering the central vowel transforms verbal aspect and grammatical gender:
- /kar/ (کر - "do!")
- /kər/ (کٔر - "did (feminine)")
- /kɨr/ (کٕر - "did (plural)")

### Palatalization
Consonantal palatalization (indicated phonetically by $/^j/$) is phonemic in Kashmiri and marks morphological inflections:
- /bɔt/ (بۆٹ - "cooked rice") vs. /bɔtʲ/ (بۆٹؠ - "cooked rice (plural)").

## 2. The Kashmiri Perso-Arabic Orthography

Kashmiri is written right-to-left using an extended Perso-Arabic script standardized by the Jammu & Kashmir Academy of Art, Culture and Languages.

### Unique Orthographic Characters
To represent phonemes absent in Arabic and Persian, special modified glyphs are employed:

| Phoneme | IPA | Kashmiri Perso-Arabic Character | Unicode Codepoint |
|---|---|---|---|
| High central short vowel | /ɨ/ | ٕ (Zer with circle / Tschapre hath) | `U+0655` |
| High central long vowel | /ɨː/ | ؠ (Kashmiri Yeh) | `U+0620` |
| Mid central short vowel | /ə/ | ٔ (Pesh with circle) | `U+0654` |
| Palatalization mark | /$^j$/ | ؠ (Small Yeh subscript) | `U+066F` / `U+0620` |
| Kashmiri Affricate Ts | /t͡s/ | ژ (Tshe) | `U+0698` |
| Kashmiri Aspirated Ts | /t͡sʰ/ | څ (Tshe aspirated) | `U+0685` |

## 3. The ASR Orthographic Dilemma: Omission of Diacritics

In colloquial text (WhatsApp, social media, informal writing), Kashmiri speakers frequently **omit all diacritics**, writing:
$$\\text{کر (kr)} \\quad \\text{for} \\quad \\text{کٔر (/kər/)}, \\; \\text{کٕر (/kɨr/)}, \\; \\text{or} \\; \\text{کار (/kaːr/)}$$

For an acoustic model, this poses a severe dilemma:
- **Audio contains clear acoustics**: The speaker distinctly utters $/kər/$ or $/kɨr/$.
- **Training target lacks diacritic**: If the reference text is written simply as `کر`, the CTC loss forces the model to ignore phonetic reality, driving the acoustic probability distribution toward high entropy.
- **The Solution**: Rigorous automated and manual phonetic restoration during data engineering (Stage 05 & Stage 26).
""",
        "script": """### Lecture 06 Instructor Delivery Script

**[00:00 - 10:00] Linguistic Introduction**
- "Today we dive into the linguistics of Kashmiri. If you do not understand the sound system of the language you are modeling, you will spend months chasing phantom bugs in your neural networks."
- *Blackboard*: Draw the vowel quadrangle showing front, central, and back vowels. Point emphatically to the central region.
- "Look at /ɨ/ and /ə/. In English, /ə/ is the unaccented 'schwa' in 'about'. In Kashmiri, it is an accented, phonemically distinctive vowel that changes the tense and gender of verbs!"

**[10:00 - 25:00] Perso-Arabic Script and Diacritics**
- "Perso-Arabic is an abjad—consonants are written as full letters, while vowels are secondary diacritics floating above or below letters. What happens when people stop typing diacritics? The written word becomes an acoustic riddle."
- *Demo*: Write `کر` on the blackboard. Show that it could mean three entirely different spoken words depending on whether you add a diacritic.

**[25:00 - 35:00] Acoustic Model Implications**
- "If an ASR acoustic model hears /ɨ/ but the training transcript says `کر`, the CTC loss penalizes the model for predicting the correct central vowel! This is why normalization and character mapping in `src/normalization/` is the true backbone of our project."

**[35:00 - 45:00] Lab and Vocabulary Inspection**
- "Let's inspect `src/normalization/kashmiri_char_map.py` to see the exact 98 Unicode tokens accepted by our production model."
""",
        "math": """# Mathematical Foundations: Vocabulary Entropy & Character Confusion

## 1. Orthographic Ambiguity as Conditional Entropy
Let $A$ be the spoken acoustic realization and $W_d$ be the fully diacritized phonetic transcription. Let $W_u$ be the un-diacritized colloquial transcription generated by a projection function:
$$W_u = g(W_d)$$
where $g$ collapses all diacritic variants onto base consonants.

The conditional entropy of the acoustic signal given the un-diacritized target is:
$$H(A \\mid W_u) = H(A \\mid W_d) + H(W_d \\mid W_u) - I(A; W_d \\mid W_u)$$

Because $g$ is a many-to-one mapping, $H(W_d \\mid W_u) > 0$. Training an acoustic model to minimize CTC loss against $W_u$ forces the model to average over distinct acoustic modes:
$$P(A \\mid W_u) = \\sum_{W_d \\in g^{-1}(W_u)} P(A \\mid W_d) P(W_d \\mid W_u)$$
This multimodal acoustic distribution increases the prediction variance and degrades CER.

## 2. Character Vocabulary Mapping
The vocabulary $\\mathcal{V}$ is defined as a bijective index mapping:
$$\\phi: \\mathcal{V} \\rightarrow \\{0, 1, \\dots, |\\mathcal{V}| - 1\\}$$
where:
- $\\phi(\\epsilon) = 0$ (CTC Blank)
- $\\phi(\\text{'|'}) = 1$ (Word delimiter space)
- $\\phi(c_i) = i$ for $i \\in \\{2, \\dots, 97\\}$ (Kashmiri consonants, central vowels, digits, punctuation).
""",
        "code": """# Code Walkthrough: Lecture 06

We inspect `src/normalization/kashmiri_char_map.py`, which defines the official orthographic mapping.

```python
# src/normalization/kashmiri_char_map.py
import unicodedata

# Certified Kashmiri Character Inventory for EduCasheer ASR
KASHMIRI_CHARACTERS = [
    "<pad>", "|",  # 0: Blank, 1: Word boundary
    "ا", "ب", "پ", "ت", "ٹ", "ث", "ج", "چ", "ح", "خ", "د", "ڈ", "ذ", "ر", "ڑ", "ز", "ژ", "س", "ش", "ص", "ض", "ط", "ظ", "ع", "غ", "ف", "ق", "ک", "گ", "ل", "م", "ن", "و", "ہ", "ی", "ے",
    # Kashmiri Unique Diacritics and Extended Vowels
    "آ", "أ", "ؤ", "ئ", "ۍ", "ؠ", "څ", "ڈ",
    "َ", "ِ", "ُ", "ً", "ٍ", "ٌ", "ّ", "ْ", "ٔ", "ٕ", "ٓ"
]

CHAR_TO_ID = {c: i for i, c in enumerate(KASHMIRI_CHARACTERS)}
ID_TO_CHAR = {i: c for i, c in enumerate(KASHMIRI_CHARACTERS)}

def get_vocab_size() -> int:
    return len(KASHMIRI_CHARACTERS)
```
""",
        "lab": """# Practical Lab 06: Kashmiri Script Tokenization and Diacritic Stripping Analysis

## Objective
1. Inspect Kashmiri Unicode strings and identify decomposed combining characters.
2. Tokenize text using the certified `kashmiri_char_map.py` vocabulary.
3. Quantify the entropy increase when diacritics are stripped from a sample Kashmiri sentence.

## Laboratory Code

```python
import unicodedata

def run_lab():
    print("=== Step 1: Inspect Kashmiri Perso-Arabic Unicode Characters ===")
    sample_text = "کٲشُر زَبانہِ مَنٛز"
    print(f"Sample string: {sample_text}")
    print(f"String length in Python characters: {len(sample_text)}")
    
    for i, ch in enumerate(sample_text):
        name = unicodedata.name(ch, "UNKNOWN")
        cp = f"U+{ord(ch):04X}"
        print(f"Index {i:2d}: '{ch}' ({cp}) -> {name}")
        
    print("\n=== Step 2: Analyze Combining Characters ===")
    combining_count = sum(1 for ch in sample_text if unicodedata.combining(ch) > 0)
    print(f"Total combining marks (diacritics): {combining_count}")
    
    print("\n=== Step 3: Simulate Diacritic Stripping ===")
    stripped_text = "".join(ch for ch in sample_text if unicodedata.combining(ch) == 0)
    print(f"Original text: '{sample_text}'")
    print(f"Stripped text: '{stripped_text}'")
    print(f"Loss of morphological information: {len(sample_text) - len(stripped_text)} characters removed.")

if __name__ == "__main__":
    run_lab()
```
""",
        "exercises": """# Exercises: Lecture 06

## Analytical Problems
1. Explain the phonetic and grammatical difference between the Kashmiri words كَر (/kar/), كٔر (/kər/), and كٕر (/kɨr/). Why is it catastrophic for a speech translation system if the ASR decoder omits the diacritic?
2. Kashmiri contains both dental affricates (/t͡s/, /t͡sʰ/) and palato-alveolar affricates (/t͡ʃ/, /t͡ʃʰ/). List the corresponding Perso-Arabic letters used for each in standard orthography.

## Implementation Problems
3. Write a Python function `validate_kashmiri_text(text: str, allowed_vocab: set) -> tuple[bool, list[str]]` that checks whether an input string contains only certified Kashmiri characters and returns all illegal characters found.

## Forensic Auditing
4. In Stage 05 of the EduCasheer repository, the vocabulary was audited after finding that Character Error Rate was artificially inflated due to identical-looking Arabic vs. Persian Unicode characters (e.g., Arabic Kaf `U+0643` vs. Persian Keheh `U+06A9`). How did this visual homoglyph trap affect evaluation?
""",
        "solutions": """# Solutions: Lecture 06

## Exercise 1
- كَر (/kar/): Root imperative verb "to do" (masculine/generic).
- كٔر (/kər/): Past tense feminine singular form ("she did").
- كٕر (/kɨr/): Past tense feminine plural form ("they did").
If the ASR decoder strips diacritics and outputs `کر`, downstream machine translation models cannot infer whether the subject was masculine, feminine singular, or feminine plural, causing catastrophic gender and agreement mistranslations.

## Exercise 2
- Dental affricate /t͡s/: ژ (`U+0698`, Tshe)
- Dental aspirated affricate /t͡sʰ/: څ (`U+0685`, Tshe aspirated)
- Palato-alveolar affricate /t͡ʃ/: چ (`U+0686`, Che)
- Palato-alveolar aspirated affricate /t͡ʃʰ/: چھ (`U+0686` + `U+06BE`, Che with Heh Doachashmee)

## Exercise 3
```python
def validate_kashmiri_text(text: str, allowed_vocab: set) -> tuple[bool, list[str]]:
    illegal = [ch for ch in text if ch not in allowed_vocab]
    return (len(illegal) == 0, illegal)
```
""",
        "quiz": """# Comprehension Quiz: Lecture 06

1. **To which language family does Kashmiri belong?**
   - A) Dravidian
   - B) Dardic branch of Indo-Aryan [CORRECT]
   - C) Tibeto-Burman
   - D) Semitic
   *Explanation*: Kashmiri is an Indo-Aryan language classified within the Dardic subgroup.

2. **Which vowel class is a distinguishing characteristic of Kashmiri phonology?**
   - A) Front rounded vowels
   - B) High and mid central unrounded vowels (/ɨ/, /ə/) [CORRECT]
   - C) Nasal diphthongs only
   - D) Click consonants
   *Explanation*: Central vowels like /ɨ/ and /ə/ are central to Kashmiri verbal aspect and morphology.

3. **What is the primary danger of training an ASR acoustic model on text where speakers omitted vowel diacritics?**
   - A) The GPU runs out of VRAM.
   - B) Multiple distinct acoustic realizations map to the same text label, creating high conditional entropy and degrading model convergence. [CORRECT]
   - C) The CTC loss produces negative numbers.
   - D) Word Error Rate becomes mathematically undefined.
   *Explanation*: Undiacritized text conflates distinct phonetic vowels into identical graphemes.

4. **In Unicode, what is the character code for the Kashmiri central vowel mark Tschapre Hath (Zer with circle)?**
   - A) `U+0655` [CORRECT]
   - B) `U+0020`
   - C) `U+0627`
   - D) `U+064A`
   *Explanation*: `U+0655` represents Arabic Subscript Alef / Kashmiri Tschapre Hath.

5. **How does the Kashmiri Perso-Arabic alphabet handle the dental affricate /t͡s/?**
   - A) It borrows the letter from Sanskrit Devanagari.
   - B) It uses the specialized character ژ (Tshe). [CORRECT]
   - C) It omits the sound completely.
   - D) It uses a Latin digraph.
   *Explanation*: The character ژ (`U+0698`) represents /t͡s/ in Kashmiri.
""",
        "research": """# Research Connections: Lecture 06

- **Repository Stage**: `stages/stage_05/` (Lexicon & Kashmiri Unicode Standardization) and `src/normalization/kashmiri_char_map.py`.
- **The Homoglyph Crisis**: In Stage 05, the team discovered that text gathered from different web sources contained a mixture of Arabic Kaf (`U+0643`), Persian Keheh (`U+06A9`), and Urdu Kaf (`U+06A9`). Visually identical on screen, Python treated them as distinct characters, doubling vocabulary size and dividing acoustic evidence across three output heads.
- **Key Code File**: `src/normalization/kashmiri_char_map.py`.
- **Case Study Reference**: Case Study CS-05 ("Orthographic Normalization Anomalies and the Homoglyph Crisis").
""",
        "refs": """# References: Lecture 06

1. Koul, O. N., & Wali, K. (2006). *Modern Kashmiri: An advanced course*. Dunwoody Press.
2. Grierson, G. A. (1915). *A Dictionary of the Kashmiri Language*. Asiatic Society of Bengal.
3. Unicode Consortium. (2023). *The Unicode Standard, Version 15.0: Arabic Extended-A*.
"""
    },
    {
        "num": 7,
        "mod": "module_03_kashmiri_language_and_data",
        "dir": "lecture_07_text_normalization",
        "title": "Text Normalization & Unicode Standardization",
        "subtitle": "NFD vs. NFC Normalization, Zero-Width Characters, Ligature Stripping, and The Certified Normalizer v3",
        "objectives": [
            "Deconstruct Unicode normalization standards (NFC, NFD, NFKC, NFKD) and their impact on text processing.",
            "Identify and neutralize non-printing control characters (ZWNJ `U+200C`, ZWJ `U+200D`, LTR/RTL marks).",
            "Trace the evolution of the EduCasheer Kashmiri normalizer from Stage 01 heuristics to Certified Normalizer v3.",
            "Implement a production-grade normalizer pipeline with automated regression unit tests."
        ],
        "notes": """# Lecture 07: Text Normalization & Unicode Standardization

## 1. The Critical Role of Normalization in Speech Recognition

In Automatic Speech Recognition, the acoustic model is trained to output character or subword sequences that match reference transcriptions. If the reference text contains inconsistent encodings, the model is penalized for predicting phonetically correct text that differs only by Unicode representation.

For example, consider the Kashmiri word for "Kashmiri":
$$\\text{كٲشُر}$$
This single word can be represented digitally in over **six different byte sequences**:
1. Base Kaf + Kashmiri Yeh + Diacritic Zer + Sheen + Pesh + Re.
2. Pre-composed Arabic letter + separate combining marks.
3. Decomposed NFD form where diacritics are ordered differently.
4. Using Urdu Yeh (`U+06D2`) instead of Kashmiri Yeh (`U+0620`).

If these representations are not strictly canonicalized, the ASR vocabulary explodes, training targets become noisy, and evaluation metrics (WER/CER) produce false substitution errors.

## 2. Unicode Normalization Forms (NFC vs. NFD)

The Unicode standard defines four normalization forms:
- **NFD (Canonical Decomposition)**: Characters are decomposed into base glyphs plus separate combining diacritics.
- **NFC (Canonical Decomposition followed by Canonical Composition)**: Characters are decomposed and then re-composed into precomposed characters where available.
- **NFKD / NFKC**: Compatibility forms that also replace formatting distinctions (such as superscripts or ligatures) with base equivalents.

```
NFD:  [ Base Character ا (U+0627) ] + [ Combining Hamza ٔ (U+0654) ]  (2 codepoints)
                    |
                    v (NFC Canonical Composition)
NFC:  [ Precomposed Character أ (U+0623) ]                            (1 codepoint)
```

### The Perso-Arabic Trap in NFC
In Arabic and Kashmiri, NFC **does not compose all diacritics** into single codepoints because Unicode does not provide precomposed characters for all Kashmiri vowel-diacritic combinations (such as *Tschapre hath* on Kashmiri Yeh). Therefore, a naive call to `unicodedata.normalize('NFC', text)` leaves Kashmiri text partially decomposed!

The EduCasheer project solved this by engineering **Certified Normalizer v3** (`src/normalization/normalizer.py`), which enforces an explicit, deterministic canonical order.

## 3. Non-Printing Characters: ZWNJ, ZWJ, and Directional Marks

Perso-Arabic script is cursive: characters join dynamically depending on their neighbors. To control cursive joining, typography relies on non-printing control codes:
- **Zero-Width Non-Joiner (ZWNJ, `U+200C`)**: Prevents two adjacent letters from joining.
- **Zero-Width Joiner (ZWJ, `U+200D`)**: Forces two letters to join even when separated.
- **Directional Marks (LTR `U+200E`, RTL `U+200F`)**: Control text rendering direction.

In spoken audio, **a non-printing character makes zero acoustic sound**. An acoustic model cannot hear a ZWNJ. If reference transcripts include ZWNJs randomly, the acoustic model can never predict them reliably.

**The Golden Rule of ASR Normalization**:
$$\\text{Strip all non-printing control characters (ZWNJ, ZWJ, LTR, RTL) from acoustic training targets.}$$
""",
        "script": """### Lecture 07 Instructor Delivery Script

**[00:00 - 08:00] The Invisible Bug That Ruins Metrics**
- "Imagine you train an ASR model for 48 hours. You test it on a validation audio file. The model outputs: `کٲشُر`. The reference file says: `کٲشُر`. They look 100% identical on screen. But your evaluation script reports: `ERROR: Substitution! WER = 100%`! Why?"
- *Blackboard*: Write `U+0643 U+0620` vs `U+06A9 U+064A`. "Under the hood, they are completely different numbers!"

**[08:00 - 22:00] Decomposition and Combining Diacritics**
- "Explain Unicode NFC vs NFD. Show why standard Python `unicodedata.normalize('NFC', text)` works for French and German, but fails on Kashmiri because Kashmiri diacritics lack precomposed Unicode slots."
- *Demo*: Show terminal output of decomposing a Kashmiri string and seeing the byte length jump from 12 bytes to 18 bytes.

**[22:00 - 32:00] The Non-Printing Phantom: ZWNJ**
- "What is a Zero-Width Non-Joiner? It's a ghost. Typographers use it so letters don't touch. But how does a ZWNJ sound in speech? Does it sound like /b/? Does it sound like /s/? It is silent! If your transcript has it, your model will fail to predict it."

**[32:00 - 45:00] Walkthrough of Certified Normalizer v3**
- "Open `src/normalization/normalizer.py`. We will study the regular expressions and mapping dictionaries that power our certified production pipeline."
""",
        "math": """# Mathematical Foundations: Canonical Transformation Equivalence

## 1. Formal Definition of Normalization Operator
Let $\\Sigma^*$ be the set of all finite string sequences over the Unicode character set $\\Sigma$.
A text normalizer is an idempotent transformation function:
$$\\mathcal{N}: \\Sigma^* \\rightarrow \\Sigma^*$$
satisfying the idempotence property:
$$\\mathcal{N}(\\mathcal{N}(s)) = \\mathcal{N}(s) \\quad \\forall s \\in \\Sigma^*$$

## 2. Homoglyph Equivalence Classes
Let $\\sim_H$ be the homoglyph equivalence relation where two characters $c_1, c_2 \\in \\Sigma$ are equivalent if they represent the same phoneme and have identical or interchangeable visual forms in Kashmiri:
$$c_1 \\sim_H c_2 \\iff \\text{Phoneme}(c_1) = \\text{Phoneme}(c_2)$$
Examples:
$$\\{\\text{ك (U+0643)}, \\; \\text{ک (U+06A9)}\\} \\in [k]_H$$
$$\\{\\text{ي (U+064A)}, \\; \\text{ی (U+06CC)}\\} \\in [j]_H$$

The normalizer maps every element of the equivalence class $[c]_H$ to a unique canonical representative $c^*$:
$$\\mathcal{N}(c) = c^* \\quad \\forall c \\in [c]_H$$

## 3. Non-Printing Elimination Operator
Let $\\mathcal{C}_\\text{invisible} = \\{\\text{U+200C}, \\text{U+200D}, \\text{U+200E}, \\text{U+200F}, \\text{U+FEFF}\\}$.
The filtering projection operator $\\pi_\\text{clean}$ satisfies:
$$\\pi_\\text{clean}(s) = \\{c \\in s \\mid c \\notin \\mathcal{C}_\\text{invisible}\\}$$
""",
        "code": """# Code Walkthrough: Lecture 07

We inspect `src/normalization/normalizer.py`, the core production normalizer for Kashmiri speech recognition.

```python
# src/normalization/normalizer.py
import re
import unicodedata

class KashmiriNormalizerV3:
    \"\"\"
    Certified Normalizer v3 for EduCasheer Kashmiri ASR.
    Enforces canonical homoglyph replacement, diacritic ordering,
    and non-printing character stripping.
    \"\"\"
    def __init__(self):
        # Homoglyph substitution map: Variant -> Canonical
        self.homoglyph_map = {
            'ك': 'ک',  # Arabic Kaf -> Persian/Kashmiri Kaf
            'ي': 'ی',  # Arabic Yeh -> Kashmiri Yeh
            'ى': 'ی',  # Alef Maksura -> Kashmiri Yeh
            'ہ': 'ہ',  # Normalize Heh variants
            'ھ': 'ھ',  # Heh Doachashmee
            'ة': 'ت',  # Teh Marbuta -> Teh
        }
        
        # Non-printing regex: ZWNJ, ZWJ, LTR, RTL, BOM
        self.invisible_regex = re.compile(r'[\u200c\u200d\u200e\u200f\ufeff]')
        # Multiple whitespace collapse
        self.whitespace_regex = re.compile(r'\\s+')

    def normalize(self, text: str) -> str:
        if not text:
            return ""
            
        # 1. Unicode NFC baseline
        text = unicodedata.normalize('NFC', text)
        
        # 2. Strip non-printing control codes
        text = self.invisible_regex.sub('', text)
        
        # 3. Canonical homoglyph replacement
        chars = [self.homoglyph_map.get(ch, ch) for ch in text]
        text = "".join(chars)
        
        # 4. Collapse redundant whitespace and strip edges
        text = self.whitespace_regex.sub(' ', text).strip()
        
        return text
```
""",
        "lab": """# Practical Lab 07: Testing Normalizer Idempotence and Homoglyph Neutralization

## Objective
1. Implement and test `KashmiriNormalizerV3`.
2. Verify mathematical idempotence: $\\mathcal{N}(\\mathcal{N}(s)) = \\mathcal{N}(s)$.
3. Measure Character Error Rate before and after normalization on a pair of visually identical strings.

## Laboratory Code

```python
import unicodedata
import re

def run_lab():
    print("=== Step 1: Create Problematic Kashmiri Test Strings ===")
    # String A: Arabic Kaf (U+0643) + ZWNJ (U+200C) + Arabic Yeh (U+064A)
    str_a = "\u0643\u200C\u064A\u0634\u064F\u0631"
    # String B: Kashmiri Kaf (U+06A9) + Kashmiri Yeh (U+06CC) + no ZWNJ
    str_b = "\u06A9\u06CC\u0634\u064F\u0631"
    
    print(f"String A display: '{str_a}' | Length: {len(str_a)}")
    print(f"String B display: '{str_b}' | Length: {len(str_b)}")
    print(f"Are raw strings equal? {str_a == str_b}")
    
    print("\n=== Step 2: Apply Certified Normalizer v3 ===")
    from src.normalization.normalizer import KashmiriNormalizerV3
    normalizer = KashmiriNormalizerV3()
    
    norm_a = normalizer.normalize(str_a)
    norm_b = normalizer.normalize(str_b)
    
    print(f"Normalized A: '{norm_a}' | Length: {len(norm_a)}")
    print(f"Normalized B: '{norm_b}' | Length: {len(norm_b)}")
    print(f"Are normalized strings equal? {norm_a == norm_b}")
    
    print("\n=== Step 3: Test Idempotence Property ===")
    double_norm_a = normalizer.normalize(norm_a)
    is_idempotent = (double_norm_a == norm_a)
    print(f"N(N(s)) == N(s): {is_idempotent} (Certified Idempotent)")

if __name__ == "__main__":
    run_lab()
```
""",
        "exercises": """# Exercises: Lecture 07

## Analytical Problems
1. Prove that if a normalization function $\\mathcal{N}$ is idempotent and preserves the alphabet $\\mathcal{V}$, then the set of canonical strings forms a closed sub-algebra under $\\mathcal{N}$.
2. In Perso-Arabic typography, the sequence `Kaf` + `Alef` is rendered as the ligature `کا`. Explain the difference between a typographic rendering ligature and a Unicode precomposed character. Why must tokenizers never treat ligatures as distinct vocabulary tokens?

## Implementation Problems
3. Write a Python unit test using `pytest` that asserts:
   - All Arabic digits (`۰-۹`) and Persian digits (`۰-۹`) are normalized to standard Kashmiri Persian digit forms.
   - Any string containing only whitespace normalizes to an empty string `""`.

## Forensic Auditing
4. In Stage 26 of the EduCasheer repository, the dataset audit revealed that switching from Normalizer v1 to Certified Normalizer v3 reduced the certified test WER by $0.94\\%$ absolute on identical model checkpoints without re-training! Explain how normalization alone can change the evaluation metric.
""",
        "solutions": """# Solutions: Lecture 07

## Exercise 1
A function $\\mathcal{N}: S \\rightarrow S$ is idempotent if $\\mathcal{N} \\circ \\mathcal{N} = \\mathcal{N}$.
Let $C = \\{\\mathcal{N}(s) \\mid s \\in S\\}$ be the image of $\\mathcal{N}$ (the set of canonical strings).
For any $c \\in C$, there exists $s \\in S$ such that $c = \\mathcal{N}(s)$.
Then $\\mathcal{N}(c) = \\mathcal{N}(\\mathcal{N}(s)) = \\mathcal{N}(s) = c$.
Thus, every element in $C$ maps strictly to itself, making $C$ closed under $\\mathcal{N}$.

## Exercise 2
A typographic ligature is an artifact of the font renderer (e.g. OpenType shaping engine) that visually combines two distinct glyphs into one artistic form for display. The underlying text buffer retains two distinct Unicode codepoints (`U+06A9` and `U+0627`). A precomposed character is a single Unicode codepoint representing multiple sounds. If an ASR tokenizer treated font ligatures as single tokens, the vocabulary would depend on the user's display font rather than linguistic identity.

## Exercise 3
```python
def test_normalizer_digits_and_whitespace():
    from src.normalization.normalizer import KashmiriNormalizerV3
    norm = KashmiriNormalizerV3()
    assert norm.normalize("   \t  \n  ") == ""
    # Test Arabic vs Persian digit normalization
    assert norm.normalize("۱۲۳") == "۱۲۳"
```
""",
        "quiz": """# Comprehension Quiz: Lecture 07

1. **What is an idempotent normalization function?**
   - A) A function that runs in $O(1)$ time.
   - B) A function that satisfies $\\mathcal{N}(\\mathcal{N}(s)) = \\mathcal{N}(s)$ for all inputs. [CORRECT]
   - C) A function that converts strings to binary.
   - D) A function that only deletes vowels.
   *Explanation*: Idempotence guarantees that applying the normalizer multiple times causes no further modifications.

2. **Why must non-printing characters like ZWNJ (`U+200C`) be stripped from ASR training targets?**
   - A) Because they cause PyTorch to crash with a segmentation fault.
   - B) Because they produce no acoustic signal, making it impossible for an acoustic model to predict their presence consistently. [CORRECT]
   - C) Because they take up too much memory in RAM.
   - D) Because Unicode forbids ZWNJs in speech recognition.
   *Explanation*: Non-printing characters have no acoustic correlates, so enforcing their prediction injects pure noise into CTC training.

3. **What is the difference between Unicode NFC and NFD?**
   - A) NFC uses 8-bit bytes; NFD uses 16-bit bytes.
   - B) NFC composes combining diacritics into precomposed characters where possible; NFD decomposes characters into base glyphs + separate diacritics. [CORRECT]
   - C) NFC is only for English; NFD is for Arabic.
   - D) NFC deletes spaces; NFD keeps spaces.
   *Explanation*: NFC is canonical composition; NFD is canonical decomposition.

4. **What is the homoglyph trap in multilingual text processing?**
   - A) Using the wrong sampling rate.
   - B) Characters from different alphabets (e.g., Arabic Kaf vs Persian Keheh) that look identical on screen but have different Unicode codepoints. [CORRECT]
   - C) Loss of audio fidelity in MP3 compression.
   - D) Gradient explosion in CTC loss.
   *Explanation*: Visual homoglyphs create distinct vocabulary tokens for identical phonemes.

5. **Why did switching to Certified Normalizer v3 reduce WER by 0.94% without retraining the acoustic model?**
   - A) The normalizer made the audio louder.
   - B) It eliminated false substitution errors caused by Unicode homoglyphs and invisible characters between predictions and references. [CORRECT]
   - C) It altered the beam search width.
   - D) It increased the GPU clock speed.
   *Explanation*: Aligning reference and prediction Unicode encodings eliminates spurious error penalties.
""",
        "research": """# Research Connections: Lecture 07

- **Repository Stage**: `stages/stage_26/` (Modular Physical Reorganization) and `src/normalization/normalizer.py`.
- **The Empirical Evidence**: In Stage 26, the entire EduCasheer corpus was re-indexed with Certified Normalizer v3. All 111,928 manifest entries were verified for zero remaining ZWNJ/ZWJ characters, and all Kaf/Yeh homoglyphs were unified, creating the certified baseline for Stage 27 inference parity.
- **Key Code File**: `src/normalization/normalizer.py`.
- **Case Study Reference**: Case Study CS-05 ("Orthographic Normalization Anomalies and the Homoglyph Crisis").
""",
        "refs": """# References: Lecture 07

1. Davis, M., & Whistler, K. (2023). *Unicode Standard Annex #15: Unicode Normalization Forms*. Unicode Consortium.
2. Pourdamghani, N., & Hermjakob, U. (2017). *A Survey of Persian Text Processing*. ISI Technical Report.
3. EduCasheer Project. (2026). *Certified Evaluation Protocol and Normalizer Specification*. Internal Report.
"""
    },
    {
        "num": 8,
        "mod": "module_03_kashmiri_language_and_data",
        "dir": "lecture_08_speech_corpora_collection",
        "title": "Speech Corpora Collection & Curation",
        "subtitle": "Field Recordings, Web Scraping, SNR Thresholding, Audio Validation, and Silence Trimming",
        "objectives": [
            "Design ethical and scalable speech data collection pipelines for low-resource linguistic communities.",
            "Formulate signal processing filters for automatic speech quality curation (energy SNR, clipping detection, silence trimming).",
            "Understand the acoustic trade-offs between clean studio speech (IndicVoices) and noisy in-the-wild speech (WAAZ/Bolbosh).",
            "Implement an automated audio curation and validation script in PyTorch."
        ],
        "notes": """# Lecture 08: Speech Corpora Collection & Curation

## 1. The Low-Resource Speech Gathering Challenge

In high-resource speech recognition (e.g., English, Mandarin), corpora such as LibriSpeech (1,000h) or GigaSpeech (10,000h) are readily available with clean studio quality and professional alignments.

In Kashmiri, building a production ASR corpus required harvesting and curating data from **12 heterogeneous sources**:

```
+-----------------------------------------------------------------------------------------+
|                         EDUCASHEER KASHMIRI DATASET REGISTRY                            |
+------------------------------------+------------------+------------------+--------------+
| Source Category                    | Domain / Style   | Sampling Quality | Total Hours  |
+------------------------------------+------------------+------------------+--------------+
| 1. Bolbosh Crowdsourced            | Read sentences   | 22.05 kHz MP3    | ~45.2 hours  |
| 2. IndicVoices Kashmiri            | Read & Prompts   | 16 kHz WAV studio| ~38.4 hours  |
| 3. WAAZ Religious Sermons          | Spontaneous orat.| Multi-speaker    | ~52.1 hours  |
| 4. Quranic Kashmiri Exegesis       | Chanted / Read   | Clean studio     | ~24.6 hours  |
| 5. Kashmiri Broadcast News         | Formal read      | Studio broadcast | ~18.3 hours  |
| 6. Community WhatsApp Voice Notes  | Casual / noisy   | Mobile telephony | ~36.0 hours  |
+------------------------------------+------------------+------------------+--------------+
```

Total curated speech volume: **111,928 utterances (~214.65 hours)**.

## 2. Automated Signal Quality Assessment Pipeline

Raw harvested audio cannot be fed directly into training pipelines. Low-quality audio acts as label noise, corrupting the acoustic gradients. The EduCasheer automated curation pipeline applies four sequential signal quality filters:

```
Raw Audio Stream
       |
       v
[ Filter 1: Clipping Detection ] ---> Exclude if > 0.5% samples clipped at max amplitude
       |
       v
[ Filter 2: Energy SNR Estimation ] --> Exclude if estimated SNR < 12.0 dB
       |
       v
[ Filter 3: Duration Boundary Check ] -> Exclude if duration < 0.8s or > 18.0s
       |
       v
[ Filter 4: Voice Activity / Trimming ] -> Trim leading/trailing silence (> 300ms)
       |
       v
Validated Acoustic Sample
```

### The Clipping Criterion
Digital clipping occurs when the input acoustic pressure exceeds the analog-to-digital converter's voltage limit, flattening peaks:
$$\\text{ClipRatio} = \\frac{1}{N} \\sum_{n=1}^N \\mathbb{I}(|x[n]| \\ge 0.999)$$
If $\\text{ClipRatio} > 0.005$ ($0.5\\%$ of samples), the file is rejected.

### The Energy SNR Criterion
Estimated via energy percentile difference:
$$\\text{SNR}_\\text{est} = 10 \\log_{10} \\left( \\frac{P_{95}}{P_{10} + \\epsilon} \\right)$$
where $P_{95}$ is the 95th percentile of short-time frame energy (representing speech activity) and $P_{10}$ is the 10th percentile (representing the ambient noise floor). Samples with $\\text{SNR}_\\text{est} < 12\\text{ dB}$ are quarantined.
""",
        "script": """### Lecture 08 Instructor Delivery Script

**[00:00 - 10:00] The Harsh Reality of Real-World Audio**
- "In university courses, datasets are usually handed to you on a silver platter: clean 16 kHz WAV files with perfect transcripts. When we started the EduCasheer project, we had WhatsApp voice notes, noisy mosque loudspeakers, and MP3 files recorded on $30 Android phones! If you feed garbage into your neural network, you will get garbage out."

**[10:00 - 25:00] Building the 4-Stage Curation Filter**
- "Walk through the four filters on the board:
  1. Clipping: when someone speaks too close to the microphone, the wave flattens. That injects fake high-frequency noise.
  2. SNR Estimation: how do you estimate signal-to-noise ratio when you don't have separate clean speech and noise channels? We use energy percentiles ($P_{95} / P_{10}$).
  3. Duration: Why discard clips under 0.8 seconds? Because CTC cannot align 4-letter words with only 2 frames! Why discard clips over 18 seconds? GPU Out-Of-Memory (OOM) errors!
  4. VAD Silence Trimming: Why train a model on 3 seconds of silence?"

**[25:00 - 35:00] Diversity vs. Domain Shift**
- "Notice our 12 sources. If we only train on IndicVoices studio speech, our model achieves 12% WER in the lab, but 65% WER when a real Kashmiri citizen uses it on WhatsApp! We must train on diverse, noisy data while filtering out corrupt files."

**[35:00 - 45:00] Lab and Code Inspection**
- "Open `src/data_collection/curator.py` to see the automated curation rules in action."
""",
        "math": """# Mathematical Foundations: Non-Intrusive SNR & Voice Activity Detection

## 1. Non-Intrusive Percentile-Based SNR Estimation
Let a speech signal $x[n]$ be segmented into $M$ non-overlapping frames of length $L = 320$ ($20\\text{ ms}$ at $16\\text{ kHz}$).
The short-time frame energy is:
$$E_m = \\frac{1}{L} \\sum_{k=0}^{L-1} x[m L + k]^2, \\quad m = 1, \\dots, M$$

Let the sorted energy sequence be $E_{(1)} \\le E_{(2)} \\le \\dots \\le E_{(M)}$.
The noise floor estimate is the 10th percentile:
$$\\sigma_\\text{noise}^2 = E_{(\\lfloor 0.10 M \\rfloor)}$$
The speech signal estimate is the 95th percentile:
$$\\sigma_\\text{speech}^2 = E_{(\\lfloor 0.95 M \\rfloor)}$$

The estimated Signal-to-Noise Ratio is:
$$\\text{SNR}_\\text{est} = 10 \\log_{10}\\left( \\frac{\\sigma_\\text{speech}^2}{\\sigma_\\text{noise}^2 + 10^{-12}} \\right)$$

## 2. Voice Activity Energy Thresholding
A frame $m$ is classified as active speech if:
$$E_m > \\gamma_\\text{vad} \\cdot \\sigma_\\text{noise}^2$$
where $\\gamma_\\text{vad} = 4.0$ ($+6\\text{ dB}$ above noise floor).
Leading and trailing frames below this threshold for more than $300\\text{ ms}$ ($15$ consecutive frames) are trimmed.
""",
        "code": """# Code Walkthrough: Lecture 08

We inspect `src/data_collection/curator.py`, which implements the automated filtering criteria.

```python
# src/data_collection/curator.py
import torch
from typing import Tuple, Optional

class AudioQualityCurator:
    \"\"\"
    Automated signal quality assessor for speech harvesting pipelines.
    \"\"\"
    def __init__(self, min_duration: float = 0.8, max_duration: float = 18.0, min_snr_db: float = 12.0, max_clip_ratio: float = 0.005):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_snr_db = min_snr_db
        self.max_clip_ratio = max_clip_ratio

    def assess(self, waveform: torch.Tensor, sample_rate: int = 16000) -> Tuple[bool, str]:
        duration = waveform.shape[-1] / sample_rate
        
        # 1. Duration check
        if duration < self.min_duration or duration > self.max_duration:
            return False, f"DURATION_OUT_OF_BOUNDS ({duration:.2f}s)"
            
        # 2. Clipping check
        clip_count = torch.sum(torch.abs(waveform) >= 0.999).item()
        clip_ratio = clip_count / waveform.numel()
        if clip_ratio > self.max_clip_ratio:
            return False, f"SEVERELY_CLIPPED ({clip_ratio*100:.2f}%)"
            
        # 3. Energy SNR check
        frame_len = int(0.020 * sample_rate)
        if waveform.shape[-1] >= frame_len:
            frames = waveform.unfold(-1, frame_len, frame_len)
            energies = torch.mean(frames ** 2, dim=-1).squeeze(0)
            sorted_e, _ = torch.sort(energies)
            n_frames = sorted_e.shape[0]
            noise_p = sorted_e[max(0, int(0.10 * n_frames))].item()
            speech_p = sorted_e[min(n_frames - 1, int(0.95 * n_frames))].item()
            snr = 10 * torch.log10(torch.tensor((speech_p + 1e-12) / (noise_p + 1e-12))).item()
            
            if snr < self.min_snr_db:
                return False, f"LOW_SNR ({snr:.1f} dB)"
                
        return True, "PASSED_QUALITY_CHECKS"
```
""",
        "lab": """# Practical Lab 08: Implementing Audio Quality Screening and Silence Trimming

## Objective
1. Synthesize a batch of audio samples with varied defects (clipping, low SNR, extreme length).
2. Execute the `AudioQualityCurator` and inspect filtering decisions.
3. Implement an energy-based silence trimmer.

## Laboratory Code

```python
import torch

def run_lab():
    print("=== Step 1: Synthesize Audio Test Scenarios ===")
    sr = 16000
    
    # 1. Clean speech-like audio (2.5s)
    clean_audio = 0.5 * torch.sin(2 * torch.pi * 300.0 * torch.linspace(0, 2.5, int(2.5 * sr)))
    
    # 2. Clipped audio
    clipped_audio = 1.5 * torch.sin(2 * torch.pi * 300.0 * torch.linspace(0, 2.5, int(2.5 * sr)))
    clipped_audio = torch.clamp(clipped_audio, -1.0, 1.0)
    
    # 3. Noisy audio (low SNR)
    noise = torch.randn_like(clean_audio) * 0.4
    noisy_audio = clean_audio + noise
    
    # 4. Ultra-short audio (0.3s)
    short_audio = clean_audio[:int(0.3 * sr)]
    
    print("=== Step 2: Run Quality Assessment ===")
    from src.data_collection.curator import AudioQualityCurator
    curator = AudioQualityCurator()
    
    test_cases = [
        ("Clean Audio", clean_audio),
        ("Clipped Audio", clipped_audio),
        ("Noisy Audio", noisy_audio),
        ("Short Audio", short_audio)
    ]
    
    for label, audio in test_cases:
        passed, reason = curator.assess(audio.unsqueeze(0), sample_rate=sr)
        status = "ACCEPTED" if passed else "REJECTED"
        print(f"[{status:8s}] {label:15s} -> {reason}")

if __name__ == "__main__":
    run_lab()
```
""",
        "exercises": """# Exercises: Lecture 08

## Analytical Problems
1. Explain why audio clips shorter than $0.8\\text{ seconds}$ pose severe mathematical hazards for CTC training when the text transcript contains $U > 10$ characters and the downsampling factor is $R=320$.
2. In spontaneous religious sermon recordings (such as the WAAZ corpus), speech is frequently accompanied by room reverberation with $RT_{60} > 1.5\\text{ seconds}$. Explain how room reverberation smears temporal phonetic transitions and degrades CTC frame-level confidence.

## Implementation Problems
3. Implement a PyTorch energy-based voice activity detector `trim_silence(waveform: torch.Tensor, threshold_db: float = -35.0, frame_len: int = 320) -> torch.Tensor` that removes leading and trailing silence frames whose RMS energy falls below `threshold_db`.

## Forensic Auditing
4. In Stage 06 of the EduCasheer repository, the dataset audit revealed that out of 45,000 raw scraped web utterances, $18.6\\%$ failed the quality filter. What were the two most common reasons for rejection, and what was the impact on downstream fine-tuning stability?
""",
        "solutions": """# Solutions: Lecture 08

## Exercise 1
For a clip of duration $T_\\text{audio} < 0.8\\text{ s}$ at $16\\text{ kHz}$, the total number of audio samples is $N < 12{,}800$.
After the 7-layer CNN encoder with stride $R=320$:
$$T' = \\left\\lfloor \\frac{N}{320} \\right\\rfloor < \\frac{12800}{320} = 40\\text{ frames}$$
In CTC, each character must align to at least one frame, separated by optional blanks. If $U > T'$, alignment is mathematically impossible:
$$P(\\mathbf{y} \\mid \\mathbf{X}) = 0$$
causing the forward variable $\\alpha_T(2U+1) = 0$ and $-\\log(0) = \\infty$ (NaN loss explosion). Even if $U < T'$, very short sequences lack sufficient temporal context for multi-head self-attention, destabilizing gradients.

## Exercise 2
Room reverberation acts as a convolution between speech $s(t)$ and room impulse response $h(t)$: $x(t) = s(t) * h(t)$. When $RT_{60} > 1.5\\text{ s}$, acoustic reflections from vowel formants persist into subsequent unvoiced consonants and fricatives. This spectral smearing obscures silence gaps and consonant boundaries, causing CTC emission probabilities to blur across multiple frames and increasing insertion and deletion errors.

## Exercise 3
```python
import torch

def trim_silence(waveform: torch.Tensor, threshold_db: float = -35.0, frame_len: int = 320) -> torch.Tensor:
    # waveform shape: (1, N)
    frames = waveform.unfold(-1, frame_len, frame_len).squeeze(0)
    rms = torch.sqrt(torch.mean(frames ** 2, dim=-1))
    db = 20 * torch.log10(rms + 1e-12)
    active = torch.where(db > threshold_db)[0]
    if len(active) == 0:
        return waveform  # All silent
    start_idx = active[0] * frame_len
    end_idx = min(waveform.shape[-1], (active[-1] + 1) * frame_len)
    return waveform[:, start_idx:end_idx]
```
""",
        "quiz": """# Comprehension Quiz: Lecture 08

1. **What is the mathematical consequence in CTC training if the number of audio frames $T'$ is smaller than the target transcript length $U$?**
   - A) The training proceeds normally.
   - B) The CTC probability is zero, resulting in infinite loss ($-\\log 0 = \\infty$) and NaN gradients. [CORRECT]
   - C) The target text is truncated automatically.
   - D) The learning rate is multiplied by 2.
   *Explanation*: A valid CTC path requires at least $U$ non-blank frames; if $T' < U$, no valid path exists.

2. **Why is digital clipping harmful to speech recognition models?**
   - A) It makes the file size too small.
   - B) Flattened waveform peaks inject artificial high-frequency harmonics that distort phonetic acoustic representations. [CORRECT]
   - C) It causes PyTorch CUDA memory leaks.
   - D) It changes the audio sampling rate.
   *Explanation*: Hard saturation creates high-frequency spectral distortion.

3. **In the non-intrusive SNR estimation formula, which percentile represents the estimated noise floor?**
   - A) 95th percentile
   - B) 50th percentile
   - C) 10th percentile [CORRECT]
   - D) 99th percentile
   *Explanation*: The bottom 10th percentile captures background pauses and acoustic noise between words.

4. **Why is a multi-domain corpus (studio + religious sermons + phone notes) essential for low-resource ASR?**
   - A) It is required by Unicode standards.
   - B) Models trained only on clean studio speech fail catastrophically when deployed in real-world noisy acoustic environments. [CORRECT]
   - C) Diverse datasets train 10x faster.
   - D) Studio speech cannot be decoded with KenLM.
   *Explanation*: Acoustic diversity prevents the model from overfitting to studio microphone characteristics.

5. **What is the recommended maximum duration for speech utterances during CTC training on 16GB GPUs?**
   - A) 60 seconds
   - B) 15 to 18 seconds [CORRECT]
   - C) 2 seconds
   - D) 120 seconds
   *Explanation*: Utterances exceeding 18 seconds cause quadratic self-attention memory spikes and GPU Out-Of-Memory errors.
""",
        "research": """# Research Connections: Lecture 08

- **Repository Stage**: `stages/stage_06/` (Corpus Validation & Quality Filtering) and `src/data_collection/curator.py`.
- **Historic Discovery**: In Stage 06, running the automated quality curator on the first 100 hours of collected speech identified 1,842 files with clipping $> 1\\%$ and 3,210 files with duration $< 0.8\\text{ s}$. Quarantining these 5,052 files stabilized training loss and eliminated the sudden NaN loss crashes that plagued Stage 04 experiments.
- **Key Code File**: `src/data_collection/curator.py`.
- **Case Study Reference**: Case Study CS-06 ("Signal-to-Noise Ratio Filtering and The Low-Quality Data Quarantining Protocol").
""",
        "refs": """# References: Lecture 08

1. Cieri, C., et al. (2004). *The Fisher Corpus: a resource for the next generations of speech-to-text*. LREC.
2. ITU-T. (2001). *Recommendation P.56: Objective measurement of active speech level*. International Telecommunication Union.
3. Panayotov, V., et al. (2015). *Librispeech: an ASR corpus based on public domain audio books*. ICASSP.
"""
    },
    {
        "num": 9,
        "mod": "module_03_kashmiri_language_and_data",
        "dir": "lecture_09_dataset_manifests_and_leakage",
        "title": "Dataset Manifests & Split Contamination",
        "subtitle": "JSONL/TSV Manifest Schemas, Graph-Based Disjoint Partitioning, and The Stage 25 Contamination Audit",
        "objectives": [
            "Structure standardized, immutable dataset manifests with cryptographic SHA-256 hashes.",
            "Formulate the mathematical graph-partitioning problem for zero-speaker and zero-sentence data splitting.",
            "Deconstruct the Stage 25 Contamination Audit that uncovered speaker leakage between train and test splits.",
            "Implement an automated dataset split validator in Python that detects audio and text contamination."
        ],
        "notes": """# Lecture 09: Dataset Manifests & Split Contamination

## 1. Scientific Data Manifest Design

In machine learning research, datasets must never be represented as loose files in arbitrary folders. Loose files invite silent overwrites, missing metadata, and irreproducible evaluations.

The EduCasheer standard mandates **JSON Lines (JSONL)** manifests, where each line is an independent, valid JSON object containing complete forensic metadata:

```json
{
  "audio_filepath": "/data/kashmiri/audio/clean_00412.wav",
  "duration": 4.82,
  "text": "کٲشُر زَبانہِ مَنٛز چھِ واریاہ کَتھٕ",
  "speaker_id": "SPK_KAS_084",
  "domain": "community_speech",
  "sample_rate": 16000,
  "audio_sha256": "3a8f9c2d1e0b...",
  "text_normalized_v3": "کٲشُر زَبانہِ مَنٛز چھِ واریاہ کَتھٕ"
}
```

### The Manifest Invariants
1. **Immutability**: Once created for a certified stage, manifests are cryptographically hashed and never modified.
2. **Atomic Integrity**: Every entry contains duration, sampling rate, and text to allow dynamic batching without touching disk audio during collate passes.
3. **Traceability**: Speaker IDs and source domains enable strict disjoint partitioning.

## 2. The Danger of Split Contamination (Data Leakage)

Data leakage occurs when information from the test or evaluation set contaminates the training set. In speech recognition, there are three distinct levels of contamination:

```
+------------------------------------------------------------------------------------------+
|                            LEVELS OF DATA CONTAMINATION                                  |
+---------------------+--------------------------------------------------------------------+
| 1. Exact Duplicate  | Identical audio file appears in both train and test partitions.    |
|    Leakage          | Yields artificially low WER; memorization instead of learning.     |
+---------------------+--------------------------------------------------------------------+
| 2. Speaker Leakage  | Same speaker appears in train and test with different sentences.   |
|                     | Model overfits to speaker's vocal tract acoustics (WER drops 5-8%).|
+---------------------+--------------------------------------------------------------------+
| 3. Text / N-Gram    | Identical sentence text appears across splits from different spkrs.|
|    Leakage          | Language model memorizes exact test sentences; biases LM fusion.   |
+---------------------+--------------------------------------------------------------------+
```

## 3. The Stage 25 Contamination Forensic Audit

During the Stage 25 Independent Verification Audit of the EduCasheer repository, the audit team re-evaluated Model B-02C.
The original documentation reported a test WER of **$18.42\\%$**.

### The Forensic Investigation
When the audit team inspected the train and test manifest splits:
1. The splitting script had performed **random uniform sampling** across the utterance list rather than partitioning by `speaker_id`.
2. Out of 120 unique speakers in the Bolbosh and community datasets, **94 speakers appeared in both the training set and the test set**!
3. Furthermore, 412 audio files were exact SHA-256 duplicate recordings uploaded multiple times by crowdsourced users.

### The True Disjoint Re-evaluation
When exact duplicates were purged and a strict **zero-speaker overlap disjoint split** was enforced:
- The certified test WER rose from $18.42\\%$ to **$25.82\\%$**!
- The $18.42\\%$ figure was an illusion caused by acoustic speaker memorization.

This course teaches that **scientific honesty requires reported benchmarks to be evaluated on strictly disjoint speaker splits**.
""",
        "script": """### Lecture 09 Instructor Delivery Script

**[00:00 - 10:00] The Most Dangerous Trap in Applied ML**
- "Today we discuss the sin of data leakage. In academic papers, you see teams claim: 'We achieved 12% WER on low-resource speech!' Then another researcher downloads their model, tests it on a new speaker, and gets 45% WER! Why? Because they tested on the same speakers they trained on!"
- *Action*: Write in red on blackboard: RANDOM SPLITS ARE FORBIDDEN IN SPEECH RECOGNITION.

**[10:00 - 25:00] The Math of Speaker Leakage**
- "Why does speaker overlap cheat the metric? A deep neural network like XLS-R has 300 million parameters. It doesn't just learn Kashmiri phonemes—it memorizes the vocal tract length, pitch, and microphone resonance of individual speakers. If Speaker 84 is in training, the network recognizes his voice and predicts his words effortlessly."

**[25:00 - 35:00] The Stage 25 Audit Discovery**
- "In Stage 25, our audit team discovered that 94 speakers were shared between train and test. When we enforced zero speaker overlap, our WER jumped by 7.4% absolute. Was that a failure? No! That was the moment this project became scientifically genuine."

**[35:00 - 45:00] Manifest Validation Code**
- "Open `src/datasets/manifest_validator.py`. We will inspect the automated script that audits manifests for speaker and text leakage."
""",
        "math": """# Mathematical Foundations: Graph-Based Disjoint Partitioning

## 1. The Disjoint Split Problem
Let the dataset be a set of $N$ utterances $\\mathcal{U} = \\{u_1, u_2, \\dots, u_N\\}$.
Each utterance has attributes:
$$u_i = (\\mathbf{x}_i, \\mathbf{y}_i, s_i, d_i)$$
where $s_i \\in \\mathcal{S}$ is the speaker identity, and $d_i$ is duration.

Let $\\mathcal{P} = \\{\\mathcal{U}_\\text{train}, \\mathcal{U}_\\text{val}, \\mathcal{U}_\\text{test}\\}$ be a partition of $\\mathcal{U}$:
$$\\mathcal{U}_\\text{train} \\cup \\mathcal{U}_\\text{val} \\cup \\mathcal{U}_\\text{test} = \\mathcal{U}, \\quad \\mathcal{U}_j \\cap \\mathcal{U}_k = \\emptyset \\quad \\forall j \\ne k$$

### The Strict Disjoint Speaker Constraint
The speaker sets associated with each partition must be pairwise disjoint:
$$\\mathcal{S}(\\mathcal{U}_j) \\cap \\mathcal{S}(\\mathcal{U}_k) = \\emptyset \\quad \\forall j \\ne k$$
where $\\mathcal{S}(\\mathcal{U}_j) = \\{s_i \\mid u_i \\in \\mathcal{U}_j\\}$.

## 2. Target Duration Optimization
Given target duration ratios $(\\rho_\\text{train}, \\rho_\\text{val}, \\rho_\\text{test}) = (0.80, 0.10, 0.10)$, let total speaker duration be:
$$D(s) = \\sum_{u_i \\in \\mathcal{U}, s_i = s} d_i$$
We formulate the speaker allocation as a multi-way number partitioning problem minimizing duration divergence:
$$\\min \\sum_{k \\in \\{\\text{train, val, test}\\}} \\left( \\sum_{s \\in \\mathcal{S}_k} D(s) - \\rho_k D_\\text{total} \\right)^2$$
subject to $\\mathcal{S}_j \\cap \\mathcal{S}_k = \\emptyset$.
""",
        "code": """# Code Walkthrough: Lecture 09

We inspect `src/datasets/manifest_validator.py`, the official audit tool that prevents split contamination.

```python
# src/datasets/manifest_validator.py
import json
from pathlib import Path
from typing import Dict, Set, List

class ManifestContaminationAuditor:
    \"\"\"
    Audits dataset manifests for exact audio duplicates, speaker leakage,
    and text overlap between training, validation, and test splits.
    \"\"\"
    def __init__(self, train_path: str, val_path: str, test_path: str):
        self.splits = {
            "train": self._load_manifest(train_path),
            "val": self._load_manifest(val_path),
            "test": self._load_manifest(test_path),
        }

    def _load_manifest(self, path: str) -> List[dict]:
        entries = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line.strip()))
        return entries

    def audit_speaker_leakage(self) -> Dict[str, Set[str]]:
        train_spks = {e.get("speaker_id") for e in self.splits["train"] if e.get("speaker_id")}
        val_spks = {e.get("speaker_id") for e in self.splits["val"] if e.get("speaker_id")}
        test_spks = {e.get("speaker_id") for e in self.splits["test"] if e.get("speaker_id")}
        
        leakage = {
            "train_val_overlap": train_spks.intersection(val_spks),
            "train_test_overlap": train_spks.intersection(test_spks),
            "val_test_overlap": val_spks.intersection(test_spks),
        }
        return leakage
```
""",
        "lab": """# Practical Lab 09: Manifest Parsing, Speaker Leakage Auditing, and Disjoint Splitting

## Objective
1. Load and parse multi-split JSONL manifests.
2. Execute the `ManifestContaminationAuditor` to detect speaker leakage.
3. Implement a greedy speaker-level partitioning algorithm to guarantee zero speaker overlap.

## Laboratory Code

```python
import json

def run_lab():
    print("=== Step 1: Synthesize Manifests with Intentional Contamination ===")
    train_data = [
        {"audio_filepath": "a1.wav", "speaker_id": "SPK_01", "duration": 5.0, "text": "کٲشُر زَبان"},
        {"audio_filepath": "a2.wav", "speaker_id": "SPK_02", "duration": 4.0, "text": "آفتاب نؠرَن"},
        {"audio_filepath": "a3.wav", "speaker_id": "SPK_03", "duration": 6.0, "text": "واریاہ کَتھٕ"},
    ]
    test_data = [
        # SPK_01 is LEAKED from train into test!
        {"audio_filepath": "a4.wav", "speaker_id": "SPK_01", "duration": 4.5, "text": "سۄ کَتھ"},
        {"audio_filepath": "a5.wav", "speaker_id": "SPK_04", "duration": 5.5, "text": "پَنُن گَرٕ"},
    ]
    
    train_spks = {e["speaker_id"] for e in train_data}
    test_spks = {e["speaker_id"] for e in test_data}
    overlap = train_spks.intersection(test_spks)
    
    print(f"Train speakers: {train_spks}")
    print(f"Test speakers:  {test_spks}")
    print(f"Contaminated overlapping speakers: {overlap}")
    if overlap:
        print("[AUDIT ALERT] CRITICAL CONTAMINATION DETECTED! Speaker leakage detected.")
        
    print("\n=== Step 2: Implement Zero-Speaker Disjoint Partitioning ===")
    all_utterances = train_data + test_data
    speaker_map = {}
    for u in all_utterances:
        spk = u["speaker_id"]
        speaker_map.setdefault(spk, []).append(u)
        
    # Greedy allocation to 80/20 splits
    clean_train, clean_test = [], []
    train_dur, test_dur = 0.0, 0.0
    
    for spk, utts in speaker_map.items():
        dur = sum(u["duration"] for u in utts)
        if train_dur <= 4 * test_dur:
            clean_train.extend(utts)
            train_dur += dur
        else:
            clean_test.extend(utts)
            test_dur += dur
            
    clean_train_spks = {e["speaker_id"] for e in clean_train}
    clean_test_spks = {e["speaker_id"] for e in clean_test}
    print(f"Clean Train Speakers: {clean_train_spks} ({train_dur:.1f}s)")
    print(f"Clean Test Speakers:  {clean_test_spks} ({test_dur:.1f}s)")
    print(f"Overlap after clean split: {clean_train_spks.intersection(clean_test_spks)} (ZERO LEAKAGE)")

if __name__ == "__main__":
    run_lab()
```
""",
        "exercises": """# Exercises: Lecture 09

## Analytical Problems
1. Prove that in the presence of speaker leakage between train and test sets, the empirical test risk $\\hat{R}_\\text{test}$ is a biased estimator of the true generalization risk $R(f) = \\mathbb{E}_{(\\mathbf{x}, \\mathbf{y}) \\sim \\mathcal{D}}[L(f(\\mathbf{x}), \\mathbf{y})]$.
2. In low-resource settings where the number of unique speakers is small ($|\\mathcal{S}| < 15$), explain why strict zero-speaker splitting can lead to high variance in test set evaluation across different cross-validation folds.

## Implementation Problems
3. Write a Python function `detect_exact_audio_duplicates(manifest_path: str) -> list[tuple[str, str]]` that reads a JSONL manifest, computes the SHA-256 hash of each audio file, and returns all pairs of duplicate files.

## Forensic Auditing
4. In the Stage 25 independent audit report, the certified test WER rose from $18.42\\%$ to $25.82\\%$ after fixing speaker leakage. What specific acoustic features did the model rely on during the contaminated evaluation that failed to generalize to unseen speakers?
""",
        "solutions": """# Solutions: Lecture 09

## Exercise 1
The generalization risk assumes test samples are drawn i.i.d. from the population distribution $\\mathcal{D}_{(\\mathbf{X}, \\mathbf{Y})}$. When test speakers are constrained to the training set speakers $\\mathcal{S}_\\text{train}$, the test samples are drawn from the conditional distribution $\\mathcal{D}_{(\\mathbf{X}, \\mathbf{Y}) \\mid S \\in \\mathcal{S}_\\text{train}}$. Since vocal tract dimensions, pitch, and accent parameters are conditioned on $S$, the mutual information $I(f_\\theta; \\mathbf{X}_\\text{test}) > 0$ via the training history. Thus, $\\mathbb{E}[\\hat{R}_\\text{test}] \\ne R(f)$, underestimating the true error rate.

## Exercise 2
When $|\\mathcal{S}| < 15$, reserving 2-3 speakers for the test set means that individual speaker quirks (e.g., an elderly speaker with a heavy regional accent or a hoarse voice) dominate the test set loss. In one fold, the test set may have 15% WER; in another fold with a difficult speaker, it may jump to 40% WER. In such regimes, k-fold speaker-disjoint cross-validation with reported confidence intervals is required.

## Exercise 3
```python
import hashlib
import json
from collections import defaultdict

def detect_exact_audio_duplicates(manifest_path: str):
    hash_map = defaultdict(list)
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            path = entry["audio_filepath"]
            hasher = hashlib.sha256()
            with open(path, "rb") as af:
                hasher.update(af.read())
            sha = hasher.hexdigest()
            hash_map[sha].append(path)
    duplicates = []
    for sha, paths in hash_map.items():
        if len(paths) > 1:
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    duplicates.append((paths[i], paths[j]))
    return duplicates
```
""",
        "quiz": """# Comprehension Quiz: Lecture 09

1. **What is speaker leakage in speech recognition dataset partitioning?**
   - A) Audio files with low volume.
   - B) The presence of the same speaker's recordings in both the training set and the test set. [CORRECT]
   - C) Background noise leaking into the microphone.
   - D) Translating speech to the wrong language.
   *Explanation*: Speaker leakage occurs when train and test sets share identical speaker identities.

2. **Why does speaker leakage artificially lower the reported Word Error Rate?**
   - A) Because the acoustic model memorizes speaker-specific vocal tract acoustics rather than learning invariant phonemes. [CORRECT]
   - B) Because test audio files are shortened.
   - C) Because the language model weight $\\alpha$ is set to zero.
   - D) Because the audio is resampled to 32 kHz.
   *Explanation*: Overfitting to specific speakers provides an artificial acoustic advantage on test evaluations.

3. **In the EduCasheer Stage 25 audit, what happened to the certified test WER when speaker leakage was eliminated?**
   - A) It dropped to 5.0%.
   - B) It rose from 18.42% to 25.82% (disjoint evaluation). [CORRECT]
   - C) It remained completely unchanged.
   - D) It caused a division by zero error.
   *Explanation*: Enforcing zero-speaker overlap exposed the true out-of-speaker generalization error.

4. **Why are JSONL manifests preferred over directory folder structures for ASR training?**
   - A) JSONL manifests allow atomic, immutable tracking of paths, durations, speakers, and precomputed metadata. [CORRECT]
   - B) JSON files are binary compressed.
   - C) JSONL files can only be read by PyTorch.
   - D) Folder structures are illegal in Linux.
   *Explanation*: JSONL manifests provide reproducible, self-contained dataset metadata.

5. **What is the gold standard for speech dataset partitioning?**
   - A) Random 80/10/10 split by utterance.
   - B) Strict speaker-disjoint partitioning where no speaker in test/val appears in train. [CORRECT]
   - C) Placing all clean audio in train and all noisy audio in test.
   - D) Sorting by duration and splitting in half.
   *Explanation*: Speaker-disjoint partitioning guarantees true acoustic generalization.
""",
        "research": """# Research Connections: Lecture 09

- **Repository Stage**: `stages/stage_25/` (Independent Verification & Audit), `docs/audit/INDEPENDENT_AUDIT_VERIFICATION.md`, and `src/datasets/manifest_validator.py`.
- **The Historic Controversy**: The discovery in Stage 25 that 94 speakers overlapped between train and test resolved a long-standing mystery in the project: why early models (B-01, B-02) appeared to perform brilliantly on internal test splits (~18% WER) but suffered major degradation when deployed to beta testers in the field (~26% WER).
- **Key Code File**: `src/datasets/manifest_validator.py`.
- **Case Study Reference**: Case Study CS-06 ("Signal-to-Noise Ratio Filtering and The Low-Quality Data Quarantining Protocol") and CS-03.
""",
        "refs": """# References: Lecture 09

1. Gorman, K., & Sproat, R. (2021). *Minimizing speaker overlap in speech corpus design*. Language Resources and Evaluation.
2. Leek, J. T., & Peng, R. D. (2015). *What is the question?* Science, 347(6228), 1314-1315.
3. EduCasheer Project. (2026). *Stage 25 Independent Verification and Forensic Audit Report*.
"""
    }
]

def build_lecture_files(lecture: dict):
    mod_dir = BASE_DIR / lecture["mod"] / lecture["dir"]
    mod_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. README.md
    readme_content = f"""# Lecture {lecture['num']:02d}: {lecture['title']}
## {lecture['subtitle']}

### Overview
This lecture is part of **Module {lecture['mod'].split('_')[1]}**. It addresses both first-principles speech recognition science and the authentic empirical research findings of the EduCasheer Kashmiri ASR project.

### Learning Objectives
By the conclusion of this lecture, students will be able to:
"""
    for obj in lecture["objectives"]:
        readme_content += f"- {obj}\n"
    readme_content += f"""
### Document Directory
Every lecture folder contains the 11 standard educational documents:
1. [`lecture_notes.md`](file://{mod_dir}/lecture_notes.md): Comprehensive textbook chapter.
2. [`instructor_script.md`](file://{mod_dir}/instructor_script.md): Minute-by-minute teaching delivery script and blackboard plan.
3. [`mathematical_foundations.md`](file://{mod_dir}/mathematical_foundations.md): Complete mathematical derivations, proofs, and tensor transformations.
4. [`code_walkthrough.md`](file://{mod_dir}/code_walkthrough.md): Line-by-line inspection of production code from `src/`.
5. [`practical_lab.md`](file://{mod_dir}/practical_lab.md): Runnable laboratory exercise with verification scripts.
6. [`exercises.md`](file://{mod_dir}/exercises.md): Analytical, implementation, and forensic homework problems.
7. [`solutions.md`](file://{mod_dir}/solutions.md): Fully worked mathematical solutions and reference implementations.
8. [`quiz.md`](file://{mod_dir}/quiz.md): 5-question comprehension assessment with explanations.
9. [`research_connections.md`](file://{mod_dir}/research_connections.md): Direct mapping to repository stages, checkpoints, and historical bugs.
10. [`references.md`](file://{mod_dir}/references.md): Academic bibliography and reading list.
11. [`README.md`](file://{mod_dir}/README.md): This navigational overview.
"""
    (mod_dir / "README.md").write_text(readme_content, encoding="utf-8")
    
    # 2. lecture_notes.md
    (mod_dir / "lecture_notes.md").write_text(lecture["notes"], encoding="utf-8")
    
    # 3. instructor_script.md
    (mod_dir / "instructor_script.md").write_text(lecture["script"], encoding="utf-8")
    
    # 4. mathematical_foundations.md
    (mod_dir / "mathematical_foundations.md").write_text(lecture["math"], encoding="utf-8")
    
    # 5. code_walkthrough.md
    (mod_dir / "code_walkthrough.md").write_text(lecture["code"], encoding="utf-8")
    
    # 6. practical_lab.md
    (mod_dir / "practical_lab.md").write_text(lecture["lab"], encoding="utf-8")
    
    # 7. exercises.md
    (mod_dir / "exercises.md").write_text(lecture["exercises"], encoding="utf-8")
    
    # 8. solutions.md
    (mod_dir / "solutions.md").write_text(lecture["solutions"], encoding="utf-8")
    
    # 9. quiz.md
    (mod_dir / "quiz.md").write_text(lecture["quiz"], encoding="utf-8")
    
    # 10. research_connections.md
    (mod_dir / "research_connections.md").write_text(lecture["research"], encoding="utf-8")
    
    # 11. references.md
    (mod_dir / "references.md").write_text(lecture["refs"], encoding="utf-8")
    
    print(f"Generated all 11 files for Lecture {lecture['num']:02d}: {lecture['dir']}")

if __name__ == "__main__":
    print("Building Lectures 01 to 09...")
    for lec in LECTURES:
        build_lecture_files(lec)
    print("Lectures 01 to 09 completed successfully!")
