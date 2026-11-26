#!/usr/bin/env python3
"""
Pristine LaTeX Enhancement Script for Module 01, Lecture 02:
"The Anatomy of Modern ASR Pipelines"

Fixes all GitHub Flavored Markdown (GFM) / KaTeX math rendering issues:
- Eliminates non-Latin Perso-Arabic text from math mode delimiters.
- Eliminates inline backtick-math collisions.
- Replaces floor brackets with dynamic \left\lfloor ... \right\rfloor.
- Replaces raw LaTeX commands outside math mode with proper delimiters.
- Unindents all $$ display math to column 0 with surrounding blank lines.
- Uses Unicode in Markdown table headers.
"""

from pathlib import Path

BASE_DIR = Path("/home/musa/Kashmiri Project")
LECTURE_DIR = BASE_DIR / "teaching/modules/module_01_introduction/lecture_02_asr_anatomy"

# ---------------------------------------------------------------------------
# 1. README.md
# ---------------------------------------------------------------------------
README_CONTENT = r"""# Lecture 02: The Anatomy of Modern ASR Pipelines

## Executive Overview

Lecture 02 breaks down the physical, algorithmic, and mathematical anatomy of modern Automatic Speech Recognition (ASR) pipelines. Moving beyond the classical modular paradigm (Acoustic Model + Pronunciation Lexicon + Language Model composed into a Weighted Finite-State Transducer), this lecture explores why modern speech recognition transitioned to unified End-to-End (E2E) neural architectures, and dissects the specific 6-stage production pipeline developed in the EduCasheer Kashmiri ASR research project.

Students will examine the fundamental trade-offs between the three dominant modern neural paradigms:
1. **Connectionist Temporal Classification (CTC)**
2. **Attention-based Encoder-Decoder (AED)** (e.g., Whisper, Listen-Attend-Spell)
3. **Recurrent Neural Network Transducer (RNN-T)**

Special emphasis is placed on why CTC over a self-supervised acoustic backbone (XLS-R 300M) proved uniquely effective for Kashmiri, avoiding the catastrophic hallucination loops and severe insertion penalties of autoregressive decoders on low-resource speech.

---

## Pedagogical Context & Learning Objectives

By the conclusion of this lecture, students will be able to:
1. **Contrast Architectural Paradigms**: Articulate why classical Kaldi-style hybrid HMM-GMM/DNN pipelines failed for Kashmiri due to the absence of a comprehensive phonetic lexicon ($L$), and explain why CTC guarantees alignment monotonicity whereas autoregressive AED models suffer from hallucination loops on silence and noise.
2. **Trace the 6-Stage EduCasheer Production Stack**: Walk step-by-step through the real-world inference architecture implemented in [`src/asr/transcribe.py`](file:///home/musa/Kashmiri%20Project/src/asr/transcribe.py), from raw 16 kHz audio loading through Kaiser sinc resampling, temporal Conv1d feature extraction, 24-layer Transformer encoding, temperature-calibrated logit slicing, KenLM prefix beam search, and Certified Normalizer v3.
3. **Derive Receptive Field and CTC Frame Slicing**: Calculate the exact receptive field ($400$ samples / $25$ ms) and cumulative downsampling stride ($320$ samples / $20$ ms / $50$ Hz frame rate) of the 7-layer temporal convolutional feature extractor, proving why $T = \left\lfloor \frac{N - 400}{320} \right\rfloor + 1$.
4. **Evaluate Vocabulary Design Trade-offs**: Contrast subword Byte-Pair Encoding (BPE) against character-level modeling ($|V|=47$) for Perso-Arabic scripts, explaining why BPE fragments Kashmiri morphological inflections and induces out-of-vocabulary catastrophes.
5. **Formulate the Shallow Fusion Decoding Objective**: Analyze the log-linear scoring function $\mathcal{S}(\mathbf{w}) = \log P_{\text{CTC}}(\mathbf{w} \mid \mathbf{X}) + \alpha \log P_{\text{LM}}(\mathbf{w}) + \beta |\mathbf{w}|$, proving the necessity of the word insertion penalty $\beta$ to counteract character-level brevity bias.
6. **Enforce Disjoint Evaluation Discipline**: Explain why random data splitting in speech recognition causes severe speaker leakage (artificially deflating WER by 10–18% absolute), and how bipartite graph partitioning guarantees zero-speaker-overlap evaluation integrity.

---

## Document Navigation

| File | Pedagogical Purpose |
| :--- | :--- |
| [`lecture_notes.md`](file:///home/musa/Kashmiri%20Project/teaching/modules/module_01_introduction/lecture_02_asr_anatomy/lecture_notes.md) | Comprehensive 340+ line textbook chapter covering the complete anatomy of modern ASR, paradigm trade-offs, pipeline stages, vocabulary design, and decoding mechanics. |
| [`instructor_script.md`](file:///home/musa/Kashmiri%20Project/teaching/modules/module_01_introduction/lecture_02_asr_anatomy/instructor_script.md) | Turn-by-turn 90-minute lecture delivery guide with minute-by-minute cues, spoken dialogue, 3-panel blackboard diagrams, and student misconception interventions. |
| [`mathematical_foundations.md`](file:///home/musa/Kashmiri%20Project/teaching/modules/module_01_introduction/lecture_02_asr_anatomy/mathematical_foundations.md) | Rigorous mathematical derivations: Conv1d receptive field recursion, CTC collapse operator $\mathcal{B}$ preimage bounds, temperature scaling derivatives, and brevity bias proofs. |
| [`code_walkthrough.md`](file:///home/musa/Kashmiri%20Project/teaching/modules/module_01_introduction/lecture_02_asr_anatomy/code_walkthrough.md) | Exhaustive code inspection tracing [`src/asr/transcribe.py`](file:///home/musa/Kashmiri%20Project/src/asr/transcribe.py), [`src/decoding/ctc_slice.py`](file:///home/musa/Kashmiri%20Project/src/decoding/ctc_slice.py), [`src/decoding/kenlm_decoder.py`](file:///home/musa/Kashmiri%20Project/src/decoding/kenlm_decoder.py), and [`src/normalization/normalizer.py`](file:///home/musa/Kashmiri%20Project/src/normalization/normalizer.py). |
| [`practical_lab.md`](file:///home/musa/Kashmiri%20Project/teaching/modules/module_01_introduction/lecture_02_asr_anatomy/practical_lab.md) | Runnable hands-on laboratory exercises implementing Conv1d feature slicing, pure Python prefix beam search, and temperature-entropy analysis. |
| [`exercises.md`](file:///home/musa/Kashmiri%20Project/teaching/modules/module_01_introduction/lecture_02_asr_anatomy/exercises.md) | 4 multi-part analytical and implementation problems on receptive fields, CTC alignment combinatorics, brevity bias calibration, and speaker leakage graph partitioning. |
| [`solutions.md`](file:///home/musa/Kashmiri%20Project/teaching/modules/module_01_introduction/lecture_02_asr_anatomy/solutions.md) | Complete step-by-step mathematical proofs, numerical answers, and reference Python code. |
| [`quiz.md`](file:///home/musa/Kashmiri%20Project/teaching/modules/module_01_introduction/lecture_02_asr_anatomy/quiz.md) | 5 challenging diagnostic multiple-choice questions with full correct answer justifications and distractor post-mortems. |
| [`research_connections.md`](file:///home/musa/Kashmiri%20Project/teaching/modules/module_01_introduction/lecture_02_asr_anatomy/research_connections.md) | Direct links to research stages (Stage 01, Stage 04, Stage 09, Stage 16, Stage 22, Stage 26) and authentic checkpoint artifacts (`E22-01`). |
| [`references.md`](file:///home/musa/Kashmiri%20Project/teaching/modules/module_01_introduction/lecture_02_asr_anatomy/references.md) | Annotated academic bibliography including seminal papers by Graves, Baevski, Babu, Hannun, Radford, and Koul. |
"""

# ---------------------------------------------------------------------------
# 2. lecture_notes.md
# ---------------------------------------------------------------------------
LECTURE_NOTES_CONTENT = r"""# Lecture 02: The Anatomy of Modern ASR Pipelines

---

## 1. Architectural Paradigms: From Hybrid WFSTs to End-to-End Neural Networks

The history of Automatic Speech Recognition (ASR) is the history of representation learning replacing hand-engineered pipeline modularity. For nearly three decades (from the late 1980s until approximately 2015), the field was dominated by **Classical Hybrid Architectures**. To understand why the EduCasheer Kashmiri project adopted its specific modern architecture, we must contrast the three major historical and modern design paradigms.

```
PARADIGM 1: CLASSICAL HYBRID (HMM-GMM / HMM-DNN)
[Audio] ──► [Feature Extraction: MFCC] ──► [Acoustic Model P(X|s)] ──► ┐
                                                                        ▼
[Dictionary: L] ───────────────────────────────────────────────► [WFST Decoder: H ∘ C ∘ L ∘ G] ──► [Text]
                                                                        ▲
[Text Corpus] ──► [Language Model P(W)] ───────────────────────────────┘

PARADIGM 2: ATTENTION-BASED ENCODER-DECODER (AED / Whisper)
[Audio] ──► [Acoustic Encoder] ──► [Cross-Attention] ◄── [Autoregressive Decoder] ◄── [y_{<u}] ──► [Text]
                                           │
                                           └── (Non-monotonic, prone to hallucination loops)

PARADIGM 3: CONNECTIONIST TEMPORAL CLASSIFICATION (CTC + Pretrained Encoder)
[Audio] ──► [7-layer CNN Feature Extractor] ──► [24-layer XLS-R Transformer] ──► [Linear Head] ──► [Logits]
                                                                                                      │
                                                                                                      ▼
[Text] ◄── [Certified Normalizer v3] ◄── [Prefix Beam Search: CTC Logits + α KenLM + β Length] ◄──────┘
```

---

### 1.1 The Classical Hybrid Pipeline (Kaldi WFST Formulation)

In classical hybrid systems (such as Kaldi), the speech recognition problem is decomposed into strictly decoupled probabilistic models:
1. **Acoustic Model (AM)**: Models $P(\mathbf{X} \mid s)$, representing the probability of acoustic feature vectors $\mathbf{X} = (\mathbf{x}_1, \dots, \mathbf{x}_T)$ (typically 39-dimensional Mel-Frequency Cepstral Coefficients with $\Delta$ and $\Delta\Delta$) given a sequence of hidden HMM states $s \in \mathcal{S}$.
2. **Pronunciation Lexicon ($L$)**: A deterministic or probabilistic bipartite mapping from words to sequences of phonetic phone symbols:

$$L: \mathcal{W} \to \mathcal{P}^+$$

   For example, the Kashmiri word **"كٲشُر"** maps to the phone sequence `/k/ /əː/ /ʃ/ /u/ /r/`.

3. **Language Model ($G$)**: An $n$-gram or recurrent neural network modeling prior word sequence probability $P(\mathbf{W})$.
4. **Weighted Finite-State Transducer (WFST) Composition**: The individual components are compiled into finite-state transducers and statically composed:

$$\mathcal{H} \circ \mathcal{C} \circ \mathcal{L} \circ \mathcal{G}$$

where $\mathcal{H}$ represents HMM state topology, $\mathcal{C}$ maps context-dependent triphones to monophones, $\mathcal{L}$ maps phones to words, and $\mathcal{G}$ represents the grammar automaton.

#### Why the Hybrid Model Failed Catastrophically for Kashmiri:
- **The Lexicon Bottleneck**: Building $\mathcal{L}$ requires thousands of person-hours of expert phonetician labor. For English, the CMU Pronouncing Dictionary contains over 134,000 words. For Kashmiri, no standardized, machine-readable phonetic pronunciation lexicon existed.
- **Orthographic Non-Standardization**: Kashmiri Perso-Arabic text frequently omits diacritics marking critical short and central vowels (/ə/, /ɨ/, /u/, /o/). Without diacritics, orthographic words are phonetically ambiguous, causing the lexicon mapper $L$ to explode with homographic pronunciations or fail completely.
- **Alignment Failure in Stage 01**: In EduCasheer Stage 01, when an Urdu Kaldi hybrid system was adapted to Kashmiri, forced alignment rejected **38.2% of all training utterances** because the acoustic features could not be matched to the acoustic states of the Urdu lexicon, resulting in an unrecoverable **88.40% Word Error Rate (WER)**.

---

### 1.2 Attention-Based Encoder-Decoder (AED) Architectures

Attention-based Encoder-Decoder models (such as Listen, Attend and Spell [LAS] and OpenAI's Whisper) eliminate the pronunciation lexicon by learning an autoregressive mapping directly from acoustic frames $\mathbf{X}$ to output subword tokens $\mathbf{y} = (y_1, \dots, y_U)$:

$$P(\mathbf{y} \mid \mathbf{X}) = \prod_{u=1}^U P(y_u \mid y_{\lt u}, \mathbf{X}) = \prod_{u=1}^U \text{Softmax}\left( \text{Decoder}(y_{\lt u}, \text{CrossAttention}(\mathbf{H}, \mathbf{Z}_{\text{dec}})) \right)$$

#### Advantages:
- Completely removes the conditional independence assumption; each token $y_u$ is conditioned on the entire history of previously generated tokens $y_{\lt u}$.
- Jointly integrates acoustic and language modeling within a single unified neural network.

#### Severe Failure Modes on Low-Resource Kashmiri:
1. **Catastrophic Hallucination Loops**: In autoregressive cross-attention, there is no structural guarantee of temporal monotonicity. When encountering non-speech segments, ambient background noise, coughing, or prolonged pauses (frequent in field recordings like Kashmiri WAAZ sermons or Bolbosh radio broadcasts), Whisper's decoder attends to its own previous outputs $y_{\lt u}$, entering self-reinforcing repetition loops (e.g., repeating a single phrase 50 times until reaching the maximum sequence length).
2. **Subword Fragmentation in Perso-Arabic**: Whisper utilizes byte-level Byte-Pair Encoding (BPE). In Kashmiri Perso-Arabic, BPE frequently merges consonants with diacritics in common words, but splits identical stems into arbitrary single-byte tokens in rare words, causing extreme insertion errors and high Out-Of-Vocabulary (OOV) distortion.
3. **Quadratic Attention Latency**: Full self-attention across long audio sequences scales as $\mathcal{O}(T^2)$, making streaming and real-time edge deployment computationally prohibitive.

---

### 1.3 Connectionist Temporal Classification (CTC) + Self-Supervised Acoustic Backbones

Connectionist Temporal Classification (Graves et al., 2006) maps an input sequence of length $T$ directly to a sequence of length $U$ ($U \le T$) without requiring intermediate phonetic alignments or autoregressive decoding.

#### Theoretical Formulation:
Let $\Omega$ be the alphabet of target characters, and define $\Omega' = \Omega \cup \{\epsilon\}$, where $\epsilon$ denotes a special non-emitting **blank token**. At each frame $t \in \{1, \dots, T\}$, the acoustic model outputs a posterior probability distribution over $\Omega'$:

$$P(\pi_t \mid \mathbf{X}) = \text{Softmax}(\mathbf{W}_p \mathbf{h}_t + \mathbf{b}_p)$$

A CTC alignment $\boldsymbol{\pi} = (\pi_1, \dots, \pi_T) \in (\Omega')^T$ is mapped to a label sequence $\mathbf{y} = (y_1, \dots, y_U) \in \Omega^U$ via a deterministic collapse operator $\mathcal{B}: (\Omega')^T \to \Omega^{\le T}$, which sequentially collapses consecutive duplicate tokens and removes all blank symbols $\epsilon$:

$$\mathcal{B}(\boldsymbol{\pi}) = \mathbf{y}$$

The conditional probability of the sequence $\mathbf{y}$ is the marginal sum over all valid alignments in the preimage $\mathcal{B}^{-1}(\mathbf{y})$:

$$P(\mathbf{y} \mid \mathbf{X}) = \sum_{\boldsymbol{\pi} \in \mathcal{B}^{-1}(\mathbf{y})} P(\boldsymbol{\pi} \mid \mathbf{X}) = \sum_{\boldsymbol{\pi} \in \mathcal{B}^{-1}(\mathbf{y})} \prod_{t=1}^T P(\pi_t \mid \mathbf{X})$$

#### Why CTC + XLS-R Became the Pillar of EduCasheer:
1. **Strict Alignment Monotonicity**: CTC advances strictly forward in time ($t=1 \to T$). It is physically incapable of hallucinating past phrases or looping indefinitely on silence. When silence or noise is encountered, the model emits blank tokens $\epsilon$.
2. **Mitigating Conditional Independence via XLS-R**: Historically, CTC was criticized for assuming that label emissions $\pi_t$ are conditionally independent given $\mathbf{X}$. However, when CTC is placed atop a 24-layer self-supervised Transformer backbone (XLS-R 300M, Baevski et al., 2020; Babu et al., 2021) pretrained on 436,000 hours of speech across 128 languages, each hidden vector $\mathbf{h}_t$ has a receptive field covering the entire audio sequence. The contextual representations within $\mathbf{h}_t$ resolve phonetic ambiguities without requiring an autoregressive text decoder.
3. **Linear Decoding Complexity**: Finding the best path or performing prefix beam search scales linearly as $\mathcal{O}(T \cdot W)$, where $W$ is the beam width.

---

## 2. Structural Decomposition of the EduCasheer Production Pipeline

The production Kashmiri ASR system is implemented in [`src/asr/transcribe.py`](file:///home/musa/Kashmiri%20Project/src/asr/transcribe.py) and supported by modular components in `src/decoding/`, `src/data_processing/`, and `src/normalization/`. It executes a rigorous 6-stage sequential pipeline:

```
┌────────────────────────────────────────────────────────────────────────┐
│               STAGE 1: AUDIO FRONT-END & RESAMPLING                     │
│  - Raw Audio Input (Any format, sample rate, or channel count)         │
│  - Stereo-to-mono downmixing via channel averaging                    │
│  - Polyphase Kaiser-windowed sinc interpolation to 16,000 Hz           │
│  - Peak amplitude normalization to [-1.0, 1.0]                          │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ torch.Tensor [1, N_samples]
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│             STAGE 2: FEATURE EXTRACTION & XLS-R BACKBONE                │
│  - 7-layer temporal Conv1d feature extractor (stride 320, RF 400)      │
│  - Downsampling from 16,000 Hz to 50 Hz frames (20 ms stride)          │
│  - 24-layer Transformer encoder (d_model=1024, 16 heads, d_ff=4096)    │
│  - Group-convolutional relative positional embeddings                  │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ torch.Tensor [1, T_frames, 1024]
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│              STAGE 3: LINEAR CTC PROJECTION HEAD                        │
│  - Linear transformation: W_p ∈ R^{47 x 1024}, b_p ∈ R^{47}             │
│  - Positive blank bias initialization (b_0 = +3.0) for training        │
│  - Unnormalized acoustic emission logits                                │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ torch.Tensor [1, T_frames, 47]
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│          STAGE 4: TEMPERATURE CALIBRATION & CTC LOGIT SLICING          │
│  - Exact mathematical frame length calculation: T = floor((N-400)/320)+1│
│  - Strict removal of padded frames via slice_exact_logits()            │
│  - Acoustic temperature scaling: z_t / T_calib (T_calib = 1.08)        │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ np.ndarray [T_valid, 47]
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│         STAGE 5: PREFIX BEAM SEARCH & SHALLOW FUSION                    │
│  - KenLM 5-gram Kashmiri language model (Pruned Kneser-Ney)            │
│  - Beam width W = 100, beam prune logp = -10.0, token min logp = -5.0  │
│  - Scoring: S(w) = log P_CTC(w|X) + α log P_LM(w) + β |w|               │
│  - Optimal calibrated hyperparameters: α = 0.50, β = 1.50              │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ Raw Transcribed Perso-Arabic String
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│              STAGE 6: CERTIFIED NORMALIZER V3                           │
│  - Unicode NFC canonical decomposition and recomposition               │
│  - Kashmiri character mapping (Yeh, Kaf, Heh, Waw unification)         │
│  - Diacritic policy: preserve vowel carriers, strip non-phonemic signs  │
│  - Zero-Width Non-Joiner (ZWNJ) preservation & punctuation cleaning    │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
                     [Final Certified Kashmiri Transcript]
```

---

## 3. Receptive Field and Downsampling Arithmetic

A critical point of mathematical precision in speech recognition is calculating the exact relationship between the input raw audio sample count $N$ and the resulting sequence of acoustic frames $T$.

In the HuggingFace `facebook/wav2vec2-xls-r-300m` architecture, the feature extractor consists of a temporal 1D convolutional network with 7 sequential layers. Each layer $\ell \in \{1, \dots, 7\}$ has a 1D convolution, layer normalization, and a GELU activation function.

### Layer Specifications:
| Layer $\ell$ | Kernel Size $K_\ell$ | Stride $S_\ell$ | Input Channels | Output Channels | Output Temporal Dimension $L_\ell$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Conv1** | 10 | 5 | 1 | 512 | $\lfloor (L_0 - 10) / 5 \rfloor + 1$ |
| **Conv2** | 3 | 2 | 512 | 512 | $\lfloor (L_1 - 3) / 2 \rfloor + 1$ |
| **Conv3** | 3 | 2 | 512 | 512 | $\lfloor (L_2 - 3) / 2 \rfloor + 1$ |
| **Conv4** | 3 | 2 | 512 | 512 | $\lfloor (L_3 - 3) / 2 \rfloor + 1$ |
| **Conv5** | 3 | 2 | 512 | 512 | $\lfloor (L_4 - 3) / 2 \rfloor + 1$ |
| **Conv6** | 2 | 2 | 512 | 512 | $\lfloor (L_5 - 2) / 2 \rfloor + 1$ |
| **Conv7** | 2 | 2 | 512 | 512 | $\lfloor (L_6 - 2) / 2 \rfloor + 1$ |

### Mathematical Proof of Cumulative Stride and Total Receptive Field:
The cumulative stride $S_{\text{total}}$ of a cascade of $L$ convolutional layers is the product of their individual strides:

$$S_{\text{total}} = \prod_{\ell=1}^7 S_\ell = 5 \times 2 \times 2 \times 2 \times 2 \times 2 \times 2 = 5 \times 2^6 = 5 \times 64 = 320 \text{ samples}$$

At a sampling rate of $f_s = 16,000 \text{ Hz}$, the stride between adjacent CTC frames corresponds to:

$$\Delta t = \frac{S_{\text{total}}}{f_s} = \frac{320}{16,000} \text{ s} = 0.020 \text{ s} = 20 \text{ ms}$$

This yields an exact frame rate of:

$$\text{Frame Rate} = \frac{1}{\Delta t} = \frac{16,000}{320} = 50 \text{ frames per second (Hz)}$$

The receptive field $R_\ell$ of layer $\ell$ is calculated using the standard backward recursion relation:

$$R_\ell = R_{\ell-1} + (K_\ell - 1) \cdot \prod_{j=1}^{\ell-1} S_j$$

Starting with $R_0 = 1$:
1. $R_1 = 1 + (10 - 1) \times 1 = 10$
2. $R_2 = 10 + (3 - 1) \times 5 = 10 + 10 = 20$
3. $R_3 = 20 + (3 - 1) \times 10 = 20 + 20 = 40$
4. $R_4 = 40 + (3 - 1) \times 20 = 40 + 40 = 80$
5. $R_5 = 80 + (3 - 1) \times 40 = 80 + 80 = 160$
6. $R_6 = 160 + (2 - 1) \times 80 = 160 + 80 = 240$
7. $R_7 = 240 + (2 - 1) \times 160 = 240 + 160 = 400 \text{ samples}$

$$\text{Total Temporal Receptive Field} = 400 \text{ samples} = \frac{400}{16,000} \text{ s} = 25 \text{ ms}$$

Thus, each acoustic frame covers **25 milliseconds of context** and shifts forward by **20 milliseconds**, resulting in a **5 ms temporal overlap** between consecutive frames.

### Why Exact CTC Slicing Matters:
When a batch of audio files is padded to the maximum length in the batch, standard PyTorch forward passes produce padded frames. If these padded frames are passed to a KenLM beam search decoder, the language model can hallucinate words based on small numerical noise in the padded region. In [`src/decoding/ctc_slice.py`](file:///home/musa/Kashmiri%20Project/src/decoding/ctc_slice.py), the function `slice_exact_logits` computes the exact valid length:

$$T_{\text{valid}} = \text{compute\_exact\_ctc\_length}(N) = \left\lfloor \frac{N - 400}{320} \right\rfloor + 1$$

and slices the logit tensor strictly before decoding, eliminating padding-induced hallucinations.

---

## 4. The Vocabulary and Tokenizer Architecture

In speech recognition for Perso-Arabic scripts, the choice of modeling unit represents a critical architectural decision.

### Subword (BPE) vs. Character-Level Modeling

```
Text: "کٲشُر زَبانہِ مَنٛز" (In the Kashmiri language)

Subword BPE Fragmentation:
["ک", "ٲش", "##ُر", " ز", "##َب", "##انہ", "##ِ", " م", "##َن", "##ٛز"]
Risk: Diacritics split from base consonants; unseen inflections trigger OOV byte splits.

Character-Level Modeling (|V| = 47):
['ک', 'ٲ', 'ش', 'ُ', 'ر', '|', 'ز', 'َ', 'ب', 'ا', 'ن', 'ہ', 'ِ', '|', 'م', 'َ', 'ن', 'ٛ', 'ز']
Guarantee: Zero Out-Of-Vocabulary; perfect 1-to-1 mapping with Perso-Arabic cursive orthography.
```

### The 47-Token Kashmiri Character Inventory

The EduCasheer project standardized on a character-level vocabulary of $|V|=47$ distinct tokens, documented in `configs/vocab.json`. This inventory encompasses:
1. **Special Control Tokens**:
   - `[PAD]` (Index 0): Padding token for batch alignment.
   - `[UNK]` (Index 1): Unknown character fallback.
   - `|` (Index 2): Word boundary token (replaces standard ASCII space U+0020).
2. **Standard Perso-Arabic Consonants**:
   - `ا` (Alef), `ب` (Bay), `پ` (Pay), `ت` (Tay), `ٹ` (Ttay), `ث` (Say), `ج` (Jeem), `چ` (Tche), `ح` (Hah), `خ` (Khah), `د` (Dal), `ڈ` (Ddal), `ذ` (Zal), `ر` (Ray), `ڑ` (Rray), `ز` (Zay), `ژ` (Tsay), `س` (Seen), `ش` (Sheen), `ص` (Suad), `ض` (Zuad), `ط` (Toay), `ظ` (Zoay), `ع` (Ain), `غ` (Ghain), `ف` (Fay), `ق` (Qaf), `ک` (Kaf / Keheh), `گ` (Gaf), `ل` (Lam), `م` (Meem), `ن` (Noon), `و` (Waw), `ہ` (Heh Goal), `ی` (Yeh), `ے` (Yeh Barree).
3. **Kashmiri-Specific Phonemic Central Vowel Carriers & Diacritics**:
   - `ٲ` (Alef with wavy Hamza above /əː/, U+0672): High-central long vowel.
   - `ٳ` (Alef with wavy Hamza below /ɨː/, U+0673): High-back unrounded vowel.
   - `ۄ` (Waw with ring /ɔ/, U+06C4): Low-back open-mid rounded vowel.
   - `ۆ` (Waw with inverted V /oː/, U+06C6): Close-mid back rounded vowel.
   - `ؠ` (Kashmiri Yeh / Palatalization marker, U+0660): Phonemic palatalization marker following consonants.
   - `ں` (Noon Ghunna / Nasalization marker, U+06BA): Word-final and medial vowel nasalization.

By constraining $|V|=47$, the linear CTC projection layer requires only:

$$\text{Parameters} = 1024 \times 47 + 47 = 48,175 \text{ weights}$$

This minimal parameter overhead avoids overfitting on scarce Kashmiri speech data and ensures fast logit computation.

---

## 5. Decoding Mechanics: The Shallow Fusion Triad

At inference time, the acoustic model outputs a sequence of unnormalized logit vectors $\mathbf{Z} = (\mathbf{z}_1, \dots, \mathbf{z}_T) \in \mathbb{R}^{T \times 47}$. Decoding requires mapping this acoustic evidence to the most probable word sequence $\hat{\mathbf{w}} = (w_1, \dots, w_K)$.

### 5.1 The Log-Linear Scoring Objective

In the EduCasheer production pipeline, decoding is executed via **KenLM Prefix Beam Search with Shallow Fusion** ([`src/decoding/kenlm_decoder.py`](file:///home/musa/Kashmiri%20Project/src/decoding/kenlm_decoder.py)). The total score of a candidate word sequence $\mathbf{w}$ is formulated as:

$$\mathcal{S}(\mathbf{w} \mid \mathbf{X}) = \log P_{\text{CTC}}(\mathbf{w} \mid \mathbf{X}) + \alpha \log P_{\text{LM}}(\mathbf{w}) + \beta |\mathbf{w}|$$

Where:
- $\log P_{\text{CTC}}(\mathbf{w} \mid \mathbf{X})$: The acoustic sequence log-likelihood obtained by summing over all valid CTC alignments in $\mathcal{B}^{-1}(\mathbf{w})$.
- $\log P_{\text{LM}}(\mathbf{w})$: The linguistic log-probability assigned by an external 5-gram KenLM language model trained on validated Kashmiri text corpora:

$$\log P_{\text{LM}}(\mathbf{w}) = \sum_{k=1}^K \log P(w_k \mid w_{k-4}, \dots, w_{k-1})$$

- $|\mathbf{w}|$: The number of words in the candidate transcription.
- $\alpha \ge 0$: **Language Model Weight**, balancing acoustic evidence against linguistic prior.
- $\beta \ge 0$: **Word Insertion Bonus** (Brevity Penalty Compensation), preventing the decoder from favoring abnormally short sentences.
- $T_{\text{calib}}$: **Acoustic Temperature**, scaling logits prior to probability computation: $\tilde{\mathbf{z}}_t = \mathbf{z}_t / T_{\text{calib}}$.

### 5.2 The Crucial Role of Acoustic Temperature ($T_{\text{calib}} = 1.08$)

Standard CTC models suffer from **overconfidence**: due to the cross-entropy objective and frame-independence modeling, the softmax distribution tends toward extreme, peaky, Dirac-like probability distributions where the top token has probability $P \approx 0.999$ and competing candidates have probability $P \approx 10^{-6}$.

When acoustic probabilities are so peaked, the language model score $\alpha \log P_{\text{LM}}(\mathbf{w})$ is overwhelmed by the acoustic score during prefix beam search, rendering the LM unable to guide the beam away from acoustic errors.

By scaling logits by temperature $T = 1.08 > 1.0$:

$$P(\pi_t = k \mid \mathbf{X}) = \frac{\exp(z_{t,k} / T)}{\sum_{j=1}^{|V|} \exp(z_{t,j} / T)}$$

the distribution is smoothly softened. This increases the entropy of the acoustic posterior, allowing the KenLM language model to effectively rerank phonetically ambiguous hypotheses.

### 5.3 The Brevity Bias and the Word Insertion Bonus $\beta$

Because language models compute probabilities via the chain rule:

$$P(\mathbf{w}) = \prod_{k=1}^K P(w_k \mid w_{\lt k}) \implies \log P(\mathbf{w}) = \sum_{k=1}^K \log P(w_k \mid w_{\lt k})$$

and each conditional probability is strictly bounded by $P(w_k \mid w_{\lt k}) \le 1$ (meaning $\log P(w_k \mid w_{\lt k}) \le 0$), **every additional word adds a negative quantity to the sequence score**. Consequently, an unweighted beam search exhibits an inherent mathematical bias toward generating fewer words (brevity bias), frequently dropping suffixes, postpositions, and monosyllabic auxiliary verbs.

The word insertion bonus $\beta |\mathbf{w}|$ adds a constant positive reward $+\beta$ for every completed word token, counteracting the negative log-probability penalty and preventing catastrophic sentence truncation.

---

## 6. Disjoint Evaluation Discipline: The Speaker Leakage Trap

A foundational research insight established in Stage 04 and Stage 16 of the EduCasheer project is that **evaluation methodology dictates pipeline validity**.

### The Speaker Leakage Phenomenon

When an audio dataset is partitioned into training and test sets using naive random k-fold splitting (e.g., `train_test_split(test_size=0.15)`), audio recordings from the same human speaker inevitably appear in both the training set and the evaluation set.

```
RANDOM SPLITTING (FATALLY FLAWED):
Speaker 01 (Radio Host A) ──► 85% Train ──► Model learns vocal tract, F0, microphone EQ
                          ──► 15% Test  ──► Artificially inflated test accuracy (16.8% WER)

DISJOINT BIPARTITE GRAPH SPLITTING (METHODOLOGICALLY RIGOROUS):
Speaker 01 ─────────────────► 100% Train only
Speaker 02 ─────────────────► 100% Train only
Speaker 03 (Unseen) ────────► 100% Test only ──► True acoustic generalization (34.2% WER)
```

In Stage 04, empirical testing demonstrated that:
- On randomly split test sets, the model achieved a misleadingly low **16.8% WER**.
- When evaluated on a strictly **disjoint speaker test set** (where test speakers were completely quarantined from training), the true WER was **34.2%**—an absolute difference of **17.4% WER**!

The model was not learning generalizable acoustic-phonetic representations of Kashmiri speech; it was memorizing speaker-specific vocal tract lengths, fundamental frequencies ($F_0$), idiosyncratic microphone responses, and room acoustics.

### The Bipartite Graph Partitioning Solution

To eliminate speaker and source leakage, EduCasheer established a strict disjoint evaluation discipline:
1. **Bipartite Speaker-Audio Graph**: Utterances are modeled as vertices in a bipartite graph connecting audio IDs to speaker IDs.
2. **Connected Component Partitioning**: Splits are constructed by assigning entire connected components to either train, validation, or test.
3. **Source Quarantining**: Specific vulnerable sources (such as Bolbosh 22.05 kHz radio archives and religious sermons) are segmented into distinct evaluation domains (e.g., Clean Kashmiri, WAAZ Spontaneous, Quranic Kashmiri) to certify out-of-domain acoustic robustness.

---

## 7. Summary of Core Architectural Principles

1. **Acoustic Front-End**: Always resample using polyphase Kaiser sinc interpolation to 16,000 Hz, enforce mono mixing, and normalize peak amplitude to $[-1.0, 1.0]$.
2. **Backbone & CTC**: Leverage self-supervised pretraining (XLS-R 300M) to overcome the conditional independence assumption of CTC. Maintain alignment monotonicity to eliminate hallucination loops.
3. **Receptive Field Math**: Cumulative stride is $320$ samples ($20$ ms / $50$ Hz); total receptive field is $400$ samples ($25$ ms); valid frame count is $T = \left\lfloor \frac{N - 400}{320} \right\rfloor + 1$.
4. **Vocabulary Discipline**: Use character-level tokens ($|V|=47$) to avoid subword BPE morphological fragmentation in Perso-Arabic cursive script.
5. **Shallow Fusion Triad**: Calibrate acoustic temperature ($T = 1.08$), language model weight ($\alpha = 0.50$), and word insertion bonus ($\beta = 1.50$) to counteract brevity bias and soften overconfident posteriors.
6. **Evaluation Rigor**: Enforce strict zero-speaker-overlap disjoint evaluation to measure authentic generalized acoustic competence rather than speaker memorization.
"""

# ---------------------------------------------------------------------------
# 3. instructor_script.md
# ---------------------------------------------------------------------------
INSTRUCTOR_SCRIPT_CONTENT = r"""# Instructor Delivery Script: Lecture 02 (The Anatomy of Modern ASR Pipelines)

## Pedagogical Goals and Class Rhythm

This lecture pulls back the curtain on the engineering and mathematical architecture of modern speech recognition. Students often conceptualize neural networks as black boxes. Your goal today is to demystify every single component of the speech stack—from the continuous acoustic waveform entering the microphone to the normalized Unicode text string appearing in the user interface.

The lecture is structured across **90 minutes**, balancing conceptual paradigm trade-offs, rigorous board mathematics, live terminal code inspection, and student misconception interventions.

---

## Master Timetable & Class Phasing (90 Minutes)

```
00:00 - 00:15 | Segment 1: The Death of the Pronunciation Lexicon & The Hybrid Breakdown
00:15 - 00:35 | Segment 2: The Modern Neural Paradigm Face-Off: CTC vs. AED vs. RNN-T
00:35 - 00:55 | Segment 3: Deep Dive: The 6-Stage EduCasheer Production Pipeline
00:55 - 01:15 | Segment 4: Live Code & Architectural Trace: src/asr/ and configs/vocab.json
01:15 - 01:25 | Segment 5: The Decoding Triad: Temperature, KenLM, and the Brevity Penalty
01:25 - 01:30 | Segment 6: Summary, Lab Assignment, and Q&A
```

---

## Minute-by-Minute Instructor Delivery Guide

### 00:00 - 00:15 | Segment 1: The Death of the Pronunciation Lexicon

- **Instructor Action**: Enter the lecture hall. Walk directly to the center board. Draw three large rectangular boxes labeled: `[Acoustic Model P(X|s)]`, `[Pronunciation Lexicon L]`, and `[Language Model P(W)]`. Connect them with arrows meeting at `[WFST Decoder: H ∘ C ∘ L ∘ G]`.
- **Speaking Script**:
  > "Good morning, everyone. In Lecture 01, we examined the foundational physics of speech and witnessed the failure of the Stage 01 Urdu Kaldi baseline on Kashmiri, where WER skyrocketed to 88.40%.
  > Today, we answer the question: *Why did that classical system fail, and how is a modern End-to-End ASR pipeline physically constructed?*
  > Look at the board. If you open any speech textbook published between 1990 and 2015, this is what you see: the classical hybrid architecture. It requires three separate components. The acoustic model maps audio frames to phoneme states. The language model predicts word sequences. But look at the middle box: the **Pronunciation Lexicon**, denoted by $L$.
  > What is $L$? It is a dictionary where every single word in the language is phonetically transcribed by a team of linguists. In English, you have CMUdict—134,000 words.
  > Now, imagine you are tasked with building ASR for Kashmiri.
  > Where is the Kashmiri phonetic dictionary? It does not exist. Kashmiri has 14 phonemic vowels, including central vowels and contrastive vowel length, none of which are reliably written in standard Perso-Arabic text.
  > In Stage 01 of our research project, when Kaldi attempted forced alignment using an Urdu lexicon, it threw an alignment error on **38.2% of all audio files**. The system could not align the sounds to the words because the words had no phonetic entries!
  > If your system requires a human-engineered phonetic dictionary, your project is stillborn before you write a single line of neural network code.
  > Today, we explore how End-to-End deep learning eradicated the pronunciation lexicon."

---

### 00:15 - 00:35 | Segment 2: Paradigm Face-Off: CTC vs. AED vs. RNN-T

- **Instructor Action**: Erase the board. Divide the blackboard into three distinct vertical columns:
  - **Left Panel**: `CTC (Graves 2006)`
  - **Center Panel**: `AED / Whisper (Vaswani 2017)`
  - **Right Panel**: `RNN-T (Graves 2012)`

```
┌──────────────────────────────┬──────────────────────────────┬──────────────────────────────┐
│ PANEL 1: CTC (E2E MONOTONIC) │ PANEL 2: AED (AUTOREGRESSIVE)│ PANEL 3: RNN-TRANSDUCER      │
│ - Frame independence: P(π|X) │ - Cross-attention: P(y|y<u,X)│ - 2D Time-Label Lattice      │
│ - Non-emitting blank ε       │ - Autoregressive decoder     │ - Transcription + Prediction │
│ - Monotonic collapse B(π)=y  │ - Prone to hallucination     │ - Streaming capable          │
│ - Linear time: O(T)          │ - Quadratic memory: O(T^2)   │ - Memory-heavy lattice       │
│ - ZERO hallucination loops   │ - Subword byte-fragmentation │ - Complex joint network      │
└──────────────────────────────┴──────────────────────────────┴──────────────────────────────┘
```

- **Speaking Script**:
  > "In modern deep learning, we have three major End-to-End paradigms.
  > Let's look at the center panel: **Attention-based Encoder-Decoder**, made famous by models like OpenAI's Whisper. Why didn't we just deploy Whisper for Kashmiri?
  > In AED models, the decoder is autoregressive: $P(y_u \mid y_{\lt u}, \mathbf{X})$. The decoder generates text by looking at its own past predictions.
  > What happens when a speaker pauses for 3 seconds? Or when there is loud static on a radio broadcast?
  > The cross-attention mechanism searches for acoustic cues, finds none, and falls back entirely on its language model. It begins attending to its own previous words, entering an unrecoverable **hallucination loop**: *'and he said, and he said, and he said...'* repeating until the maximum sequence length is reached!
  > Furthermore, Whisper uses byte-level BPE. In Kashmiri Perso-Arabic, BPE shreds morphological inflections into disjoint byte fragments.
  > Now look at the left panel: **CTC (Connectionist Temporal Classification)**.
  > CTC enforces **strict alignment monotonicity**. The time index $t$ marches forward from $t=1$ to $t=T$. It cannot loop back. When the speaker is silent, what does CTC do? It emits the blank token $\epsilon$. It physically cannot hallucinate words that have no acoustic support!
  > And what about the old criticism that CTC assumes conditional independence between output frames?
  > That was true in 2006 when CTC was run on shallow RNNs. But in EduCasheer, CTC sits atop a 24-layer self-supervised **XLS-R Transformer** with 300 million parameters. Each frame representation $\mathbf{h}_t$ has already attended to the entire utterance! The contextual representation solves the independence problem, while CTC gives us monotonic decoding and zero hallucination risk."

---

### 00:35 - 00:55 | Segment 3: The 6-Stage EduCasheer Production Stack

- **Instructor Action**: Draw the full 6-stage pipeline flowchart on the board (matching the architecture diagram in `lecture_notes.md`).
- **Speaking Script**:
  > "Let's trace the physical lifecycle of an audio recording through the EduCasheer production pipeline.
  > **Stage 1**: An audio file arrives. Could be 44.1 kHz, stereo MP3 from YouTube, or 22.05 kHz WAV from Radio Kashmir. We mix channels to mono, perform Kaiser-windowed polyphase sinc resampling to exactly 16,000 Hz, and normalize the waveform amplitude to the range $[-1.0, 1.0]$.
  > **Stage 2**: The 16 kHz waveform enters the temporal Conv1d feature extractor. Seven convolutional layers downsample the audio by a factor of 320.
  > Let's do the arithmetic together. 16,000 samples per second divided by 320 samples per frame equals... **50 frames per second**! Each frame represents a 20 ms stride. The receptive field is 400 samples, or 25 ms.
  > These 50 Hz frames pass through 24 Transformer encoder layers with 1024-dimensional hidden representations.
  > **Stage 3**: A linear projection head maps 1024 dimensions down to 47 vocabulary logits.
  > **Stage 4**: Notice this critical detail—**Temperature Calibration & Logit Slicing**. If the audio was padded in a batch, we calculate the exact valid mathematical frame count $T = \left\lfloor \frac{N - 400}{320} \right\rfloor + 1$ and slice off the padding. Then we divide logits by an acoustic temperature $T = 1.08$. We'll see why in a few minutes.
  > **Stage 5**: We feed these scaled logits into a prefix beam search decoder shallow-fused with a 5-gram KenLM language model.
  > **Stage 6**: Finally, the raw decoded string passes through **Certified Normalizer v3**, which unifies Perso-Arabic homoglyphs, cleans up diacritics, and handles Zero-Width Non-Joiners."

---

### 00:55 - 01:15 | Segment 4: Live Code Inspection & Vocabulary Analysis

- **Instructor Action**: Switch display to terminal. Open [`src/asr/transcribe.py`](file:///home/musa/Kashmiri%20Project/src/asr/transcribe.py) and [`src/decoding/ctc_slice.py`](file:///home/musa/Kashmiri%20Project/src/decoding/ctc_slice.py).
- **Terminal Execution Cue**:
  ```bash
  cat src/decoding/ctc_slice.py | head -n 35
  ```
- **Speaking Script**:
  > "Look at lines 12–16 of `ctc_slice.py`:

```python
WAV2VEC2_CONV_KERNEL = [10, 3, 3, 3, 3, 2, 2]
WAV2VEC2_CONV_STRIDE = [5, 2, 2, 2, 2, 2, 2]
WAV2VEC2_RECEPTIVE_FIELD = 400
WAV2VEC2_TOTAL_STRIDE = 320
```

  > This is not an approximation. This is the exact physical geometry of the convolutional frontend.
  > Now let's look at `configs/vocab.json`. Why do we have only 47 tokens?
  > If we used subword BPE with 4,000 merge rules, what happens when a speaker says 'کٲشُر' with a slightly prolonged central vowel? BPE splits the word into arbitrary byte tokens, creating out-of-vocabulary artifacts.
  > By modeling at the character level ($|V|=47$), any valid Kashmiri word can be transcribed. The output projection matrix has only $1024 \times 47 = 48,128$ weights. It is lightweight, fast, and mathematically impossible to experience Out-Of-Vocabulary errors."

---

### 01:15 - 01:25 | Segment 5: The Decoding Triad & Brevity Penalty

- **Instructor Action**: Write the shallow fusion decoding equation on the blackboard:

$$\mathcal{S}(\mathbf{w}) = \log P_{\text{CTC}}(\mathbf{w} \mid \mathbf{X}) + \alpha \log P_{\text{LM}}(\mathbf{w}) + \beta |\mathbf{w}|$$

- **Speaking Script**:
  > "Look at this equation. This is the **Shallow Fusion Triad**.
  > There are three parameters here: $\alpha$, $\beta$, and temperature $T$.
  > Why do we need $\beta$, the word insertion bonus?
  > Think about what $\log P_{\text{LM}}(\mathbf{w})$ is doing. Because $P(w_k \mid w_{\lt k}) \le 1$, its logarithm is **always negative**.
  > If a candidate sentence has 10 words, you are adding 10 negative numbers. If a candidate has 2 words, you are only adding 2 negative numbers!
  > Left to itself, the decoder has a catastrophic **brevity bias**: it prefers deleting words to avoid accumulating negative language model penalties!
  > The parameter $\beta$ rewards the hypothesis with a positive constant for every word emitted. It acts as an acoustic fuel injection that balances sentence length."

---

### 01:25 - 01:30 | Segment 6: Summary & Student Misconception Interventions

- **Key Misconceptions to Address**:
  1. *Misconception*: "Higher beam width $W$ always improves WER."
     * *Correction*: In shallow fusion, excessively wide beams ($W > 200$) allow the language model to dominate acoustic evidence, introducing grammatical substitutions of proper nouns and village names. $W = 100$ is the empirically proven sweet spot.
  2. *Misconception*: "Random 80/20 train/test split is completely fine."
     * *Correction*: Speech is not tabular data! Random splitting causes speaker leakage, underestimating real-world WER by up to 18% absolute. Always enforce zero-speaker-overlap graph partitioning.
- **Concluding Script**:
  > "In today's lab, you will implement the 7-layer convolutional downsampling logic, build a pure Python prefix beam search decoder, and test the impact of temperature scaling.
  > See you in the lab!"
"""

# ---------------------------------------------------------------------------
# 4. mathematical_foundations.md
# ---------------------------------------------------------------------------
MATH_FOUNDATIONS_CONTENT = r"""# Mathematical Foundations: Pipeline Anatomy and Decoding Formulation

---

## 1. Convolutional Receptive Field and Stride Arithmetic

The acoustic feature extractor of the `facebook/wav2vec2-xls-r-300m` model consists of a 7-layer temporal 1D convolutional network operating on raw audio waveforms sampled at $f_s = 16,000 \text{ Hz}$.

### 1.1 Temporal Convolution Dimension Equation
For a 1D convolution with input sequence length $L_{\text{in}}$, kernel size $K$, stride $S$, and zero padding $P=0$, the output sequence length $L_{\text{out}}$ is given by:

$$L_{\text{out}} = \left\lfloor \frac{L_{\text{in}} - K}{S} \right\rfloor + 1$$

### 1.2 Cumulative Stride Derivation
Let $S_\ell$ denote the stride of layer $\ell \in \{1, \dots, 7\}$. The cumulative temporal downsampling factor $S_{\text{total}}$ of the 7-layer cascade is:

$$S_{\text{total}} = \prod_{\ell=1}^7 S_\ell = 5 \times 2 \times 2 \times 2 \times 2 \times 2 \times 2 = 5 \times 2^6 = 320 \text{ samples}$$

Given $f_s = 16,000 \text{ Hz}$, the frame period $\Delta t$ between consecutive output frames is:

$$\Delta t = \frac{S_{\text{total}}}{f_s} = \frac{320}{16,000} \text{ s} = 0.020 \text{ s} = 20 \text{ ms}$$

The corresponding frame emission frequency is:

$$f_{\text{frame}} = \frac{1}{\Delta t} = 50 \text{ Hz}$$

### 1.3 Total Receptive Field Recursion
The receptive field $R_\ell$ defines the number of input audio samples in the raw waveform that influence a single output activation at layer $\ell$. It satisfies the recursive formula:

$$R_\ell = R_{\ell-1} + (K_\ell - 1) \cdot \prod_{j=1}^{\ell-1} S_j \quad \text{with } R_0 = 1$$

Evaluating step-by-step:
1. Layer 1 ($K_1=10, S_1=5$):

$$R_1 = 1 + (10 - 1) \times 1 = 10 \text{ samples}$$

2. Layer 2 ($K_2=3, S_2=2$):

$$R_2 = 10 + (3 - 1) \times 5 = 10 + 10 = 20 \text{ samples}$$

3. Layer 3 ($K_3=3, S_3=2$):

$$R_3 = 20 + (3 - 1) \times 10 = 20 + 20 = 40 \text{ samples}$$

4. Layer 4 ($K_4=3, S_4=2$):

$$R_4 = 40 + (3 - 1) \times 20 = 40 + 40 = 80 \text{ samples}$$

5. Layer 5 ($K_5=3, S_5=2$):

$$R_5 = 80 + (3 - 1) \times 40 = 80 + 80 = 160 \text{ samples}$$

6. Layer 6 ($K_6=2, S_6=2$):

$$R_6 = 160 + (2 - 1) \times 80 = 160 + 80 = 240 \text{ samples}$$

7. Layer 7 ($K_7=2, S_7=2$):

$$R_7 = 240 + (2 - 1) \times 160 = 240 + 160 = 400 \text{ samples}$$

$$\text{Total Receptive Field} = 400 \text{ samples} = \frac{400}{16,000} \text{ s} = 25 \text{ ms}$$

### 1.4 Exact Valid CTC Length
An input audio buffer with $N$ samples produces exactly $T_{\text{valid}}$ frames if and only if $N \ge 400$:

$$T_{\text{valid}} = \left\lfloor \frac{N - 400}{320} \right\rfloor + 1$$

---

## 2. The CTC Collapse Operator $\mathcal{B}$ and Alignment Combinatorics

Let $\Omega$ be the alphabet of 46 non-blank characters, and define $\Omega' = \Omega \cup \{\epsilon\}$. An alignment $\boldsymbol{\pi} = (\pi_1, \dots, \pi_T) \in (\Omega')^T$ maps to sequence $\mathbf{y} = (y_1, \dots, y_U) \in \Omega^U$ via operator $\mathcal{B}$.

### 2.1 Deterministic Collapse Definition
The operator $\mathcal{B}$ executes two sequential transformations:
1. **Consecutive Duplicate Deduplication**:

$$\mathcal{B}_{\text{dedup}}(\pi_1, \dots, \pi_T) = (\pi_{t_1}, \pi_{t_2}, \dots) \quad \text{where } \pi_{t_k} \neq \pi_{t_k-1}$$

2. **Blank Removal**:

$$\mathcal{B}(\boldsymbol{\pi}) = [c \in \mathcal{B}_{\text{dedup}}(\boldsymbol{\pi}) \mid c \neq \epsilon]$$

### 2.2 Cardinality of the CTC Preimage $|\mathcal{B}^{-1}(\mathbf{y})|$
Consider a target sequence $\mathbf{y} = (y_1, \dots, y_U)$ of length $U$ decoded across $T$ frames ($T \ge U$).

#### Case A: No Consecutive Identical Tokens ($y_u \neq y_{u+1} \forall u$)
When no adjacent characters are identical, blank tokens are optional. The number of valid alignments corresponds to distributing $T$ frames across the $U$ labels and possible intermediate blanks:

$$|\mathcal{B}^{-1}(\mathbf{y})| = \sum_{k=0}^{T-U} \binom{T}{k} \binom{T-k-1}{U-1}$$

#### Case B: Consecutive Identical Tokens ($y_u = y_{u+1}$)
If $y_u = y_{u+1}$, the alignment **must** contain at least one blank token $\epsilon$ between them:

$$\pi_{t_a} = y_u, \quad \pi_{t_b} = \epsilon, \quad \pi_{t_c} = y_{u+1} \quad (t_a < t_b < t_c)$$

Without this intermediate blank, the collapse operator would merge the identical characters: $\mathcal{B}(a, a) = a \neq (a, a)$. This constraint strictly reduces the cardinality of $\mathcal{B}^{-1}(\mathbf{y})$.

---

## 3. Acoustic Temperature Scaling and Posterior Entropy

Let $\mathbf{z}_t \in \mathbb{R}^{|V|}$ denote the raw acoustic logit vector at frame $t$. Under temperature scaling with $T_{\text{calib}} > 0$:

$$P(\pi_t = k \mid \mathbf{X}; T_{\text{calib}}) = y_k(T_{\text{calib}}) = \frac{\exp(z_{t,k} / T_{\text{calib}})}{\sum_{j=1}^{|V|} \exp(z_{t,j} / T_{\text{calib}})}$$

### 3.1 First Derivative with Respect to Temperature
To determine how temperature alters token probabilities, we differentiate $y_k$ with respect to $T_{\text{calib}}$:

$$\frac{\partial y_k}{\partial T_{\text{calib}}} = \frac{\partial}{\partial T_{\text{calib}}} \left[ \frac{\exp(z_k / T_{\text{calib}})}{\sum_j \exp(z_j / T_{\text{calib}})} \right]$$

Using the quotient rule:

$$\frac{\partial y_k}{\partial T_{\text{calib}}} = \frac{-\frac{z_k}{T_{\text{calib}}^2} \exp(z_k / T_{\text{calib}}) \sum_j \exp(z_j / T_{\text{calib}}) - \exp(z_k / T_{\text{calib}}) \sum_j \left( -\frac{z_j}{T_{\text{calib}}^2} \exp(z_j / T_{\text{calib}}) \right)}{\left( \sum_j \exp(z_j / T_{\text{calib}}) \right)^2}$$

Factoring out $y_k$ and $\frac{1}{T_{\text{calib}}^2}$:

$$\frac{\partial y_k}{\partial T_{\text{calib}}} = -\frac{y_k}{T_{\text{calib}}^2} \left[ z_k - \sum_{j=1}^{|V|} y_j z_j \right] = -\frac{y_k}{T_{\text{calib}}^2} (z_k - \bar{z})$$

where $\bar{z} = \mathbb{E}_{y}[z]$ is the expected logit value under the current distribution.

### 3.2 Implication for Peaked Distributions:
- If token $k$ is the dominant argmax prediction, $z_k > \bar{z} \implies \frac{\partial y_k}{\partial T_{\text{calib}}} < 0$. Increasing temperature **decreases** overconfident peak probabilities.
- For non-dominant tokens where $z_j < \bar{z} \implies \frac{\partial y_j}{\partial T_{\text{calib}}} > 0$. Increasing temperature **boosts** low-probability competing candidates.
- The Shannon entropy $H(Y) = -\sum_k y_k \log y_k$ is a strictly increasing function of $T_{\text{calib}}$ for $T_{\text{calib}} \ge 1.0$.

---

## 4. Log-Linear Shallow Fusion and Brevity Bias Derivation

The decoding search objective finds the word sequence $\hat{\mathbf{w}}$ maximizing:

$$\mathcal{S}(\mathbf{w}) = \log P_{\text{CTC}}(\mathbf{w} \mid \mathbf{X}) + \alpha \sum_{k=1}^K \log P_{\text{LM}}(w_k \mid w_{\lt k}) + \beta K$$

### 4.1 Proof of Brevity Bias
Let $\mathbf{w}_A$ be a hypothesis of length $K_A$ and $\mathbf{w}_B$ be an extended hypothesis of length $K_B = K_A + M$ ($M \ge 1$).

Assume the acoustic model provides neutral evidence across the extension:

$$\log P_{\text{CTC}}(\mathbf{w}_B \mid \mathbf{X}) \approx \log P_{\text{CTC}}(\mathbf{w}_A \mid \mathbf{X})$$

Evaluating the score difference without length compensation ($\beta = 0$):

$$\Delta \mathcal{S} = \mathcal{S}(\mathbf{w}_B) - \mathcal{S}(\mathbf{w}_A) = \alpha \sum_{j=1}^M \log P_{\text{LM}}(w_{K_A + j} \mid w_{\lt K_A + j})$$

Because $0 < P_{\text{LM}}(\cdot) \le 1$, every log-probability term is non-positive:

$$\log P_{\text{LM}}(w \mid \cdot) \le 0 \implies \Delta \mathcal{S} \le 0$$

Without a compensatory term $\beta > 0$, **any hypothesis that emits more words is strictly penalized by the language model**, causing beam search to systematically truncate sentences. Setting:

$$\beta > \alpha \cdot \mathbb{E}\left[ -\log P_{\text{LM}}(w \mid w_{\text{context}}) \right]$$

counterbalances the average cross-entropy of the language model, restoring unbiased sequence length generation.
"""

# ---------------------------------------------------------------------------
# 5. code_walkthrough.md
# ---------------------------------------------------------------------------
CODE_WALKTHROUGH_CONTENT = r"""# Code Walkthrough: Pipeline Inspection and Execution Trace

---

## 1. Architectural Architecture & Source File Index

This walkthrough traces the execution path of the production Kashmiri ASR pipeline across four primary codebase modules:
1. [`src/asr/transcribe.py`](file:///home/musa/Kashmiri%20Project/src/asr/transcribe.py): Unified top-level inference engine.
2. [`src/decoding/ctc_slice.py`](file:///home/musa/Kashmiri%20Project/src/decoding/ctc_slice.py): Mathematical CTC receptive field frame slicing.
3. [`src/decoding/kenlm_decoder.py`](file:///home/musa/Kashmiri%20Project/src/decoding/kenlm_decoder.py): Calibrated KenLM beam search decoder.
4. [`src/normalization/normalizer.py`](file:///home/musa/Kashmiri%20Project/src/normalization/normalizer.py): Certified Perso-Arabic text normalizer.

---

## 2. Audio Preprocessing and Feature Slicing Trace

In [`src/asr/transcribe.py`](file:///home/musa/Kashmiri%20Project/src/asr/transcribe.py), audio is loaded and processed through the acoustic model:

```python
# From src/asr/transcribe.py (Lines 56-82)
def transcribe(
    self,
    audio_path: Union[str, Path],
    normalize: bool = False,
    use_kenlm: bool = False,
    temperature: Optional[float] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None
) -> Dict[str, Any]:
    # 1. Load audio and enforce 16 kHz mono float32
    audio = load_audio(audio_path, target_sr=16000)
    audio = normalize_audio(audio)
    num_samples = len(audio)

    # 2. Check minimum sample length against receptive field (400 samples = 25 ms)
    if num_samples < 400:
        return {"text": "", "tokens": [], "confidence": 0.0, "raw_text": ""}

    # 3. Model forward pass
    inputs = self.processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=False
    ).input_values.to(self.device)

    with torch.no_grad():
        logits = self.model(inputs).logits[0]  # Shape: [T_full, |V|]

    # 4. Strict receptive field slicing to eliminate padding artifacts
    logits = slice_exact_logits(logits, num_samples)  # Shape: [T_valid, 47]
```

### Key Technical Details:
- **`num_samples < 400` Guard**: If an audio sample is shorter than the 400-sample receptive field of the Conv1d feature extractor, the model cannot perform even a single convolution without error.
- **`slice_exact_logits`**: Inspect [`src/decoding/ctc_slice.py`](file:///home/musa/Kashmiri%20Project/src/decoding/ctc_slice.py):
  ```python
  def compute_exact_ctc_length(input_samples: int) -> int:
      if input_samples < 400:
          return 0
      l = input_samples
      for k, s in zip([10, 3, 3, 3, 3, 2, 2], [5, 2, 2, 2, 2, 2, 2]):
          l = (l - k) // s + 1
      return int(l)
  ```
  This iterates through the exact layer parameters, returning the precise number of valid output frames.

---

## 3. Calibrated KenLM Beam Search Decoding

Once exact logits are sliced, [`src/decoding/kenlm_decoder.py`](file:///home/musa/Kashmiri%20Project/src/decoding/kenlm_decoder.py) applies temperature scaling and shallow fusion:

```python
# From src/decoding/kenlm_decoder.py (Lines 62-84)
def decode(self, logits: Union[np.ndarray, torch.Tensor]) -> str:
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()

    if logits.ndim == 3:
        logits = logits[0]

    # Apply calibrated acoustic temperature scaling
    # T = 1.08 softens peaky CTC posteriors
    scaled_logits = logits / self.temperature

    # pyctcdecode executes prefix beam search with KenLM shallow fusion:
    # S(w) = log P_CTC + alpha * log P_LM + beta * |w|
    transcription = self.decoder.decode(
        scaled_logits,
        beam_width=self.beam_width,            # W = 100
        beam_prune_logp=self.beam_prune_logp,  # Prunes beams below -10.0
        token_min_logp=self.token_min_logp     # Prunes tokens below -5.0
    )
    return transcription.strip()
```

### Why PyCTCDecode Requires `logits / self.temperature`:
Notice that `scaled_logits = logits / self.temperature` occurs before passing the array to `self.decoder.decode()`. If $T = 1.0$, the argmax CTC token often carries $99.9\%$ probability. At $T = 1.08$, that probability reduces to $\sim 92\%$, granting competing phonetically close candidates enough score to survive beam pruning until the 5-gram language model can evaluate their syntactic plausibility.

---

## 4. Orthographic Normalization Execution

The final stage passes the raw decoded string through [`src/normalization/normalizer.py`](file:///home/musa/Kashmiri%20Project/src/normalization/normalizer.py):

```python
# From src/normalization/normalizer.py (Lines 54-85)
def normalize_kashmiri_text(
    text: str,
    strip_diacritics: bool = False,
    normalize_digits: bool = True,
    preserve_zwnj: bool = True
) -> str:
    if not text or not isinstance(text, str):
        return ""

    # 1. Unicode NFC canonical decomposition and recomposition
    text = unicodedata.normalize('NFC', text)

    # 2. Perso-Arabic homoglyph unification
    # Replaces Arabic Kaf (U+0643) -> Kashmiri Keheh (U+06A9)
    # Replaces Arabic Yeh (U+064A) -> Kashmiri Yeh (U+06CC)
    # Strips Tatweel/Kashida (U+0640)
    for src, tgt in CHAR_MAP.items():
        text = text.replace(src, tgt)

    # 3. Kashmiri Central Vowel Harmonization
    text = text.replace('\u0671', '\u0627')  # Alef Wasla -> Alef
    text = text.replace('\u0622', '\u0627')  # Alef Madda -> Alef (if in unvocalized context)

    # 4. Whitespace and ZWNJ cleanup
    text = re.sub(r'[ \t\f\v]+', ' ', text)
    return text.strip()
```

This ensures that the final evaluation metrics (WER and CER) reflect authentic recognition errors rather than trivial Unicode encoding differences.
"""

# ---------------------------------------------------------------------------
# 6. practical_lab.md
# ---------------------------------------------------------------------------
PRACTICAL_LAB_CONTENT = r"""# Practical Lab: Receptive Field Calculation, Temperature Scaling, and Prefix Decoding

## Objective
In this laboratory, students will implement and verify the core architectural components of modern CTC speech recognition:
1. Simulating the exact 7-layer Conv1d receptive field and valid frame slicing.
2. Implementing a pure Python Prefix Beam Search decoder with blank probability tracking.
3. Measuring the mathematical effect of acoustic temperature scaling on posterior distribution entropy.

---

## Exercise 1: Receptive Field & CTC Frame Slicing Verification

Run the following Python script to simulate audio lengths and verify exact CTC frame boundaries:

```python
#!/usr/bin/env python3
# Lab Exercise 1: Conv1d Feature Extractor Simulation

def compute_exact_ctc_length(input_samples: int) -> int:
    kernels = [10, 3, 3, 3, 3, 2, 2]
    strides = [5, 2, 2, 2, 2, 2, 2]
    
    if input_samples < 400:
        return 0
    
    length = input_samples
    for k, s in zip(kernels, strides):
        length = (length - k) // s + 1
    return int(length)

# Test across various audio durations at 16,000 Hz
test_durations_sec = [0.01, 0.024, 0.025, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]

print("Duration (s) | Num Samples | Valid CTC Frames | Effective Frame Rate (Hz)")
print("-" * 65)
for sec in test_durations_sec:
    samples = int(sec * 16000)
    frames = compute_exact_ctc_length(samples)
    rate = frames / sec if sec > 0 and frames > 0 else 0.0
    print(f"{sec:12.3f} | {samples:11d} | {frames:16d} | {rate:24.2f}")
```

### Lab Questions for Part 1:
1. Why does an audio sample of $0.024$ seconds ($384$ samples) produce exactly $0$ frames?
2. What is the minimum number of samples required to produce at least $2$ valid CTC frames?

---

## Exercise 2: Implementing a Pure Python Prefix Beam Search Decoder

The core algorithmic challenge in CTC decoding is that multiple distinct alignment paths map to the same label sequence. In prefix beam search, we track two distinct probabilities for each candidate prefix $\ell$:
- $P_b(\ell, t)$: Total probability of prefix $\ell$ at time $t$ given that the last emission was the **blank token** $\epsilon$.
- $P_{nb}(\ell, t)$: Total probability of prefix $\ell$ at time $t$ given that the last emission was a **non-blank character**.

The total probability of prefix $\ell$ is $P_{\text{tot}}(\ell, t) = P_b(\ell, t) + P_{nb}(\ell, t)$.

```python
#!/usr/bin/env python3
# Lab Exercise 2: Prefix Beam Search with Blank Tracking
import numpy as np

def prefix_beam_search(logits, vocab, beam_width=3, blank_idx=0):
    # [Docstring]
    # Pure Python CTC Prefix Beam Search Decoder
    # logits: [T, |V|] unnormalized emission scores
    # [Docstring]
    T, V = logits.shape
    # Compute frame-level probabilities via softmax
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    # Initialize beams with empty prefix ""
    # beam dict: prefix -> (P_blank, P_non_blank)
    beams = {"": (1.0, 0.0)}
    
    for t in range(T):
        new_beams = {}
        p_blank = probs[t, blank_idx]
        
        for prefix, (p_b, p_nb) in beams.items():
            p_tot = p_b + p_nb
            
            # Case 1: Emit blank token ε
            # Extends prefix without changing label string
            if prefix not in new_beams:
                new_beams[prefix] = [0.0, 0.0]
            new_beams[prefix][0] += p_tot * p_blank
            
            # Case 2: Emit character c in vocab
            for c_idx in range(V):
                if c_idx == blank_idx:
                    continue
                c = vocab[c_idx]
                p_c = probs[t, c_idx]
                new_prefix = prefix + c
                
                if new_prefix not in new_beams:
                    new_beams[new_prefix] = [0.0, 0.0]
                
                # If extending identical character as previous end:
                # Must differentiate whether blank intervened!
                if len(prefix) > 0 and c == prefix[-1]:
                    # If blank intervened, can extend identical char
                    new_beams[new_prefix][1] += p_b * p_c
                    # If no blank intervened, repeating char collapses to existing prefix!
                    new_beams[prefix][1] += p_nb * p_c
                else:
                    new_beams[new_prefix][1] += p_tot * p_c
        
        # Prune to beam width W based on total probability
        sorted_prefixes = sorted(
            new_beams.keys(),
            key=lambda p: new_beams[p][0] + new_beams[p][1],
            reverse=True
        )[:beam_width]
        
        beams = {p: (new_beams[p][0], new_beams[p][1]) for p in sorted_prefixes}
        print(f"Frame {t}: Top Beams: {[(p, round(beams[p][0]+beams[p][1], 4)) for p in beams]}")
    
    best_prefix = max(beams.keys(), key=lambda p: beams[p][0] + beams[p][1])
    return best_prefix

# Test run with toy vocabulary: 0=ε, 1='ك', 2='ٲ', 3='ش'
toy_vocab = {0: "<blank>", 1: "ك", 2: "ٲ", 3: "ش"}
toy_logits = np.array([
    [2.0, 0.1, 0.1, 0.1],  # t=0: blank dominant
    [0.1, 3.0, 0.1, 0.1],  # t=1: 'ك' dominant
    [1.5, 1.5, 0.1, 0.1],  # t=2: 'ك' vs blank contest
    [0.1, 0.1, 2.5, 0.1],  # t=3: 'ٲ' dominant
])

result = prefix_beam_search(toy_logits, toy_vocab, beam_width=3)
print(f"\nFinal Decoded Transcription: {result}")
```

---

## Exercise 3: Acoustic Temperature Scaling & Distribution Entropy

Run the following script to observe how temperature $T \in \{0.5, 1.0, 1.08, 2.0\}$ alters the Shannon entropy of acoustic emissions:

```python
#!/usr/bin/env python3
# Lab Exercise 3: Temperature Scaling & Shannon Entropy
import numpy as np

# Simulate a peaked logit vector from a trained CTC model
raw_logits = np.array([8.5, 2.1, 1.4, 0.2, -1.0])

temperatures = [0.5, 1.0, 1.08, 1.5, 2.0]

print("Temp (T) | Max Prob (Top-1) | Runner-up Prob | Entropy H(Y) in bits")
print("-" * 65)
for T in temperatures:
    scaled = raw_logits / T
    exp_s = np.exp(scaled - np.max(scaled))
    p = exp_s / np.sum(exp_s)
    
    top1 = np.max(p)
    top2 = np.sort(p)[-2]
    entropy = -np.sum(p * np.log2(p + 1e-12))
    
    print(f"{T:8.2f} | {top1:16.4f} | {top2:14.4f} | {entropy:20.4f}")
```
"""

# ---------------------------------------------------------------------------
# 7. exercises.md
# ---------------------------------------------------------------------------
EXERCISES_CONTENT = r"""# Exercises: Pipeline Anatomy and Decoding Formulation

## Problem 1: Conv1d Feature Extractor Receptive Field & Stride
Consider an acoustic model feature extractor consisting of the 7 temporal Conv1d layers specified in the lecture:
- Kernel sizes: $\mathbf{K} = [10, 3, 3, 3, 3, 2, 2]$
- Strides: $\mathbf{S} = [5, 2, 2, 2, 2, 2, 2]$
- Padding: $P = 0$ for all layers
- Audio sampling rate: $f_s = 16,000 \text{ Hz}$

1. Derive the mathematical formula for the receptive field $R_\ell$ at any layer $\ell \in \{1, \dots, 7\}$.
2. Calculate the exact input sample index range $[n_{\text{start}}, n_{\text{end}}]$ that affects the second output frame ($t=2$).
3. Suppose an audio file has exactly $32,150$ samples ($2.009375$ seconds). Calculate the exact number of valid CTC frames produced by the encoder.

---

## Problem 2: CTC Alignment Combinatorics
Let the target character alphabet be $\Omega = \{c_1, c_2, c_3\}$ (corresponding to Kashmiri characters "ك", "ٲ", "ش") and define the extended alphabet $\Omega' = \Omega \cup \{\epsilon\}$, where $\epsilon$ is the non-emitting blank symbol.
Consider an acoustic sequence of duration $T=4$ frames.

1. List all valid alignments $\boldsymbol{\pi} \in (\Omega')^4$ in the preimage $\mathcal{B}^{-1}(\mathbf{y})$ for the target sequence $\mathbf{y} = (c_1, c_2)$ (corresponding to "ك", "ٲ").
2. Now consider the target sequence $\mathbf{y}_{\text{dup}} = (c_1, c_1)$ (corresponding to "ك", "ك") with two consecutive identical tokens. List all valid alignments $\boldsymbol{\pi} \in (\Omega')^4$ in $\mathcal{B}^{-1}(\mathbf{y}_{\text{dup}})$.
3. Compare the cardinality $|\mathcal{B}^{-1}(\mathbf{y})|$ versus $|\mathcal{B}^{-1}(\mathbf{y}_{\text{dup}})|$. Explain mathematically why repeated tokens restrict alignment paths.

---

## Problem 3: Shallow Fusion Brevity Bias Calibration
A 5-gram KenLM language model has an average cross-entropy $\bar{H}_{\text{LM}} = 4.2 \text{ nats}$ per word on a Kashmiri test corpus.
A student runs prefix beam search with LM weight $\alpha = 0.50$ and word insertion bonus $\beta = 0.0$.

1. Explain why the student observes that the model transcribes long conversational utterances as only 2 or 3 words.
2. Calculate the minimum theoretical value of $\beta$ required to neutralize the average negative log-probability penalty introduced by the language model.
3. If $\alpha$ is doubled to $1.00$, what adjustment to $\beta$ is required to maintain sequence length balance?

---

## Problem 4: Speaker Leakage Graph Partitioning
An audio collection contains 100 audio files recorded by 10 speakers (10 files per speaker).
A naive researcher performs random 80/20 train/test splitting across the 100 files without considering speaker identity.

1. What is the probability that a specific speaker appears in *both* the training set (80 files) and the test set (20 files)?
2. If all 10 speakers have audio in both sets, explain how the model can achieve a low WER by memorizing speaker acoustic transfer functions rather than learning generalizable speech recognition.
3. Formulate the bipartite graph partitioning algorithm that guarantees $0.0\%$ speaker overlap between train and test.
"""

# ---------------------------------------------------------------------------
# 8. solutions.md
# ---------------------------------------------------------------------------
SOLUTIONS_CONTENT = r"""# Solutions to Exercises: Lecture 02

## Solution to Problem 1: Conv1d Receptive Field & Stride

### Part 1: Receptive Field Derivation
The backward recursion formula for the receptive field $R_\ell$ of layer $\ell$ with kernel $K_\ell$ and cumulative stride $S_{\ell-1}^{\text{cum}} = \prod_{j=1}^{\ell-1} S_j$ is:

$$R_\ell = R_{\ell-1} + (K_\ell - 1) \cdot S_{\ell-1}^{\text{cum}} \quad \text{with } R_0 = 1, S_0^{\text{cum}} = 1$$

Evaluating across all 7 layers:
- Layer 1: $R_1 = 1 + (10 - 1) \times 1 = 10$
- Layer 2: $R_2 = 10 + (3 - 1) \times 5 = 20$
- Layer 3: $R_3 = 20 + (3 - 1) \times 10 = 40$
- Layer 4: $R_4 = 40 + (3 - 1) \times 20 = 80$
- Layer 5: $R_5 = 80 + (3 - 1) \times 40 = 160$
- Layer 6: $R_6 = 160 + (2 - 1) \times 80 = 240$
- Layer 7: $R_7 = 240 + (2 - 1) \times 160 = 400 \text{ samples}$

### Part 2: Sample Index Range for Frame $t=2$
The first frame ($t=1$, 0-indexed $t=0$) spans input samples $[0, 399]$ (a total of 400 samples).
Because the cumulative stride across all 7 layers is $S_{\text{total}} = 320$, the second frame ($t=2$, 0-indexed $t=1$) is shifted forward by exactly 320 samples:

$$n_{\text{start}} = 1 \times 320 = 320$$
$$n_{\text{end}} = n_{\text{start}} + R_{\text{total}} - 1 = 320 + 400 - 1 = 719$$

The second frame spans sample indices **$[320, 719]$** (inclusive).

### Part 3: Number of Valid CTC Frames for $N = 32,150$ Samples
Using the exact frame formula:

$$T = \left\lfloor \frac{N - 400}{320} \right\rfloor + 1 = \left\lfloor \frac{32,150 - 400}{320} \right\rfloor + 1 = \left\lfloor \frac{31,750}{320} \right\rfloor + 1 = \lfloor 99.21875 \rfloor + 1 = 99 + 1 = 100 \text{ frames}$$

The encoder produces exactly **100 valid CTC frames**.

---

## Solution to Problem 2: CTC Alignment Combinatorics

### Part 1: Alignments for $\mathbf{y} = (c_1, c_2)$ at $T=4$
Let $c_1$ represent character "ك" and $c_2$ represent character "ٲ".
We search for all sequences $(\pi_1, \pi_2, \pi_3, \pi_4) \in (\Omega')^4$ such that $\mathcal{B}(\boldsymbol{\pi}) = (c_1, c_2)$.
Because $c_1 \neq c_2$, intermediate blanks are optional. Valid alignments are:

1. $(c_1, c_2, \epsilon, \epsilon)$
2. $(c_1, \epsilon, c_2, \epsilon)$
3. $(c_1, \epsilon, \epsilon, c_2)$
4. $(\epsilon, c_1, c_2, \epsilon)$
5. $(\epsilon, c_1, \epsilon, c_2)$
6. $(\epsilon, \epsilon, c_1, c_2)$
7. $(c_1, c_1, c_2, \epsilon)$
8. $(c_1, c_1, \epsilon, c_2)$
9. $(c_1, c_1, c_1, c_2)$
10. $(c_1, c_2, c_2, \epsilon)$
11. $(c_1, \epsilon, c_2, c_2)$
12. $(c_1, c_2, c_2, c_2)$
13. $(\epsilon, c_1, c_2, c_2)$
14. $(\epsilon, c_1, c_1, c_2)$

Total cardinality $|\mathcal{B}^{-1}(\mathbf{y})| = \mathbf{14}$ paths.

### Part 2: Alignments for $\mathbf{y}_{\text{dup}} = (c_1, c_1)$ at $T=4$
Because both tokens are identical, they **must be separated by at least one blank token $\epsilon$**. If no blank intervenes, consecutive $c_1$ tokens collapse into a single character: $\mathcal{B}(c_1, c_1) = c_1$.

The required pattern is $c_1 \to \dots \to \epsilon \to \dots \to c_1$.
Enumerating all valid alignments of length $T=4$:

1. $(c_1, \epsilon, c_1, \epsilon)$
2. $(c_1, \epsilon, \epsilon, c_1)$
3. $(\epsilon, c_1, \epsilon, c_1)$
4. $(c_1, c_1, \epsilon, c_1)$
5. $(c_1, \epsilon, c_1, c_1)$

Total cardinality $|\mathcal{B}^{-1}(\mathbf{y}_{\text{dup}})| = \mathbf{5}$ paths.

### Part 3: Combinatorial Comparison
$|\mathcal{B}^{-1}(\mathbf{y})| = 14$ whereas $|\mathcal{B}^{-1}(\mathbf{y}_{\text{dup}})| = 5$.
Repeated identical characters eliminate all direct transition paths without blanks (such as $(c_1, c_1, c_1, c_1)$, $(c_1, c_1, c_1, c_2)$, etc.). The mandatory insertion of at least one $\epsilon$ at $t_{\text{split}}$ reduces the degrees of freedom for frame allocation from $\binom{T}{U}$ to $\binom{T}{U+1}$.

---

## Solution to Problem 3: Shallow Fusion Brevity Bias Calibration

### Part 1: Cause of Extreme Truncation
When $\beta = 0$, the language model score for a candidate sequence $\mathbf{w}$ is:

$$\alpha \sum_{k=1}^K \log P_{\text{LM}}(w_k \mid w_{\lt k})$$

Because every $\log P_{\text{LM}} \le 0$, each emitted word reduces the total hypothesis score by an average of $\alpha \times \bar{H}_{\text{LM}} = 0.50 \times 4.2 = 2.1 \text{ nats}$. The decoder maximizes score by emitting as few words as possible, frequently truncating long utterances after 2 or 3 words.

### Part 2: Minimum Theoretical Value of $\beta$
To neutralize the average negative log-probability penalty per word:

$$\beta_{\text{neutral}} = \alpha \times \bar{H}_{\text{LM}} = 0.50 \times 4.2 = \mathbf{2.10}$$

Setting $\beta \ge 2.10$ provides sufficient positive reward per word to counteract language model brevity bias.

### Part 3: Adjustment when $\alpha$ Doubles
If $\alpha$ doubles from $0.50$ to $1.00$, the average language model penalty per word doubles:

$$\text{Penalty} = 1.00 \times 4.2 = 4.20 \text{ nats}$$

To maintain sequence length balance, $\beta$ must also double to:

$$\beta_{\text{new}} = 1.00 \times 4.2 = \mathbf{4.20}$$

---

## Solution to Problem 4: Speaker Leakage Graph Partitioning

### Part 1: Probability of Speaker Leakage
Let a speaker have 10 files. We sample 20 files without replacement for the test set from 100 total files.
The probability that *none* of the speaker's 10 files are chosen for the test set is:

$$P(\text{all 10 in train}) = \frac{\binom{90}{20}}{\binom{100}{20}} = \frac{\frac{90!}{20! 70!}}{\frac{100!}{20! 80!}} = \frac{80 \times 79 \times \dots \times 71}{100 \times 99 \times \dots \times 91} \approx 0.0898 \text{ (8.98\%)}$$

The probability that *all 10* are in the test set is $\frac{\binom{10}{10}\binom{90}{10}}{\binom{100}{20}} \approx 0.0000$.
Therefore, the probability that the speaker appears in **both** train and test is:

$$P(\text{leakage}) = 1 - P(\text{all in train}) - P(\text{all in test}) \approx 1 - 0.0898 - 0.0000 = \mathbf{0.9102 \text{ (91.02\%)}}$$

There is a $91.0\%$ probability that any given speaker is leaked across both splits!

### Part 2: Mechanism of Artificial Generalization
When speaker leakage occurs, the acoustic model tunes its convolutional filters to the specific formants, pitch ($F_0$), vocal tract length, and room reverberation of those 10 speakers. The model achieves low WER on the test set by recognizing the *speaker's voice profile* rather than decoding speech sounds.

### Part 3: Bipartite Graph Partitioning Algorithm
1. Construct a bipartite graph $G = (V_{\text{spk}}, V_{\text{audio}}, E)$ where edge $(s, a) \in E$ indicates that audio file $a$ was spoken by speaker $s$.
2. Compute connected components $C_1, \dots, C_M$ of $G$.
3. Partition the components: assign entire connected components to either $\mathcal{D}_{\text{train}}$ or $\mathcal{D}_{\text{test}}$, such that:

$$\left( \bigcup_{C \in \mathcal{D}_{\text{train}}} V_{\text{spk}}(C) \right) \cap \left( \bigcup_{C \in \mathcal{D}_{\text{test}}} V_{\text{spk}}(C) \right) = \emptyset$$

This guarantees strictly zero speaker overlap.
"""

# ---------------------------------------------------------------------------
# 9. quiz.md
# ---------------------------------------------------------------------------
QUIZ_CONTENT = r"""# Conceptual Quiz: Lecture 02 (The Anatomy of Modern ASR Pipelines)

## Question 1: Architectural Failure Modes
In low-resource conversational speech containing prolonged pauses, laughter, and ambient acoustic noise, which failure mode is uniquely characteristic of Attention-based Encoder-Decoder (AED) models (e.g., Listen-Attend-Spell, Whisper) but structurally impossible in CTC models?

- [ ] A) Complete failure of convolutional stride downsampling.
- [ ] B) Inability to compute softmax probabilities when the blank token is omitted.
- [ ] C) Autoregressive hallucination loops where the decoder repeats phrases indefinitely without acoustic evidence.
- [ ] D) Negative CTC loss due to zero-width non-joiner characters.

---

## Question 2: Receptive Field and Conv1d Arithmetic
A 16 kHz audio recording contains exactly $1,680$ samples ($0.105$ seconds). How many valid CTC frames will be produced by the 7-layer temporal feature extractor ($K=[10, 3, 3, 3, 3, 2, 2]$, $S=[5, 2, 2, 2, 2, 2, 2]$)?

- [ ] A) 4 frames
- [ ] B) 5 frames
- [ ] C) 8 frames
- [ ] D) 10 frames

---

## Question 3: Acoustic Temperature Scaling
Why does the EduCasheer production pipeline apply acoustic temperature scaling ($T = 1.08 > 1.0$) to logits before KenLM prefix beam search?

- [ ] A) To increase the inference execution speed of the Transformer layers.
- [ ] B) To convert negative CTC logits into positive integer values.
- [ ] C) To soften peaky, overconfident CTC posteriors, allowing the language model to effectively rerank phonetically ambiguous hypotheses.
- [ ] D) To automatically remove diacritics from Kashmiri Perso-Arabic text.

---

## Question 4: Brevity Bias and Shallow Fusion
During prefix beam search decoding with shallow language model fusion ($\log P_{\text{CTC}} + \alpha \log P_{\text{LM}} + \beta |\mathbf{w}|$), what happens if the word insertion bonus $\beta$ is set to $0.0$ while $\alpha = 0.60$?

- [ ] A) The decoder will produce infinite repetitions of the first word.
- [ ] B) The decoder will suffer from severe brevity bias, systematically truncating sentences and dropping valid words.
- [ ] C) The acoustic temperature will collapse to zero.
- [ ] D) The language model score will become strictly positive.

---

## Question 5: Evaluation Integrity
In Stage 04 of the EduCasheer project, a random train/test split yielded a reported WER of $16.8\%$, but a zero-speaker-overlap disjoint evaluation revealed a true WER of $34.2\%$. What was the root cause of this $17.4\%$ WER discrepancy?

- [ ] A) A bug in the Levenshtein distance calculation in `metrics.py`.
- [ ] B) Speaker leakage, where the model memorized speaker vocal tract characteristics and room acoustics rather than learning generalized Kashmiri phonetics.
- [ ] C) KenLM was trained on the test set instead of the training set.
- [ ] D) Audio files in the test set were sampled at 8 kHz instead of 16 kHz.

---

## Answer Key & Distractor Post-Mortems

### Question 1:
- **Correct Answer**: **C**
- **Justification**: In AED models, the autoregressive decoder attends to its own previous emissions $y_{\lt u}$. When acoustic inputs lack distinct speech features (silence or noise), cross-attention fails to ground generation, causing the decoder to generate repetitive text loops based purely on internal language modeling. CTC enforces strict temporal monotonicity and emits blank tokens $\epsilon$ on non-speech segments, preventing hallucination loops.
- **Distractor Autopsy**:
  - *Option A*: Conv1d stride downsampling operates purely on input dimensions regardless of model paradigm.
  - *Option B*: AED models do not use a blank token $\epsilon$; that is a CTC-specific construct.
  - *Option D*: CTC loss is mathematically lower-bounded by $0.0$; it can never be negative.

### Question 2:
- **Correct Answer**: **B**
- **Justification**: Using the valid length formula $T = \left\lfloor \frac{N - 400}{320} \right\rfloor + 1$:

$$T = \left\lfloor \frac{1680 - 400}{320} \right\rfloor + 1 = \left\lfloor \frac{1280}{320} \right\rfloor + 1 = 4 + 1 = 5 \text{ frames}$$

- **Distractor Autopsy**:
  - *Option A*: Omits the $+1$ boundary frame term.
  - *Option C*: Inaccurately assumes a stride of 160 instead of 320.
  - *Option D*: Assumes linear scaling without accounting for the 400-sample receptive field offset.

### Question 3:
- **Correct Answer**: **C**
- **Justification**: Untempered CTC outputs exhibit extreme peakiness where the top token probability approaches $1.0$, rendering language model shallow fusion ineffective during beam pruning. Setting $T = 1.08$ increases entropy, enabling smooth integration with KenLM.
- **Distractor Autopsy**:
  - *Option A*: Temperature scaling is a simple elementwise division; it does not alter Transformer execution speed.
  - *Option B*: Dividing by a scalar does not change logit signs.
  - *Option D*: Diacritic normalization is performed in Stage 6 by `Certified Normalizer v3`, not by logit temperature scaling.

### Question 4:
- **Correct Answer**: **B**
- **Justification**: Because each word language model probability satisfies $\log P_{\text{LM}} \le 0$, every additional emitted word accumulates a negative score. Without a positive word insertion bonus $\beta > 0$ to counteract this penalty, the decoder favors hypotheses with fewer words.
- **Distractor Autopsy**:
  - *Option A*: Repetition loops occur in autoregressive decoders, not in brevity-penalized CTC beam search.
  - *Option C*: Acoustic temperature is an independent hyperparameter and does not depend on $\beta$.
  - *Option D*: Probabilities are $\le 1.0$, so log-probabilities cannot become strictly positive.

### Question 5:
- **Correct Answer**: **B**
- **Justification**: Random splitting distributes utterances from the same human speaker across both train and test. The neural network exploits this speaker overlap by memorizing speaker-specific acoustic features ($F_0$, vocal tract shape), yielding artificially optimistic results.
- **Distractor Autopsy**:
  - *Option A*: Stage 25 and 26 independent audits verified that the Levenshtein distance implementation was mathematically exact.
  - *Option C*: KenLM text contamination was audited and ruled out in Stage 08.
  - *Option D*: Audio resampling to 16 kHz is strictly enforced by `load_audio` across all evaluation manifests.
"""

# ---------------------------------------------------------------------------
# 10. research_connections.md
# ---------------------------------------------------------------------------
RESEARCH_CONNECTIONS_CONTENT = r"""# Research Connections: Repository Mapping and Empirical Provenance

---

## 1. Concrete Research Lineage Across Historical Stages

The architectural principles taught in Lecture 02 are directly derived from the empirical breakthroughs and failure autopsies of the EduCasheer research trajectory:

```
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 01: The Lexicon Bottleneck & Kaldi Hybrid Collapse               │
│ - Script: stages/stage_01/run_urdu_kaldi_baseline.sh                   │
│ - Finding: 38.2% forced alignment rejection on Kashmiri audio          │
│ - Lesson: Hand-engineered phonetic lexicons are unviable for Kashmiri. │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 04 & 16: The Speaker Leakage Discovery & Disjoint Splits         │
│ - Script: stages/stage_04/evaluate_splits.py                           │
│ - Finding: Random split gave 16.8% WER; disjoint split gave 34.2% WER  │
│ - Lesson: Bipartite graph partitioning required for honest evaluation.  │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 09 & 13: B-02C Architectural Stabilization & Vocabulary Design   │
│ - Config: configs/vocab.json (|V| = 47)                                │
│ - Checkpoint: native_asr_training/experiments/B-02C/best_model         │
│ - Lesson: Character CTC avoids BPE Perso-Arabic subword fragmentation. │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 22: Pipeline Calibration & Data Expansion (Checkpoint E22-01)    │
│ - Script: src/decoding/kenlm_decoder.py (T = 1.08, α = 0.50, β = 1.50) │
│ - Finding: Temperature scaling reduces disjoint test WER to 24.12%     │
│ - Lesson: Calibrated shallow fusion unlocks state-of-the-art accuracy. │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Source Code & Configuration Cross-Reference

| Architectural Stage | Repository Source File | Canonical Function / Class |
| :--- | :--- | :--- |
| **Audio Loading & Resampling** | [`src/data_processing/audio_processor.py`](file:///home/musa/Kashmiri%20Project/src/data_processing/audio_processor.py) | `load_audio()`, `normalize_audio()` |
| **Exact CTC Frame Slicing** | [`src/decoding/ctc_slice.py`](file:///home/musa/Kashmiri%20Project/src/decoding/ctc_slice.py) | `compute_exact_ctc_length()`, `slice_exact_logits()` |
| **Acoustic Backbone & Head** | [`src/asr/model.py`](file:///home/musa/Kashmiri%20Project/src/asr/model.py) | `load_asr_model()`, `Wav2Vec2ForCTC` |
| **Calibrated KenLM Decoding** | [`src/decoding/kenlm_decoder.py`](file:///home/musa/Kashmiri%20Project/src/decoding/kenlm_decoder.py) | `CalibratedKenLMDecoder.decode()` |
| **Text Normalization v3** | [`src/normalization/normalizer.py`](file:///home/musa/Kashmiri%20Project/src/normalization/normalizer.py) | `normalize_kashmiri_text()` |
| **End-to-End Orchestrator** | [`src/asr/transcribe.py`](file:///home/musa/Kashmiri%20Project/src/asr/transcribe.py) | `KashmiriASR.transcribe()` |

---

## 3. Empirical Evidence from Checkpoint `E22-01`

Checkpoint `E22-01` (`kashmiri_dataset/experiments/stage22_data_expanded_training/checkpoints/E22-01/best_model/`) represents the certified state-of-the-art model resulting from this pipeline architecture.

### Benchmark Evaluation Across Decoding Configurations:
| Evaluation Manifest | Greedy CTC WER | KenLM Uncalibrated (T=1.00, α=0.3, β=1.0) | KenLM Calibrated (T=1.08, α=0.5, β=1.5) |
| :--- | :---: | :---: | :---: |
| **Clean Kashmiri Disjoint** | 29.80% | 25.82% | **24.12%** |
| **Bolbosh Radio (22.05 kHz)** | 34.20% | 29.15% | **27.40%** |
| **WAAZ Spontaneous Speech** | 58.40% | 55.15% | **51.80%** |
| **IndicVoices Kashmiri** | 31.50% | 27.20% | **25.60%** |

These empirical metrics demonstrate that:
1. Moving from Greedy CTC to Shallow Fusion provides a **3.98% to 5.68% absolute WER reduction** across all domains.
2. Calibrating acoustic temperature ($T=1.08$) and word insertion penalty ($\beta=1.50$) provides an additional **1.60% to 3.35% absolute WER reduction**, certifying the critical value of decoding calibration.
"""

# ---------------------------------------------------------------------------
# 11. references.md
# ---------------------------------------------------------------------------
REFERENCES_CONTENT = r"""# Academic References: Lecture 02 (The Anatomy of Modern ASR Pipelines)

---

## 1. Primary Foundational Literature

1. **Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J.** (2006). *Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks*. In Proceedings of the 23rd International Conference on Machine Learning (ICML '06), pp. 369–376.
   - *Significance*: Introduced the CTC loss function, the collapse operator $\mathcal{B}$, and the forward-backward dynamic programming algorithm for monotonic sequence alignment without pre-segmented phonetic boundaries.

2. **Baevski, A., Zhou, Y., Mohamed, A., & Auli, M.** (2020). *wav2vec 2.0: A framework for self-supervised learning of speech representations*. Advances in Neural Information Processing Systems (NeurIPS 2020), 33, 12449–12460.
   - *Significance*: Established the 7-layer temporal Conv1d feature extractor architecture and the Transformer-based contrastive self-supervised pretraining framework used as the acoustic backbone in EduCasheer.

3. **Babu, A., Wang, C., Tjandra, A., Lakhotia, K., Xu, Q., Goyal, N., Singh, K., von Platen, P., Saraf, Y., Pino, J., Baevski, A., & Auli, M.** (2021). *XLS-R: Self-supervised cross-lingual speech representation learning at scale*. arXiv preprint arXiv:2111.09296.
   - *Significance*: Detailed the 128-language multilingual pretraining of XLS-R (including Indo-Aryan and Dardic speech representations), providing the transfer learning base for Kashmiri acoustic modeling.

4. **Hannun, A. Y., Annapureddy, C., Xu, G., Maas, A. L., & Ng, A. Y.** (2014). *First-pass large vocabulary continuous speech recognition using bi-directional recurrent DNNs*. arXiv preprint arXiv:1408.2868.
   - *Significance*: Introduced the prefix beam search algorithm for CTC decoding with shallow $n$-gram language model integration and the word insertion bonus $\beta$.

5. **Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I.** (2023). *Robust speech recognition via large-scale weak supervision*. In International Conference on Machine Learning (ICML 2023), pp. 28492–28518.
   - *Significance*: Documented the Attention-based Encoder-Decoder architecture of Whisper, providing empirical baselines and highlighting hallucination failure modes on low-resource and noisy speech.

---

## 2. Kashmiri Linguistics and Orthography

6. **Koul, O. N.** (2005). *A Kashmiri Dictionary*. Central Institute of Indian Languages (CIIL), Mysore.
   - *Significance*: Standard reference for Kashmiri Perso-Arabic orthography, documenting the phonemic status of central vowels and palatalization markers.

7. **Wali, K., & Koul, O. N.** (1997). *Kashmiri: A Cognitive-Descriptive Grammar*. Routledge, London.
   - *Significance*: Exhaustive morphological and syntactic analysis of Kashmiri, providing grammatical foundations for language model design and text normalization rules.

8. **Grierson, G. A.** (1915). *A Dictionary of the Kashmiri Language*. Royal Asiatic Society of Bengal, Calcutta.
   - *Significance*: Historical linguistic documentation of Dardic vowel systems and phonetic length distinctions in the Kashmir Valley.
"""

# Mapping of filenames to content
FILES = {
    "README.md": README_CONTENT,
    "lecture_notes.md": LECTURE_NOTES_CONTENT,
    "instructor_script.md": INSTRUCTOR_SCRIPT_CONTENT,
    "mathematical_foundations.md": MATH_FOUNDATIONS_CONTENT,
    "code_walkthrough.md": CODE_WALKTHROUGH_CONTENT,
    "practical_lab.md": PRACTICAL_LAB_CONTENT,
    "exercises.md": EXERCISES_CONTENT,
    "solutions.md": SOLUTIONS_CONTENT,
    "quiz.md": QUIZ_CONTENT,
    "research_connections.md": RESEARCH_CONNECTIONS_CONTENT,
    "references.md": REFERENCES_CONTENT,
}

def main():
    print(f"Applying pristine LaTeX updates to Module 01, Lecture 02 in: {LECTURE_DIR}")
    LECTURE_DIR.mkdir(parents=True, exist_ok=True)
    
    total_lines = 0
    for filename, content in FILES.items():
        file_path = LECTURE_DIR / filename
        stripped_content = content.strip() + "\n"
        file_path.write_text(stripped_content, encoding="utf-8")
        line_count = len(stripped_content.splitlines())
        total_lines += line_count
        print(f"  [+] Wrote {filename:28s} ({line_count:4d} lines)")
        
    print(f"\nSuccessfully updated all 11 files in Lecture 02! Total lines: {total_lines}")

if __name__ == "__main__":
    main()
