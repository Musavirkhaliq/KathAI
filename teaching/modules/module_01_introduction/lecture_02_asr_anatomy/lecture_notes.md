# Lecture 02: The Anatomy of Modern ASR Pipelines

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
   $$L: \mathcal{W} \to \mathcal{P}^+ \quad \text{e.g.,} \quad \text{"كٲشُر"} \mapsto \text{/k/ /əː/ /ʃ/ /u/ /r/}$$
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

$$P(\mathbf{y} \mid \mathbf{X}) = \prod_{u=1}^U P(y_u \mid y_{<u}, \mathbf{X}) = \prod_{u=1}^U \text{Softmax}\left( \text{Decoder}(y_{<u}, \text{CrossAttention}(\mathbf{H}, \mathbf{Z}_{\text{dec}})) \right)$$

#### Advantages:
- Completely removes the conditional independence assumption; each token $y_u$ is conditioned on the entire history of previously generated tokens $y_{<u}$.
- Jointly integrates acoustic and language modeling within a single unified neural network.

#### Severe Failure Modes on Low-Resource Kashmiri:
1. **Catastrophic Hallucination Loops**: In autoregressive cross-attention, there is no structural guarantee of temporal monotonicity. When encountering non-speech segments, ambient background noise, coughing, or prolonged pauses (frequent in field recordings like Kashmiri WAAZ sermons or Bolbosh radio broadcasts), Whisper's decoder attends to its own previous outputs $y_{<u}$, entering self-reinforcing repetition loops (e.g., repeating a single phrase 50 times until reaching the maximum sequence length).
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

$$P(\mathbf{w}) = \prod_{k=1}^K P(w_k \mid w_{<k}) \implies \log P(\mathbf{w}) = \sum_{k=1}^K \log P(w_k \mid w_{<k})$$

and each conditional probability is strictly bounded by $P(w_k \mid w_{<k}) \le 1$ (meaning $\log P(w_k \mid w_{<k}) \le 0$), **every additional word adds a negative quantity to the sequence score**. Consequently, an unweighted beam search exhibits an inherent mathematical bias toward generating fewer words (brevity bias), frequently dropping suffixes, postpositions, and monosyllabic auxiliary verbs.

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
3. **Receptive Field Math**: Cumulative stride is $320$ samples ($20$ ms / $50$ Hz); total receptive field is $400$ samples ($25$ ms); valid frame count is $T = \lfloor (N - 400) / 320 \rfloor + 1$.
4. **Vocabulary Discipline**: Use character-level tokens ($|V|=47$) to avoid subword BPE morphological fragmentation in Perso-Arabic cursive script.
5. **Shallow Fusion Triad**: Calibrate acoustic temperature ($T = 1.08$), language model weight ($\alpha = 0.50$), and word insertion bonus ($\beta = 1.50$) to counteract brevity bias and soften overconfident posteriors.
6. **Evaluation Rigor**: Enforce strict zero-speaker-overlap disjoint evaluation to measure authentic generalized acoustic competence rather than speaker memorization.
