# Lecture 02: The Anatomy of Modern ASR Pipelines

---

## 1. Architectural Paradigms: Hybrid vs. End-to-End

Modern speech recognition has undergone a paradigm shift from modular, multi-component hybrid architectures to unified **End-to-End (E2E)** neural networks. To understand why modern systems are structured as they are, we must compare the three dominant design philosophies:

1. **Classical Hybrid (HMM-GMM / HMM-DNN)**:
   - **Acoustic Model**: Models $P(X \mid s)$, mapping frame-level features (e.g., 39-dimensional MFCCs) to hidden HMM phone states $s$.
   - **Pronunciation Lexicon ($L$)**: Hand-crafted deterministic or probabilistic mapping from phonemes to words (e.g., `/k/ /oː/ /ʃ/ /u/ /r/ \to \text{كٲشُر}`).
   - **Language Model**: $n$-gram or neural $P(W)$.
   - **Decoder**: Weighted Finite-State Transducer (WFST) composition $H \circ C \circ L \circ G$.
   - **Fatal Weakness for Kashmiri**: Building the pronunciation lexicon $L$ requires phonetic transcriptions for tens of thousands of words. In Kashmiri, phonemic central vowels and non-standardized orthography make comprehensive phonetic lexicons practically unattainable.

2. **Attention-based Encoder-Decoder (AED / Listen-Attend-Spell / Whisper)**:
   - Utilizes cross-attention between an acoustic encoder and an autoregressive text decoder:
     $$P(y_u \mid y_{<u}, \mathbf{X}) = \text{Softmax}(\text{Decoder}(y_{<u}, \text{CrossAttend}(\mathbf{h}, \mathbf{z})))$$
   - Advantage: Completely eliminates the conditional independence assumption.
   - Disadvantage: Prone to catastrophic hallucination, looping repetitions, and high latency during streaming or long-form inference. In low-resource settings, AED architectures overfit rapidly to small text corpora.

3. **Connectionist Temporal Classification (CTC) / Transducer (RNN-T)**:
   - **CTC**: Maps input sequence $\mathbf{X}_{1:T}$ directly to output character/subword sequence $\mathbf{Y}_{1:U}$ by introducing a special blank token $\epsilon$ and collapsing redundant time steps:
     $$\mathcal{B}(\pi_1, \dots, \pi_T) = \mathbf{y}$$
   - **RNN-T**: Augments CTC with a prediction network modeling previous label history, evaluated via forward-backward over a 2D time-label lattice.
   - **Why EduCasheer Selected CTC over XLS-R**:
     - Pretrained Wav2Vec2/XLS-R encoders provide deep contextual representations, mitigating CTC's conditional independence assumption.
     - Perfect alignment monotonicity prevents hallucinations.
     - Linear time complexity $\mathcal{O}(T)$ allows fast decoding and robust shallow fusion with external language models.

---

## 2. Structural Decomposition of the EduCasheer Kashmiri Pipeline

The canonical EduCasheer production pipeline (exemplified in `src/inference/pipeline.py` and checkpoint `E22-01`) comprises six tightly coupled stages:

```
[Raw Audio (16 kHz, 16-bit Mono)]
              │
              ▼
    [1. Audio Front-End] ───────── Resampling, mono mix, amplitude normalization [-1.0, 1.0]
              │
              ▼
    [2. Acoustic Encoder] ──────── 7-layer CNN Feature Extractor (stride 320 = 20ms frames)
              │                    + 24-layer Transformer (1024-dim, 16 heads) [XLS-R 300M]
              ▼
    [3. Linear Projection] ─────── Projection from 1024-dim hidden state to |V| = 47 logits
              │
              ▼
    [4. Log-Softmax & Slicing] ── Exact CTC frame slicing, temperature scaling T = 1.08
              │
              ▼
    [5. Prefix Beam Search] ────── Beam width W = 100, shallow fusion with 5-gram KenLM
              │                    Score = log P_CTC + α * log P_LM + β * |w|
              ▼
    [6. Unicode Normalization] ── Certified Normalizer v3 (Homoglyph unification, diacritic strip)
              │
              ▼
    [Final Transcribed Text]
```

---

## 3. The Vocabulary and Tokenizer Architecture

In low-resource languages with complex inflectional morphology like Kashmiri, choosing the modeling unit is critical.

### Subword (BPE / WordPiece) vs. Character-Level Modeling

| Dimension | Subword (BPE 1k-4k) | Character-Level (|V| = 47) |
| :--- | :--- | :--- |
| **Out-of-Vocabulary (OOV)** | Rare, but breaks into arbitrary fragments | Zero OOV; any legitimate Kashmiri word can be spelled |
| **Data Requirements** | Requires millions of words to train stable merges | Requires zero pre-tokenization training |
| **Alignment Granularity** | Variable length tokens distort temporal alignment | Uniform frame-to-character temporal mapping |
| **Kashmiri Perso-Arabic Suitability** | High failure rate: merges split attached letters incorrectly | Matches native orthography and cursive joining rules |

EduCasheer adopted a **strict character-level vocabulary of 47 tokens** (`configs/vocab.json`), including:
- 38 Perso-Arabic alphabetic characters (e.g., `ا`, `ب`, `پ`, `ت`, `ٹ`, `ث`, `ج`, `چ`, `ح`, `خ`, etc.)
- Kashmiri-specific central vowel carriers (`ٲ`, `ٳ`, `ۄ`, `ۆ`, `ؠ`)
- Special tokens: `<pad>` (0), `<unk>` (1), `<s>` (2), `</s>` (3), `|` (word boundary / space), and `<blank>` (CTC blank token).

---

## 4. The Decoding Triad: CTC Logits, Language Model, and Length Penalty

At time step $t$, the acoustic model outputs a unnormalized logit vector $\mathbf{z}_t \in \mathbb{R}^{|V|}$. Applying calibrated temperature scaling yields posterior probabilities:

$$P(\pi_t = k \mid \mathbf{X}) = \frac{\exp(z_{t,k} / T)}{\sum_{j=1}^{|V|} \exp(z_{t,j} / T)}$$

During beam search decoding, the total score of hypothesis $\mathbf{w}$ is evaluated as:

$$\mathcal{S}(\mathbf{w} \mid \mathbf{X}) = \log P_\text{CTC}(\mathbf{w} \mid \mathbf{X}) + \alpha \log P_\text{LM}(\mathbf{w}) + \beta |\mathbf{w}|$$

Where:
- $\alpha$ (LM weight, typically $0.30 - 0.50$): Balances acoustic evidence against syntactic plausibility. Overly high $\alpha$ causes the decoder to replace valid acoustic names with common $n$-grams.
- $\beta$ (Word insertion penalty, typically $1.00 - 1.50$): Compensates for the inherent brevity bias of character-level language models. Without $\beta$, the decoder favors short, deleted hypotheses.
- $T$ (Acoustic temperature, $T = 1.08$): Softens overconfident peaky CTC distributions, preventing the search from prematurely pruning valid candidate branches.

---

## 5. Disjoint Evaluation: Why Data Splitting Dictates Pipeline Integrity

A speech recognition pipeline cannot be deemed functional based on random train/test splits. In speech processing, **random splitting creates severe speaker leakage**:
- The model memorizes speaker-specific fundamental frequencies ($F_0$), vocal tract lengths, and room reverberation profiles.
- On randomly split test sets, Word Error Rate (WER) drops by 10–18% absolute, providing a false impression of generalized competence.

In Module 03 and Module 08, we study the **Zero-Speaker-Overlap Graph Partitioning** that enforces strict disjoint evaluation across all 8 testing domains.
