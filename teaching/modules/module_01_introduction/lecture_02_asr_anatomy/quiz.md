# Conceptual Quiz: Lecture 02 (The Anatomy of Modern ASR Pipelines)

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
- **Justification**: In AED models, the autoregressive decoder attends to its own previous emissions $y_{<u}$. When acoustic inputs lack distinct speech features (silence or noise), cross-attention fails to ground generation, causing the decoder to generate repetitive text loops based purely on internal language modeling. CTC enforces strict temporal monotonicity and emits blank tokens $\epsilon$ on non-speech segments, preventing hallucination loops.
- **Distractor Autopsy**:
  - *Option A*: Conv1d stride downsampling operates purely on input dimensions regardless of model paradigm.
  - *Option B*: AED models do not use a blank token $\epsilon$; that is a CTC-specific construct.
  - *Option D*: CTC loss is mathematically lower-bounded by $0.0$; it can never be negative.

### Question 2:
- **Correct Answer**: **B**
- **Justification**: Using the valid length formula $T = \lfloor (N - 400) / 320 \rfloor + 1$:
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
