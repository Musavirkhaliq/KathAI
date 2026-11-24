# Lecture 22: N-Gram Language Modeling, Modified Kneser-Ney, and KenLM

---

## 1. The Linguistic Function of the Language Model in ASR

While the acoustic model computes posterior emissions based purely on audio physics, speech contains inherent acoustic ambiguities:
- **Homophones**: Words that sound identical but have different spellings and meanings.
- **Acoustic Masking**: Low-energy vowels or unstressed syllables obscured by room noise.
- **Grammatical Invariants**: Agreement between subject and verb in Kashmiri's strict verb-second (V2) syntax.

The **Language Model (LM)** provides the statistical prior $P(\mathbf{w})$:
$$P(w_1, w_2, \dots, w_U) = \prod_{u=1}^U P(w_u \mid w_1, \dots, w_{u-1})$$

In an $n$-gram model, the Markov assumption restricts context history to the preceding $n-1$ words:
$$P(w_u \mid w_{<u}) \approx P(w_u \mid w_{u-n+1}, \dots, w_{u-1})$$

---

## 2. Smoothing Algorithms: From Jelinek-Mercer to Modified Kneser-Ney

In low-resource Kashmiri text corpora, the vast majority of valid word sequences never appear in training text. A naive Maximum Likelihood Estimate (MLE) assigns zero probability to unseen sequences ($P_{\text{MLE}} = 0$), causing the decoder to discard valid acoustic hypotheses.

### Modified Kneser-Ney Smoothing (The Gold Standard)
Kneser-Ney smoothing introduces a profound linguistic insight: **continuation probability**.
Instead of asking *"How often does word $w$ appear?"*, it asks:
*"In how many distinct contexts does word $w$ appear as a novel continuation?"*

The lower-order continuation probability is:
$$P_{\text{continuation}}(w) = \frac{|\{ w' : C(w', w) > 0 \}|}{\sum_{w''} |\{ w' : C(w', w'') > 0 \}|}$$

**Modified Kneser-Ney** applies **three distinct discount parameters** ($D_1, D_2, D_{3+}$) based on whether an $n$-gram count is 1, 2, or $\ge 3$:

$$P_{\text{MKN}}(w_i \mid w_{i-n+1}^{i-1}) = \frac{C(w_{i-n+1}^i) - D(C(w_{i-n+1}^i))}{C(w_{i-n+1}^{i-1})} + \gamma(w_{i-n+1}^{i-1}) P_{\text{MKN}}(w_i \mid w_{i-n+2}^{i-1})$$

where:
$$D(c) = \begin{cases} 
0 & \text{if } c = 0 \\
D_1 & \text{if } c = 1 \\
D_2 & \text{if } c = 2 \\
D_3 & \text{if } c \ge 3
\end{cases}$$
and $\gamma(\cdot)$ is the backoff normalization weight ensuring probabilities sum to $1.0$.

---

## 3. High-Performance C++ Inference with KenLM

During beam search decoding, the language model is queried hundreds of thousands of times per second. Python implementations are far too slow for real-time speech.

EduCasheer integrates **KenLM** (Heafield, 2011), an industrial C++ library utilizing:
1. **Sorted Probing / Trie Data Structures**: Minimizes memory cache misses.
2. **Quantized Probability Storage**: Compresses $n$-gram backoff weights into 8-bit quantized bins.
3. **Sub-Microsecond Query Latency**: Evaluates $P(w \mid \text{context})$ in $< 200 \text{ nanoseconds}$, allowing beam search to explore large beams without frame latency.
