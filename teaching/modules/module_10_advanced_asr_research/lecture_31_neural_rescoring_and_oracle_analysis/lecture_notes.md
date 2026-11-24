# Lecture 31: Neural Language Model Rescoring and Oracle WER Analysis

---

## 1. The Asymptotic Limit of N-Gram Language Models

While $n$-gram language models (such as KenLM 5-gram with Modified Kneser-Ney) provide sub-microsecond query speeds during first-pass beam search, they suffer from fundamental representational limits:
- **Strict Context Horizon**: A 5-gram model considers only the preceding 4 words ($n-1 = 4$). It cannot track long-distance syntactic dependencies, discourse topic, or document-level semantics.
- **Surface Form Sparsity**: $n$-grams treat each word token as an atomic discrete entity. They have no notion of semantic similarity (e.g., words like `پٔہٲڑ` /mountain/ and `کۄہ` /peak/ share zero parameters).

To break through the 20% WER barrier, state-of-the-art ASR systems employ a **two-pass decoding architecture**:
1. **First-Pass Beam Search**: Fast, lightweight decoding with an $n$-gram LM generates an $N$-best list of diverse candidate hypotheses $\mathcal{H} = (\mathbf{w}_1, \dots, \mathbf{w}_N)$.
2. **Second-Pass Neural Rescoring**: A deep pretrained Neural Language Model (NLM)—such as an autoregressive Transformer or Masked Language Model—re-ranks the $N$-best candidates using full bidirectional or long-range context.

---

## 2. Oracle Word Error Rate: Measuring the Theoretical Performance Ceiling

Before deploying second-pass rescoring, researchers must compute the **Oracle WER** to determine whether the first-pass beam search actually contains correct transcriptions within its $N$-best candidate pool.

The Oracle WER selects the single candidate from the $N$-best list that minimizes Levenshtein edit distance against the ground-truth reference $\mathbf{y}^*$:

$$\mathbf{w}_{\text{oracle}} = \arg\min_{\mathbf{w} \in \mathcal{H}} \mathcal{W}(\mathbf{w}, \mathbf{y}^*)$$
$$\text{WER}_{\text{oracle}} = \frac{\sum_{i=1}^M \mathcal{W}(\mathbf{w}_{\text{oracle}}^{(i)}, \mathbf{y}^{*(i)})}{\sum_{i=1}^M N_i}$$

### The EduCasheer Oracle Breakthrough:
In Stage 24, an oracle analysis was conducted across the 8-domain disjoint benchmark with beam width $W = 100$:
- **First-Pass 1-Best WER**: **24.12%**
- **100-Best Oracle WER**: **13.40%**

This revealed a massive **10.72% absolute accuracy gap**!
The acoustic model already generates candidates containing correct words for over 86% of speech; the first-pass $n$-gram model simply lacks the semantic sophistication to rank them at position #1.
Second-pass neural rescoring provides the mathematical mechanism to bridge this 10.72% gap.

---

## 3. Second-Pass Re-Ranking Formulation

Given candidate hypothesis $\mathbf{w} \in \mathcal{H}$, the second-pass score combines acoustic likelihood, first-pass $n$-gram probability, second-pass neural LM likelihood, and word length:

$$\mathcal{S}_{\text{rescore}}(\mathbf{w}) = \log P_{\text{CTC}}(\mathbf{w} \mid \mathbf{X}) + \alpha_1 \log P_{\text{ngram}}(\mathbf{w}) + \alpha_2 \log P_{\text{neural}}(\mathbf{w}) + \beta |\mathbf{w}|$$

where $\alpha_2$ is tuned via validation grid search.
