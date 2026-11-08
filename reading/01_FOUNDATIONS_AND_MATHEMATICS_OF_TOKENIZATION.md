# Chapter 1: Foundations and Mathematics of Tokenization

---

## 1. Introduction: Why Do Neural Networks Tokenize?

At their mathematical core, deep neural networks (Transformers, Recurrent Networks, State-Space Models) cannot process raw strings of human language. A digital text stream is fundamentally a sequence of bytes represented in an encoding scheme such as ASCII or UTF-8. 

A neural network operates via continuous linear algebraic operations (matrix multiplications, tensor contractions, non-linear activations):
$$\mathbf{h}_{t} = \sigma(\mathbf{W} \mathbf{x}_t + \mathbf{b})$$

To ingest text, discrete symbolic units must be mapped into continuous dense vector representations in $\mathbb{R}^{d_{\text{model}}}$ via an embedding lookup matrix:
$$\mathbf{E} \in \mathbb{R}^{|V| \times d_{\text{model}}}$$
where $|V|$ is the size of the discrete vocabulary and $d_{\text{model}}$ is the hidden dimension of the model.

Furthermore, during autoregressive sequence generation, the model must output a probability distribution over the discrete vocabulary at each step $t$:
$$P(y_t \mid y_{<t}) = \text{softmax}\left(\frac{\mathbf{h}_t \mathbf{E}^\top}{\tau}\right)$$

This architectural requirement immediately exposes two conflicting constraints:
1. **Vocabulary Dimension ($|V|$)**: As $|V|$ grows, the parameter footprint of the embedding table $\mathbf{E}$ and final unembedding/projection layer scales linearly:
   $$\text{Parameters}_{\text{vocab}} = 2 \times |V| \times d_{\text{model}}$$
   In a modern model where $d_{\text{model}} = 4096$, a vocabulary of 250,000 tokens allocates over **2 billion parameters** exclusively to the embedding and unembedding matrices.
2. **Sequence Length ($N$)**: If we keep $|V|$ very small (e.g., character or raw byte level), the sequence length $N$ explodes. In standard Transformer architectures, self-attention scales quadratically with sequence length in both computation and memory:
   $$\text{Attention FLOPs} = 4 B L N^2 d_{\text{model}}$$
   $$\text{KV-Cache Memory} = 2 B L N d_{\text{model}}$$
   where $B$ is batch size and $L$ is the number of transformer layers.

Tokenization is the discrete optimization problem of finding the Pareto-optimal trade-off between **sequence length $N$** and **vocabulary size $|V|$**.

---

## 2. The Granularity Spectrum: Words, Characters, and Subwords

Historically, Natural Language Processing explored three primary discrete representations:

```
[Characters / Bytes] <-------------------> [Subwords] <-------------------> [Full Words]
|V|: 256 - 5,000                           |V|: 8k - 128k                    |V|: 500k - 5M+
Seq Length: Max (4x - 8x)                  Seq Length: Balanced              Seq Length: Min (1x)
OOV Rate: 0%                               OOV Rate: 0% (via Byte Fallback)  OOV Rate: High (>15%)
Morphology: None (Learned implicitly)      Morphology: Preserved / Exposed   Morphology: Ignored
Compute: Quadratic Attention Bottleneck    Compute: Optimal Trade-off        Compute: Unembedding Bottleneck
```

### 2.1 Word-Level Tokenization
In classical NLP (Word2Vec, GloVe, early LSTMs), text was split on whitespace and punctuation into whole lexical words:
$$\text{"The quick brown fox"} \longrightarrow [\text{"The"}, \text{"quick"}, \text{"brown"}, \text{"fox"}]$$

**Fatal Pathologies of Word-Level Tokenization**:
- **Zipfian Tail & Out-of-Vocabulary (OOV) Collapse**: By Zipf's Law, word frequency is inversely proportional to its rank:
  $$f(r) \propto \frac{1}{r^s} \quad (s \approx 1)$$
  A massive fraction of unique words in any corpus occur only once (*hapax legomena*). Any word not explicitly seen during training maps to an unknown token `[UNK]`.
- **Morphological Explosion**: In morphologically rich, agglutinative, or inflected languages (e.g., Kashmiri, Turkish, Finnish, Arabic), a single lexical root can generate hundreds of surface forms through affixation, case marking, and clitic attachment:
  - English: *walk, walks, walking, walked* (4 forms)
  - Turkish: *ev* (house) $\to$ *evlerinizden* (from your houses)
  - Kashmiri: *گَر* (house) $\to$ *گَرَس* (in the house) $\to$ *گَرٕکۍ* (of the house)
  A word-level vocabulary must dedicate separate embedding rows for every single morphological permutation, diluting gradient updates.

### 2.2 Character-Level Tokenization
To eliminate the OOV problem completely, early models proposed operating strictly on individual characters:
$$\text{"کتھ"} \longrightarrow [\text{"ک"}, \text{"ت"}, \text{"ھ"}]$$

**Pathologies of Character-Level Tokenization**:
- **Sequence Length Explosion**: Sentences become 4x to 8x longer. Because vanilla self-attention is $O(N^2)$, processing an 8,000-character document requires 64 million attention operations per head per layer, whereas a 1,500-token subword document requires only 2.25 million operations (**a 28x computational reduction**).
- **Diluted Semantic Information**: An individual character carries almost zero semantic information. The neural network must expend several early layers simply acting as a morphological parser, assembling characters into n-grams before higher-level semantic reasoning can occur.

### 2.3 Subword Tokenization: The Optimal Frontier
Subword tokenization dynamically segments high-frequency character sequences into unified tokens (e.g., `"play"`, `"ing"`, `"house"`), while decomposing rare or novel words into frequent subword chunks (e.g., `"un"` + `"pre"` + `"cedented"`). 

It guarantees:
1. **Zero OOV Rate**: Any unseen word can be represented by falling back to its constituent characters or bytes.
2. **Compact Sequence Length**: Frequent words occupy a single token.
3. **Morphological Generalization**: The model shares parameter weights across related morphological derivations (e.g., `"learn"` is shared across `"learn-ed"`, `"learn-ing"`, and `"re-learn"`).

---

## 3. Information Theory and the Compression Perspective

Tokenization is fundamentally an exercise in **lossless data compression**.

### 3.1 Shannon Entropy and Code Length
According to Shannon's Source Coding Theorem, the theoretical lower bound on the expected code length required to represent a symbol $x$ drawn from an alphabet $X$ with probability distribution $P(x)$ is given by the Shannon Entropy $H(X)$:
$$H(X) = -\sum_{x \in X} P(x) \log_2 P(x)$$

An optimal subword vocabulary allocates shorter code lengths (single tokens) to high-probability strings and longer code lengths (multiple subwords or bytes) to low-probability strings. This directly mirrors **Huffman Coding** applied at the substring level.

### 3.2 Sequence Length vs. Cross-Entropy Loss
In language modeling, the cross-entropy loss per byte $\mathcal{L}_{\text{byte}}$ is related to the loss per token $\mathcal{L}_{\text{token}}$ by:
$$\mathcal{L}_{\text{byte}} = \frac{\mathcal{L}_{\text{token}}}{\text{Bytes per Token}}$$

When a tokenizer achieves a high compression ratio (high bytes per token / low fertility), the model processes more information per forward pass. However, if the vocabulary is made too large, the tokens become extremely rare, and their embedding vectors suffer from undertraining due to sparse gradient updates:
$$\mathbb{E}[\text{Grad Updates for token } v] = N_{\text{total}} \times P(v)$$
If $P(v) \ll 10^{-7}$, the embedding vector $\mathbf{e}_v$ remains near its initialization state, degrading model generalization.

---

## 4. The Core Subword Algorithms: Deep Mathematical Formulations

There are four primary subword tokenization algorithms used across modern AI:
1. **Byte-Pair Encoding (BPE)** (Sennrich et al., 2016)
2. **WordPiece** (Schuster & Nakajima, 2012; Devlin et al., 2018)
3. **Unigram Language Model** (Kudo, 2018)
4. **Byte-Level BPE (BBPE / Tiktoken)** (Radford et al., 2019)

### 4.1 Byte-Pair Encoding (BPE)

Originally designed by Philip Gage in 1994 as a data compression algorithm, BPE was adapted for neural machine translation by Rico Sennrich et al. in 2016.

#### The Training Algorithm
BPE is a **bottom-up, greedy frequency-based merging algorithm**.

```
Input: Corpus C, Target Vocabulary Size K, Base Alphabet V_0
Initialize V = V_0
Represent each word w in C as a tuple of base symbols: (c_1, c_2, ..., c_m)
While |V| < K:
    1. Count frequencies of all adjacent symbol pairs across all words in C:
       freq(s_i, s_j) = sum_{w in C} count((s_i, s_j) in w) * freq(w)
    2. Identify the most frequent pair:
       (s_a, s_b) = argmax_{(s_i, s_j)} freq(s_i, s_j)
    3. If max freq <= 1:
       break
    4. Create new merged symbol: s_new = s_a || s_b
    5. Add s_new to V
    6. Replace all occurrences of (s_a, s_b) with s_new in C
Output: Final vocabulary V, Ordered list of Merge Rules M
```

#### Mathematical Characterization
Let $\Sigma$ be the base alphabet. A merge rule is a mapping:
$$\mu: \Sigma^* \times \Sigma^* \longrightarrow \Sigma^*$$
BPE constructs an ordered sequence of merges $\mathcal{M} = (\mu_1, \mu_2, \dots, \mu_{K - |\Sigma|})$.

#### Inference (Segmentation)
At inference time, segmenting a new string $S$ is strictly deterministic:
1. Decompose $S$ into base characters: $(c_1, c_2, \dots, c_n)$.
2. Iterate through the learned merge rules in the **exact sequential order** they were learned:
   $$\text{For each } (s_a, s_b) \to s_{new} \text{ in } \mathcal{M}:$$
   $$\text{Replace adjacent } s_a, s_b \text{ with } s_{new}$$
3. Return the final token sequence.

---

### 4.2 WordPiece

WordPiece was developed by Schuster and Nakajima (2012) for Japanese and Korean speech recognition and later popularized by Google BERT (Devlin et al., 2018).

#### The Training Algorithm
Like BPE, WordPiece is bottom-up. However, instead of merging based on raw pair frequency, WordPiece selects merges that **maximize the likelihood of an underlying Unigram Language Model** over the training corpus.

Assuming a unigram language model where each subword token $t$ is generated independently:
$$P(C) = \prod_{w \in C} \prod_{t \in w} P(t)$$
The log-likelihood of the corpus is:
$$\log P(C) = \sum_{w \in C} \sum_{t \in w} \log P(t)$$

When considering merging two adjacent subwords $x$ and $y$ into $z = x \circ y$, the change in corpus log-likelihood is directly proportional to their **Pointwise Mutual Information (PMI)**:
$$\text{Score}(x, y) = \frac{\text{count}(x \circ y)}{\text{count}(x) \times \text{count}(y)} \propto \frac{P(x \circ y)}{P(x) P(y)}$$

```
Input: Corpus C, Target Vocabulary Size K
Initialize V with base characters
While |V| < K:
    1. Compute Unigram probabilities P(t) for all current tokens t in V
    2. For every adjacent candidate pair (x, y):
       Score(x, y) = P(xy) / (P(x) * P(y))
    3. Select (x*, y*) = argmax Score(x, y)
    4. Merge x* and y* into a new token z
    5. Add z to V
Output: Vocabulary V
```

#### Inference (Greedy Longest-Match-First)
Unlike BPE which applies an ordered merge table, standard WordPiece uses a greedy longest-match lookup:
1. Given a word $w$ of length $L$, set pointer $i = 0$.
2. Find the longest substring $w[i:j]$ (where $j \le L$) that exists in $V$.
3. If $i > 0$, the subword must match the continuation prefix (e.g., `##subword`).
4. Set $i = j$ and repeat. If no substring is found at index $i$, mark the entire word as `[UNK]`.

---

### 4.3 Unigram Language Model Tokenization

Introduced by Taku Kudo in 2018, the **Unigram Language Model Tokenizer** flips the paradigm: instead of bottom-up merging, it operates **top-down via probabilistic pruning**.

#### Probabilistic Foundation
Let $V$ be a vocabulary of subwords. For a given input sentence $X$, there exist multiple valid segmentations $\mathbf{x} = (x_1, x_2, \dots, x_m)$ such that the concatenation $x_1 \circ x_2 \circ \dots \circ x_m = X$.

Let $\mathcal{S}(X)$ denote the set of all valid segmentations of $X$. Under the unigram assumption, the probability of a specific segmentation $\mathbf{x}$ is:
$$P(\mathbf{x}) = \prod_{i=1}^m p(x_i) \quad \text{where} \quad \sum_{v \in V} p(v) = 1$$

The marginal probability of generating the sentence $X$ across all segmentations is:
$$P(X) = \sum_{\mathbf{x} \in \mathcal{S}(X)} P(\mathbf{x})$$

The overall corpus log-likelihood across corpus $D = \{X_1, X_2, \dots, X_N\}$ is:
$$\mathcal{L} = \sum_{k=1}^N \log P(X_k) = \sum_{k=1}^N \log \left( \sum_{\mathbf{x} \in \mathcal{S}(X_k)} \prod_{x \in \mathbf{x}} p(x) \right)$$

#### The Training Algorithm (Expectation-Maximization + Pruning)
```
1. Initialize a massive seed vocabulary V_seed:
   Include all single characters + frequent substrings of length <= L (e.g., |V_seed| = 500,000).
2. Repeat until |V| reaches target size K:
    a. Run Expectation-Maximization (EM) algorithm:
       - E-Step: Compute posterior expectations of token frequencies given current p(v)
                 using forward-backward trellis algorithms.
       - M-Step: Update token probabilities:
                 p(v) = Expected_Count(v) / Total_Expected_Counts
    b. For each token v in V:
       Compute the loss penalty if v were removed:
       Loss_Drop(v) = L(V) - L(V \ {v})
       (Single characters are given Loss_Drop = infinity to guarantee 0% OOV).
    c. Sort tokens by Loss_Drop(v).
    d. Prune the bottom eta% (e.g., 10% to 20%) of tokens that contribute least to likelihood.
3. Final Vocabulary V with optimal token probabilities p(v).
```

#### Inference (Viterbi Algorithm)
To find the most likely segmentation $\mathbf{x}^*$ for a sentence $X$:
$$\mathbf{x}^* = \arg\max_{\mathbf{x} \in \mathcal{S}(X)} P(\mathbf{x}) = \arg\max_{\mathbf{x} \in \mathcal{S}(X)} \sum_{i=1}^m \log p(x_i)$$
This is computed in $O(|X| \times \max_{v \in V}|v|)$ time using dynamic programming via the **Viterbi Algorithm**.

---

### 4.4 The SentencePiece Framework

Developed by Kudo and Richardson (Google, 2018), **SentencePiece** is not a distinct loss function, but a universal preprocessing and tokenization architecture that solves a critical flaw of classical BPE and WordPiece: **the whitespace assumption**.

#### The Whitespace Pathology
Classical tokenizers require a language-specific pre-tokenization step that splits text on whitespace before learning subwords:
`"I love apples"` $\to$ `["I", "love", "apples"]` $\to$ BPE.

This causes three catastrophic failures:
1. **Non-Segmenting Languages**: Languages such as Chinese, Japanese, Thai, and Lao do not use spaces between words. Pre-tokenization requires brittle external dictionary segmenters (e.g., Jieba, MeCab).
2. **Loss of Exact Reversibility (Detokenization)**: Once text is split on whitespace, information about spacing is permanently destroyed:
   `"hello, world"` $\to$ `["hello", ",", "world"]` $\to$ cannot distinguish whether the original was `"hello, world"`, `"hello ,world"`, or `"hello , world"`.
3. **Compound Words & Clitics**: In languages with compound words (German: *Donaudampfschifffahrt*) or clitics (Kashmiri: *گَرَس-مَنٛز*), spaces do not reliably reflect lexical boundaries.

#### The SentencePiece Solution: Lossless Tokenization
SentencePiece treats the entire input text as a continuous stream of raw Unicode characters. Whitespace is explicitly mapped to a visible Unicode meta-symbol:
$$\text{Whitespace (' ')} \longrightarrow \text{Lower One Eighth Block (' ' / U+2581)}$$

Example:
$$\text{"KathAI is great"} \longrightarrow \text{"_KathAI_is_great"}$$

Because whitespace is just another regular character in the alphabet:
- Detokenization is trivial and 100% lossless:
  $$\text{Original} = \text{Replace}("\_", \text{" "}, \text{Concatenate}(\text{tokens}))$$
- SentencePiece implements both **BPE** and **Unigram** as underlying engines directly on this space-unified string.

---

### 4.5 Byte-Level BPE (BBPE / Tiktoken)

Introduced in GPT-2 (Radford et al., 2019) and used in GPT-3, GPT-4, LLaMA, Mistral, and Tiktoken, **Byte-Level BPE** operates on raw UTF-8 bytes rather than Unicode code points.

#### Mathematical Foundation
UTF-8 encodes every Unicode character using between 1 and 4 bytes (8-bit integers from `0x00` to `0xFF`):
- ASCII characters (`A-Z`, `a-z`, `0-9`): 1 byte (`0x00` – `0x7F`)
- Latin extensions, Arabic, Hebrew: 2 bytes (`0xC0` – `0xDF`, followed by continuation bytes)
- Devanagari, Perso-Arabic extensions, East Asian (CJK): 3 bytes (`0xE0` – `0xEF`)
- Emojis, rare scripts: 4 bytes (`0xF0` – `0xF7`)

#### The BBPE Innovation: Base Vocabulary of 256
Instead of initializing $V_0$ with all known Unicode characters (which can exceed 150,000 glyphs), BBPE initializes the base alphabet with strictly **the 256 possible byte values**:
$$V_0 = \{0\text{x}00, 0\text{x}01, \dots, 0\text{xFF}\}$$

**Crucial Advantages**:
1. **Absolute Zero Out-Of-Vocabulary**: Every valid UTF-8 digital file on Earth can be decomposed into bytes from $V_0$. The token `[UNK]` is rendered completely obsolete.
2. **Compact Base Table**: Only 256 initial rows are required before merges begin.

#### Regex Pre-Tokenization Splits
If pure BBPE is run naively on raw bytes, it merges punctuation with letters, numbers with words, and across sentence boundaries:
e.g., `"dog."` $\to$ token `"dog."`.
To prevent this catastrophic semantic contamination, modern BBPE tokenizers (GPT-4 `cl100k_base`, LLaMA-3) enforce strict **regex pre-tokenization**. 

The input text is split into non-mergeable partitions using a regular expression before byte merging can occur:

```python
# LLaMA-3 / GPT-4 Style Regex Pre-tokenizer Pattern
TIKTOKEN_REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
```

**What this regex enforces**:
1. Contractions (`'s`, `'ve`) are isolated as separate blocks.
2. Letter sequences (`\p{L}+`) cannot merge with numbers (`\p{N}`).
3. Numbers are chunked in groups of at most 3 digits (`\p{N}{1,3}`) to prevent arbitrary astronomical number tokenization.
4. Punctuation clusters cannot merge with alphabetic text.

---

## 5. Subword Regularization and Stochastic Tokenization

Deterministic tokenization has a subtle flaw: it creates an artificial 1-to-1 bias between a surface string and its token sequence. The model never learns that a word can be segmented in multiple valid ways, making it fragile to typos, orthographic variants, and dialectal shifts.

### 5.1 Unigram Subword Sampling (Kudo, 2018)
Because the Unigram model computes probabilistic scores for every segmentation, we can sample segmentations during training from the posterior distribution:
$$P(\mathbf{x} \mid X) = \frac{P(\mathbf{x})^\alpha}{\sum_{\mathbf{x}' \in \mathcal{S}(X)} P(\mathbf{x}')^\alpha}$$
where $\alpha \in [0, 1]$ is a sampling temperature:
- $\alpha \to \infty$: Reverts to deterministic Viterbi decoding (1-best).
- $\alpha = 1$: Samples from the true posterior distribution.
- $\alpha = 0$: Uniform sampling over all valid segmentations.

### 5.2 BPE-Dropout (Provilkov et al., 2020)
In standard BPE, merge rules are applied with 100% certainty. In **BPE-Dropout**, at each step of segmentation during training, every merge rule in the merge table is stochastically dropped with probability $p \in [0, 1]$:
$$\text{Merge}(s_a, s_b) = \begin{cases} s_{new} & \text{with probability } 1 - p \\ s_a, s_b & \text{with probability } p \end{cases}$$

**Impact on Neural Training**:
- Acts as a powerful subword-level data augmentation.
- Exposes the model to both whole words and decomposed sub-character morphemes.
- Drastically improves model robustness to misspelled text, morphological variations, and low-resource dialectal discrepancies.

---

## 6. The Frontier: Tokenization-Free & Byte-Level Architectures

Recognizing the inherent pathologies of subword tokenizers, recent research has explored eliminating tokenizers entirely:

| Model Architecture | Core Mechanism | Strengths | Bottlenecks |
| :--- | :--- | :--- | :--- |
| **ByT5** (Xue et al., 2022) | T5 encoder-decoder operating directly on UTF-8 bytes with local downsampling. | Immune to OOV, handles noise/typos flawlessly, no tokenizer training. | 3x–4x slower inference than subword T5. |
| **MegaByte** (Meta, 2023) | Hierarchical multi-scale transformer: Global patch transformer + Local byte model. | Sub-quadratic scaling over raw byte sequences of 100k+ length. | High architectural complexity, custom CUDA kernels required. |
| **MambaByte** (Wang et al., 2024) | Pure State Space Model (Mamba) processing raw byte tokens autoregressively. | Linear $O(N)$ computational scaling with constant-memory KV state. | Slower training throughput on standard matrix-multiply accelerators (GPUs). |

Despite these advances, **subword tokenizers remain the dominant operational paradigm** for 99% of production LLMs (GPT-4o, LLaMA-3.3, Qwen-2.5, Claude 3.5, Gemini 1.5). Understanding and optimizing tokenization is therefore the single highest-leverage intervention in modern language model engineering.
