# Lecture 12: Multi-Head Self-Attention and Transformers for Speech

---

## 1. The Attention Mechanism in Acoustic Context Modeling

Speech is inherently long-range contextual. In Kashmiri, verb-second (V2) word order and case inflections mean that an acoustic realization early in a sentence can govern agreement across several seconds of audio.

While Recurrent Neural Networks (RNNs) suffer from vanishing gradients and sequential $\mathcal{O}(T)$ step latency, the **Transformer** models all pairwise temporal relationships in parallel using **Self-Attention**.

Given an input sequence of latent vectors $\mathbf{X} \in \mathbb{R}^{T \times d_{\text{model}}}$, linear projections create Queries ($\mathbf{Q}$), Keys ($\mathbf{K}$), and Values ($\mathbf{V}$):

$$\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{X} \mathbf{W}_V$$

where $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d_{\text{model}} \times d_k}$.

---

## 2. Scaled Dot-Product Attention and the $\frac{1}{\sqrt{d_k}}$ Proof

The core attention equation is:
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \right) \mathbf{V}$$

### Why the Scale Factor $\frac{1}{\sqrt{d_k}}$ is Mathematically Mandatory:
Assume query components $q_i$ and key components $k_i$ are independent random variables with mean zero and unit variance:
$$\mathbb{E}[q_i] = \mathbb{E}[k_i] = 0, \quad \text{Var}(q_i) = \text{Var}(k_i) = 1$$

The unscaled dot product of two $d_k$-dimensional vectors is:
$$z = \mathbf{q} \cdot \mathbf{k} = \sum_{i=1}^{d_k} q_i k_i$$

The expected value of $z$ is:
$$\mathbb{E}[z] = \sum_{i=1}^{d_k} \mathbb{E}[q_i k_i] = \sum_{i=1}^{d_k} \mathbb{E}[q_i] \mathbb{E}[k_i] = 0$$

The variance of $z$ is:
$$\text{Var}(z) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = \sum_{i=1}^{d_k} \left( \mathbb{E}[q_i^2 k_i^2] - (\mathbb{E}[q_i k_i])^2 \right) = \sum_{i=1}^{d_k} \text{Var}(q_i) \text{Var}(k_i) = \sum_{i=1}^{d_k} 1 = d_k$$

For large feature dimensions (e.g., $d_k = 64$ in XLS-R), the variance of the unscaled dot product is **64**, and the standard deviation is **8.0**!
In this high-variance regime, the largest logits dominate, driving the Softmax function into regions with near-zero gradients ($\text{Softmax}'(z) \to 0$), causing catastrophic training stagnation.

Scaling by $\frac{1}{\sqrt{d_k}}$ preserves unit variance:
$$\text{Var}\left( \frac{\mathbf{q} \cdot \mathbf{k}}{\sqrt{d_k}} \right) = \frac{1}{d_k} \text{Var}(\mathbf{q} \cdot \mathbf{k}) = \frac{d_k}{d_k} = 1$$
This keeps the Softmax arguments in a sensitive, well-conditioned gradient regime.

---

## 3. Multi-Head Attention Architecture

Rather than performing a single attention function with dimension $d_{\text{model}}$, **Multi-Head Attention** projects queries, keys, and values $h$ times with distinct learnable matrices:

$$\text{MHA}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \mathbf{W}_O$$
$$\text{head}_i = \text{Attention}(\mathbf{Q} \mathbf{W}_Q^{(i)}, \mathbf{K} \mathbf{W}_K^{(i)}, \mathbf{V} \mathbf{W}_V^{(i)})$$

In XLS-R (300M parameter model):
- Hidden dimension: $d_{\text{model}} = 1024$
- Number of heads: $h = 16$
- Per-head dimension: $d_k = \frac{1024}{16} = 64$
- Feed-forward intermediate dimension: $d_{\text{ff}} = 4096$
- Number of layers: $L = 24$

Each head attends to distinct linguistic phenomena: some heads specialize in local co-articulation (adjacent frames), while others capture sentence-level intonation and prosodic phrases.
