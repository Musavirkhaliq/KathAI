# Lecture 21: Decoding Strategies: Greedy Search vs. Beam Search

---

## 1. The Optimal Path Search Problem

At every acoustic frame $t \in \{1, \dots, T\}$, the acoustic model outputs a posterior probability distribution over the vocabulary $\Omega' = \Omega \cup \{\epsilon\}$:

$$\mathbf{y}_t = \text{Softmax}(\mathbf{z}_t) \in \Delta^{|\Omega'|-1}$$

The exact maximum a posteriori (MAP) decoding problem seeks the target text sequence $\mathbf{w}^*$ that maximizes the sum of probabilities over all valid alignment paths:

$$\mathbf{w}^* = \arg\max_{\mathbf{w}} P(\mathbf{w} \mid \mathbf{X}) = \arg\max_{\mathbf{w}} \sum_{\boldsymbol{\pi} \in \mathcal{B}^{-1}(\mathbf{w})} \prod_{t=1}^T y_{\pi_t}^t$$

Because the number of possible alignments grows exponentially with sequence length ($|\Omega'|^T \approx 47^{500} \approx 10^{835}$), exact global search is computationally intractable.

---

## 2. Greedy Search and the "Greedy Fallacy"

**Greedy Search (Best Path Decoding)** selects the single highest-probability token at each time step independently, and then collapses the resulting path:

$$\hat{\boldsymbol{\pi}} = (\arg\max_{k} y_k^1, \arg\max_k y_k^2, \dots, \arg\max_k y_k^T)$$
$$\mathbf{w}_{\text{greedy}} = \mathcal{B}(\hat{\boldsymbol{\pi}})$$

### The Greedy Fallacy:
Greedy search operates under the flawed assumption that:
$$\mathcal{B}\left( \arg\max_{\boldsymbol{\pi}} P(\boldsymbol{\pi} \mid \mathbf{X}) \right) \stackrel{?}{=} \arg\max_{\mathbf{w}} \sum_{\boldsymbol{\pi} \in \mathcal{B}^{-1}(\mathbf{w})} P(\boldsymbol{\pi} \mid \mathbf{X})$$

This equality fails routinely in practice.
Consider a toy example where $T = 2$:
- Path $A_1 = (\text{ك}, \text{ٲ})$ has $P = 0.40 \times 0.40 = 0.16 \implies \mathcal{B} = \text{"کٲ"}$
- Path $B_1 = (\text{ت}, \epsilon)$ has $P = 0.35 \times 0.35 = 0.1225 \implies \mathcal{B} = \text{"ت"}$
- Path $B_2 = (\epsilon, \text{ت})$ has $P = 0.35 \times 0.35 = 0.1225 \implies \mathcal{B} = \text{"ت"}$
- Path $B_3 = (\text{ت}, \text{ت})$ has $P = 0.35 \times 0.35 = 0.1225 \implies \mathcal{B} = \text{"ت"}$

Greedy search outputs $\hat{\boldsymbol{\pi}} = (\text{ك}, \text{ٲ})$ with path probability $0.16$.
However, the total marginalized probability of `"ت"` is:
$$P(\text{"ت"}) = P(B_1) + P(B_2) + P(B_3) = 0.1225 + 0.1225 + 0.1225 = \mathbf{0.3675}$$
The true optimal sequence is `"ت"` ($36.75\%$), but greedy search emitted `"کٲ"` ($16\%$)!
In Kashmiri ASR, greedy search results in an **artificially elevated WER (typically 4–6% higher than beam search)**.

---

## 3. CTC Prefix Beam Search Algorithm

**Prefix Beam Search** maintains a pruned set of $W$ highest-scoring text prefixes $\ell$, merging paths that collapse to the same prefix before pruning.

To handle consecutive identical characters correctly, each prefix $\ell$ maintains two distinct probability accumulators:
1. $P_b(\ell, t)$: Total probability mass of all paths collapsing to $\ell$ that **end in a blank token $\epsilon$** at time $t$.
2. $P_{nb}(\ell, t)$: Total probability mass of all paths collapsing to $\ell$ that **end in a non-blank character** at time $t$.

The total probability of prefix $\ell$ at time $t$ is:
$$P(\ell, t) = P_b(\ell, t) + P_{nb}(\ell, t)$$

At each time step $t$, every prefix $\ell$ in the beam is extended by all candidate characters $c \in \Omega'$:
- If extending by blank $\epsilon$:
  $$P_b(\ell, t) = P(\ell, t-1) \cdot y_\epsilon^t$$
- If extending by character $c$:
  - If $c = \text{last}(\ell)$:
    $$P_{nb}(\ell, t) = P_{nb}(\ell, t-1) \cdot y_c^t \quad (\text{repeating identical character without blank})$$
    $$P_{nb}(\ell \circ c, t) = P_b(\ell, t-1) \cdot y_c^t \quad (\text{new character emission following blank})$$
  - If $c \neq \text{last}(\ell)$:
    $$P_{nb}(\ell \circ c, t) = P(\ell, t-1) \cdot y_c^t$$

After accumulating probabilities across all extensions, only the top $W$ prefixes (typically $W = 64 \text{ to } 100$) are retained.
