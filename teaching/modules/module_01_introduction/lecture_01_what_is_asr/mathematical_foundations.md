# Mathematical Foundations: Decision Theory, Bayes Risk, and MAP Decoding

---

## 1. Rigorous Proof: Equivalence of Zero-One Loss and MAP Sequence Decoding

Let $\mathcal{W}$ be the discrete countable set of all permissible linguistic word sequences over vocabulary $\mathcal{V}$.
Let $\mathbf{X} \in \mathcal{X}$ be the observed continuous acoustic feature tensor.
Let $P(\mathbf{w} \mid \mathbf{X})$ denote the true posterior probability distribution over $\mathcal{W}$ given observation $\mathbf{X}$, satisfying:
$$\sum_{\mathbf{w} \in \mathcal{W}} P(\mathbf{w} \mid \mathbf{X}) = 1, \quad P(\mathbf{w} \mid \mathbf{X}) \ge 0 \quad \forall \mathbf{w} \in \mathcal{W}$$

Let $\delta: \mathcal{X} \to \mathcal{W}$ be a deterministic decision rule mapping acoustic observation $\mathbf{X}$ to hypothesized sequence $\hat{\mathbf{w}} = \delta(\mathbf{X})$.

### Theorem 1:
Under the Zero-One Sentence Error loss function:
$$L_{0-1}(\mathbf{w}, \hat{\mathbf{w}}) = \mathbb{I}(\mathbf{w} \neq \hat{\mathbf{w}}) = \begin{cases} 0 & \text{if } \mathbf{w} = \hat{\mathbf{w}} \\ 1 & \text{if } \mathbf{w} \neq \hat{\mathbf{w}} \end{cases}$$
the decision rule $\delta^*(\mathbf{X})$ that minimizes the expected Bayes risk is strictly the **Maximum A Posteriori (MAP)** decision rule:
$$\delta^*(\mathbf{X}) = \arg\max_{\mathbf{w} \in \mathcal{W}} P(\mathbf{w} \mid \mathbf{X})$$

### Proof:
The conditional Bayes risk $R(\hat{\mathbf{w}} \mid \mathbf{X})$ for selecting hypothesis $\hat{\mathbf{w}}$ is defined as:
$$R(\hat{\mathbf{w}} \mid \mathbf{X}) = \mathbb{E}_{\mathbf{w} \sim P(\mathbf{w} \mid \mathbf{X})} [L_{0-1}(\mathbf{w}, \hat{\mathbf{w}})] = \sum_{\mathbf{w} \in \mathcal{W}} L_{0-1}(\mathbf{w}, \hat{\mathbf{w}}) P(\mathbf{w} \mid \mathbf{X})$$

Substitute the definition of $L_{0-1}$:
$$R(\hat{\mathbf{w}} \mid \mathbf{X}) = \sum_{\mathbf{w} \in \mathcal{W}, \mathbf{w} \neq \hat{\mathbf{w}}} (1) \cdot P(\mathbf{w} \mid \mathbf{X}) + (0) \cdot P(\hat{\mathbf{w}} \mid \mathbf{X}) = \sum_{\mathbf{w} \neq \hat{\mathbf{w}}} P(\mathbf{w} \mid \mathbf{X})$$

Since the total probability mass over all sequences sums to 1:
$$\sum_{\mathbf{w} \in \mathcal{W}} P(\mathbf{w} \mid \mathbf{X}) = P(\hat{\mathbf{w}} \mid \mathbf{X}) + \sum_{\mathbf{w} \neq \hat{\mathbf{w}}} P(\mathbf{w} \mid \mathbf{X}) = 1$$
it follows directly that:
$$\sum_{\mathbf{w} \neq \hat{\mathbf{w}}} P(\mathbf{w} \mid \mathbf{X}) = 1 - P(\hat{\mathbf{w}} \mid \mathbf{X})$$

Therefore, the conditional risk is:
$$R(\hat{\mathbf{w}} \mid \mathbf{X}) = 1 - P(\hat{\mathbf{w}} \mid \mathbf{X})$$

We seek the decision rule $\hat{\mathbf{w}}^* = \delta^*(\mathbf{X})$ that minimizes conditional risk:
$$\hat{\mathbf{w}}^* = \arg\min_{\hat{\mathbf{w}} \in \mathcal{W}} R(\hat{\mathbf{w}} \mid \mathbf{X}) = \arg\min_{\hat{\mathbf{w}} \in \mathcal{W}} [1 - P(\hat{\mathbf{w}} \mid \mathbf{X})]$$

Since minimizing $1 - f(x)$ is mathematically equivalent to maximizing $f(x)$:
$$\hat{\mathbf{w}}^* = \arg\max_{\hat{\mathbf{w}} \in \mathcal{W}} P(\hat{\mathbf{w}} \mid \mathbf{X})$$
This completes the proof. $\blacksquare$

---

## 2. Invariance of the Marginal Likelihood (Evidence) in MAP Decoding

Applying Bayes' Rule to the posterior:
$$P(\mathbf{w} \mid \mathbf{X}) = \frac{P(\mathbf{X} \mid \mathbf{w}) P(\mathbf{w})}{P(\mathbf{X})}$$

### Proposition 2:
The denominator $P(\mathbf{X})$ has zero influence on the argmax decision:
$$\arg\max_{\mathbf{w} \in \mathcal{W}} \frac{P(\mathbf{X} \mid \mathbf{w}) P(\mathbf{w})}{P(\mathbf{X})} = \arg\max_{\mathbf{w} \in \mathcal{W}} [P(\mathbf{X} \mid \mathbf{w}) P(\mathbf{w})]$$

### Proof:
By the Law of Total Probability:
$$P(\mathbf{X}) = \sum_{\mathbf{w}' \in \mathcal{W}} P(\mathbf{X}, \mathbf{w}') = \sum_{\mathbf{w}' \in \mathcal{W}} P(\mathbf{X} \mid \mathbf{w}') P(\mathbf{w}')$$
Notice that $\mathbf{w}'$ is a dummy summation index over all candidate sequences in $\mathcal{W}$. Thus, $P(\mathbf{X})$ is a scalar function strictly of the acoustic tensor $\mathbf{X}$ and does not depend on the specific candidate hypothesis $\mathbf{w}$ being evaluated.

Since $P(\mathbf{X}) > 0$ for any realizable observation $\mathbf{X}$, multiplying the objective function by the strictly positive constant $\frac{1}{P(\mathbf{X})}$ preserves the order of all elements in $\mathcal{W}$:
$$P(\mathbf{w}_1 \mid \mathbf{X}) > P(\mathbf{w}_2 \mid \mathbf{X}) \iff \frac{P(\mathbf{X} \mid \mathbf{w}_1) P(\mathbf{w}_1)}{P(\mathbf{X})} > \frac{P(\mathbf{X} \mid \mathbf{w}_2) P(\mathbf{w}_2)}{P(\mathbf{X})} \iff P(\mathbf{X} \mid \mathbf{w}_1) P(\mathbf{w}_1) > P(\mathbf{X} \mid \mathbf{w}_2) P(\mathbf{w}_2)$$
Hence, the argmax operator yields the exact same maximizing element. $\blacksquare$

---

## 3. Minimum Bayes Risk (MBR) under Word Error Rate Loss

While zero-one loss optimizes Sentence Error Rate (SER), speech recognition benchmarks are evaluated on Word Error Rate (WER), defined by the Levenshtein distance:
$$\mathcal{W}(\mathbf{w}, \hat{\mathbf{w}}) = \frac{\text{EditDistance}(\mathbf{w}, \hat{\mathbf{w}})}{|\mathbf{w}|} = \frac{S(\mathbf{w}, \hat{\mathbf{w}}) + D(\mathbf{w}, \hat{\mathbf{w}}) + I(\mathbf{w}, \hat{\mathbf{w}})}{|\mathbf{w}|}$$

Under Levenshtein loss $\mathcal{W}(\mathbf{w}, \hat{\mathbf{w}})$, the conditional Bayes risk is:
$$R_{\text{WER}}(\hat{\mathbf{w}} \mid \mathbf{X}) = \sum_{\mathbf{w} \in \mathcal{W}} \mathcal{W}(\mathbf{w}, \hat{\mathbf{w}}) P(\mathbf{w} \mid \mathbf{X})$$

The MBR decision rule is:
$$\hat{\mathbf{w}}_{\text{MBR}} = \arg\min_{\hat{\mathbf{w}} \in \mathcal{W}} \sum_{\mathbf{w} \in \mathcal{W}} \mathcal{W}(\mathbf{w}, \hat{\mathbf{w}}) P(\mathbf{w} \mid \mathbf{X})$$

### Why MBR Diverges from MAP:
Suppose the posterior distribution over 3 hypotheses is:
- $\mathbf{w}_1 = \text{"کٲشُر زَبان"}$, $P(\mathbf{w}_1 \mid \mathbf{X}) = 0.45$
- $\mathbf{w}_2 = \text{"کٲشُر زَبانہٕ"}$, $P(\mathbf{w}_2 \mid \mathbf{X}) = 0.35$
- $\mathbf{w}_3 = \text{"ہِندوستان"}$, $P(\mathbf{w}_3 \mid \mathbf{X}) = 0.20$

Under MAP decoding:
$$\hat{\mathbf{w}}_{\text{MAP}} = \mathbf{w}_1 \quad (\text{since } 0.45 > 0.35 > 0.20)$$

However, notice that $\mathbf{w}_1$ and $\mathbf{w}_2$ share the word `"کٲشُر"` and differ only by a single suffix vowel. If $\mathbf{w}_2$ is the truth, predicting $\mathbf{w}_1$ incurs an edit distance of only 1 word, whereas predicting $\mathbf{w}_3$ incurs an edit distance of 2 words. MBR explicitly minimizes word-level editing risk rather than winner-take-all sentence probability.

---

## 4. Length Bias Derivation in Log-Linear Shallow Fusion

In character-level decoding, a word hypothesis $\mathbf{w}$ spanning $L$ characters requires $L$ distinct character transitions.
Because each frame emission probability $P(c_t \mid \mathbf{X}) < 1.0$, log-probabilities are strictly negative:
$$\log P(c_t \mid \mathbf{X}) < 0$$

Summing negative log-probabilities across time frames causes longer hypotheses to accumulate larger negative penalties:
$$\sum_{t=1}^{T_{\text{long}}} \log P(c_t \mid \mathbf{X}) \ll \sum_{t=1}^{T_{\text{short}}} \log P(c_t \mid \mathbf{X})$$

Without length compensation, the decoder exhibits **brevity bias**: it systematically prefers deleting words or emitting shorter strings.
By introducing the linear word insertion bonus $\beta |\mathbf{w}|$ (where $\beta > 0$):
$$\mathcal{S}(\mathbf{w}) = \log P_{\text{CTC}}(\mathbf{w} \mid \mathbf{X}) + \alpha \log P_{\text{LM}}(\mathbf{w}) + \beta |\mathbf{w}|$$
each completed word receives a positive credit $\beta$, restoring an unbiased trade-off between insertion and deletion errors.
