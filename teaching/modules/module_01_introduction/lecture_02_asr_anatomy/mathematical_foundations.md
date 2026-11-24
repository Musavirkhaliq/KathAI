# Mathematical Foundations: Pipeline Anatomy and Decoding Formulation

---

## 1. The CTC Alignment Collapse Operator $\mathcal{B}$

Let $\Omega$ be the alphabet of target linguistic units (characters), and let $\Omega' = \Omega \cup \{\epsilon\}$, where $\epsilon$ is the non-emitting blank symbol.

Given an input acoustic sequence $\mathbf{X} = (\mathbf{x}_1, \dots, \mathbf{x}_T)$, a CTC alignment $\boldsymbol{\pi} = (\pi_1, \dots, \pi_T) \in (\Omega')^T$ is a sequence of length $T$.

The deterministic collapse mapping $\mathcal{B}: (\Omega')^T \to \Omega^{\le T}$ is defined by two sequential operations:
1. **Collapsing consecutive identical tokens**:
   $$\mathcal{B}_{\text{dedup}}(\pi_1, \dots, \pi_T) = (\pi_{t_1}, \pi_{t_2}, \dots) \quad \text{where } \pi_{t_k} \neq \pi_{t_k-1}$$
2. **Removing all blank tokens $\epsilon$**:
   $$\mathcal{B}(\boldsymbol{\pi}) = [c \in \mathcal{B}_{\text{dedup}}(\boldsymbol{\pi}) \mid c \neq \epsilon]$$

### Example:
Let $\boldsymbol{\pi} = (\epsilon, \text{ك}, \text{ك}, \epsilon, \text{ٲ}, \text{ٲ}, \text{ش}, \epsilon, \text{ش}, \text{ر})$.
1. Consecutive deduplication: $(\epsilon, \text{ك}, \epsilon, \text{ٲ}, \text{ش}, \epsilon, \text{ش}, \text{ر})$
2. Blank removal: $(\text{ك}, \text{ٲ}, \text{ش}, \text{ش}, \text{ر}) \to \text{کٲشُر}$ (with double /ʃ/).
Note that the intermediate blank $\epsilon$ between the two $\text{ش}$ characters is required to emit adjacent identical letters.

---

## 2. Conditional Independence and Sequence Probability

CTC models frame-level emission probabilities conditionally independent given the input sequence $\mathbf{X}$:

$$P(\boldsymbol{\pi} \mid \mathbf{X}) = \prod_{t=1}^T P(\pi_t \mid \mathbf{X}) = \prod_{t=1}^T y_{\pi_t}^t$$

where $y_k^t$ is the softmax probability of emitting symbol $k$ at frame $t$:

$$y_k^t = \frac{\exp(z_k^t)}{\sum_{j \in \Omega'} \exp(z_j^t)}$$

The total marginal likelihood of a target transcription $\mathbf{y} = (y_1, \dots, y_U)$ is the sum over all valid alignments in the preimage $\mathcal{B}^{-1}(\mathbf{y})$:

$$P(\mathbf{y} \mid \mathbf{X}) = \sum_{\boldsymbol{\pi} \in \mathcal{B}^{-1}(\mathbf{y})} P(\boldsymbol{\pi} \mid \mathbf{X})$$

---

## 3. Log-Linear Shallow Fusion Objective

During inference, decoding searches for the word sequence $\hat{\mathbf{w}}$ that maximizes a log-linear combination of acoustic and linguistic models:

$$\hat{\mathbf{w}} = \arg\max_{\mathbf{w}} \left\{ \log P_\text{CTC}(\mathbf{w} \mid \mathbf{X}) + \alpha \log P_\text{LM}(\mathbf{w}) + \beta |\mathbf{w}| + \gamma \log P_\text{length}(\mathbf{w}) \right\}$$

Under Bayes' Theorem, if $P(\mathbf{w} \mid \mathbf{X}) \propto P(\mathbf{X} \mid \mathbf{w}) P(\mathbf{w})$, taking the logarithm gives:
$$\log P(\mathbf{w} \mid \mathbf{X}) = \log P(\mathbf{X} \mid \mathbf{w}) + \log P(\mathbf{w}) - \log P(\mathbf{X})$$

In CTC, the model estimates posterior $P_\text{CTC}(\mathbf{w} \mid \mathbf{X})$ rather than generative likelihood $P(\mathbf{X} \mid \mathbf{w})$. Hence, the language model $P_\text{LM}(\mathbf{w})$ is introduced via **shallow fusion**, weighted by scalar hyperparameter $\alpha \in [0, 1]$.
