# Mathematical Foundations: Decision Theory, Bayesian Formulation & Sequence Bounds

---

## 1. Statistical Decision Theory and the Zero-One Loss

Let $\mathcal{X}$ denote the space of all possible acoustic observation sequences:
$$\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T] \in \mathcal{X}, \quad \mathbf{x}_t \in \mathbb{R}^D$$
where $T \in \mathbb{N}$ is the variable number of acoustic frames, and $D$ is the feature dimension.

Let $\mathcal{W}$ denote the discrete countable set of all valid linguistic word sequences over vocabulary $\mathcal{V}$:
$$\mathbf{w} = (w_1, w_2, \dots, w_U) \in \mathcal{W}, \quad w_u \in \mathcal{V}, \; U \in \mathbb{N}$$

A speech recognizer is a decision rule (hypothesis function) that maps an acoustic observation to a predicted word sequence:
$$\delta: \mathcal{X} \longrightarrow \mathcal{W}$$

### Expected Risk Formulation
Let $L(\mathbf{w}, \delta(\mathbf{X})) \ge 0$ be a loss function quantifying the penalty of predicting $\delta(\mathbf{X})$ when the true intended sequence was $\mathbf{w}$.
The conditional Bayes risk of decision rule $\delta$ given observation $\mathbf{X}$ is:
$$R(\delta \mid \mathbf{X}) = \mathbb{E}_{\mathbf{w} \sim P(\mathbf{w} \mid \mathbf{X})} \left[ L(\mathbf{w}, \delta(\mathbf{X})) \right] = \sum_{\mathbf{w} \in \mathcal{W}} L(\mathbf{w}, \delta(\mathbf{X})) P(\mathbf{w} \mid \mathbf{X})$$

The overall Bayes risk across the joint data distribution $P(\mathbf{X}, \mathbf{w})$ is:
$$R(\delta) = \mathbb{E}_{(\mathbf{X}, \mathbf{w})} \left[ L(\mathbf{w}, \delta(\mathbf{X})) \right] = \int_{\mathcal{X}} R(\delta \mid \mathbf{X}) P(\mathbf{X}) d\mathbf{X}$$

An optimal decision rule $\delta^*$ minimizes the expected risk:
$$\delta^* = \arg\min_\delta R(\delta)$$

### Theorem 1: Optimality of the MAP Decision Rule under Zero-One Loss
**Theorem**: If the loss function is the zero-one sequence loss:
$$L_{0-1}(\mathbf{w}, \hat{\mathbf{w}}) = \mathbb{I}(\mathbf{w} \ne \hat{\mathbf{w}}) = 1 - \delta_{\mathbf{w}, \hat{\mathbf{w}}}$$
where $\delta_{\mathbf{w}, \hat{\mathbf{w}}}$ is the Kronecker delta, then the optimal Bayes decision rule is strictly the Maximum A Posteriori (MAP) classifier:
$$\delta^*(\mathbf{X}) = \arg\max_{\mathbf{w} \in \mathcal{W}} P(\mathbf{w} \mid \mathbf{X})$$

**Proof**:
Substitute $L_{0-1}$ into the conditional risk expression:
$$R(\hat{\mathbf{w}} \mid \mathbf{X}) = \sum_{\mathbf{w} \in \mathcal{W}} \mathbb{I}(\mathbf{w} \ne \hat{\mathbf{w}}) P(\mathbf{w} \mid \mathbf{X})$$
$$= \sum_{\mathbf{w} \ne \hat{\mathbf{w}}} P(\mathbf{w} \mid \mathbf{X})$$
$$= \sum_{\mathbf{w} \in \mathcal{W}} P(\mathbf{w} \mid \mathbf{X}) - P(\hat{\mathbf{w}} \mid \mathbf{X})$$
Since probabilities over $\mathcal{W}$ sum to $1$ ($\sum_{\mathbf{w} \in \mathcal{W}} P(\mathbf{w} \mid \mathbf{X}) = 1$):
$$R(\hat{\mathbf{w}} \mid \mathbf{X}) = 1 - P(\hat{\mathbf{w}} \mid \mathbf{X})$$

To minimize $R(\hat{\mathbf{w}} \mid \mathbf{X})$, we must choose $\hat{\mathbf{w}}$ to maximize the subtracted term:
$$\hat{\mathbf{w}}^* = \arg\min_{\hat{\mathbf{w}} \in \mathcal{W}} \left[ 1 - P(\hat{\mathbf{w}} \mid \mathbf{X}) \right] = \arg\max_{\hat{\mathbf{w}} \in \mathcal{W}} P(\hat{\mathbf{w}} \mid \mathbf{X})$$
Since $P(\mathbf{X}) > 0$ everywhere, minimizing conditional risk at every $\mathbf{X}$ minimizes total risk $R(\delta)$. $\blacksquare$

---

## 2. Derivation of the Decoupled Bayesian Objective

By Bayes' Theorem:
$$P(\mathbf{w} \mid \mathbf{X}) = \frac{P(\mathbf{X} \mid \mathbf{w}) P(\mathbf{w})}{P(\mathbf{X})}$$

Where:
- $P(\mathbf{X} \mid \mathbf{w})$ is the marginal likelihood (Acoustic Model).
- $P(\mathbf{w})$ is the prior probability (Language Model).
- $P(\mathbf{X}) = \sum_{\mathbf{w}' \in \mathcal{W}} P(\mathbf{X} \mid \mathbf{w}') P(\mathbf{w}')$ is the evidence.

Substituting into the MAP rule:
$$\hat{\mathbf{w}} = \arg\max_{\mathbf{w} \in \mathcal{W}} \frac{P(\mathbf{X} \mid \mathbf{w}) P(\mathbf{w})}{P(\mathbf{X})}$$

Because $P(\mathbf{X})$ is strictly invariant with respect to $\mathbf{w}$:
$$\hat{\mathbf{w}} = \arg\max_{\mathbf{w} \in \mathcal{W}} P(\mathbf{X} \mid \mathbf{w}) P(\mathbf{w})$$

Applying the natural logarithm (a monotonic strictly increasing transformation on $\mathbb{R}^+$):
$$\hat{\mathbf{w}} = \arg\max_{\mathbf{w} \in \mathcal{W}} \left[ \ln P(\mathbf{X} \mid \mathbf{w}) + \ln P(\mathbf{w}) \right]$$

---

## 3. Generalized Shallow Fusion and Acoustic Weight Scaling

In practical modern neural systems, the acoustic model does not output a true generative density $P(\mathbf{X} \mid \mathbf{w})$; instead, a deep neural network parameterized by $\theta$ predicts discriminative posterior probabilities or CTC sequence probabilities $P_\theta(\mathbf{w} \mid \mathbf{X})$.

Furthermore, two empirical phenomena distort pure Bayesian addition:
1. **Acoustic Overconfidence**: Neural networks trained on cross-entropy or CTC output peaked posterior distributions where acoustic log-probabilities have a dynamic range vastly exceeding that of statistical $n$-gram language models.
2. **Length Bias**: Because the language model probability $P(\mathbf{w}) = \prod_{u=1}^U P(w_u \mid w_{<u})$ is a product of terms strictly less than $1.0$, $\ln P(\mathbf{w}) < 0$. Each additional word penalizes the hypothesis score, causing beam search to systematically favor unrealistically short transcriptions (deleting short words).

To restore calibration, speech scientists introduce two hyperparameters:
- $\alpha \in \mathbb{R}^+$: Language model interpolation weight.
- $\beta \in \mathbb{R}$: Word insertion bonus (or character length bonus).

The generalized decoding objective becomes:
$$\mathcal{S}(\mathbf{w} \mid \mathbf{X}) = \ln P_\theta(\mathbf{w} \mid \mathbf{X}) + \alpha \ln P_\text{LM}(\mathbf{w}) + \beta |\mathbf{w}|$$
where $|\mathbf{w}| = U$ is the number of words in candidate sequence $\mathbf{w}$.

### Gradient Interpretation of $\alpha$ and $\beta$
During beam search hypothesis expansion, extending a prefix by word $w_{u}$ changes the total score by:
$$\Delta \mathcal{S} = \underbrace{\Delta \ln P_\text{AM}(w_u)}_{\text{Acoustic Evidence}} + \alpha \underbrace{\ln P_\text{LM}(w_u \mid w_{<u})}_{\text{Linguistic Plausibility (} \le 0 \text{)}} + \underbrace{\beta}_{\text{Length Compensation}}$$

- If $\beta = 0$, every extension carries a negative LM penalty $\alpha \ln P_\text{LM} < 0$. If an acoustic sound is quiet or ambiguous, the decoder maximizes score by terminating early.
- When $\beta > 0$, it compensates for the average word perplexity:
  $$\beta \approx -\alpha \mathbb{E}_{w} [\ln P_\text{LM}(w \mid \text{context})]$$
  neutralizing the length penalty.
