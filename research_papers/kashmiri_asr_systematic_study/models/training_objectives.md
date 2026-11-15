# Training Objectives and Loss Formulations

This document provides the mathematical formulations and operational mechanics for the loss functions implemented throughout the research project.

---

## 1. Connectionist Temporal Classification (CTC) Loss

Used in Stages 8, 9, 11, 12, 13, and as the base objective in Stage 15:

$$\mathcal{L}_{\text{CTC}}(\theta) = -\ln P(y | x; \theta) = -\ln \sum_{\pi \in \mathcal{B}^{-1}(y)} \prod_{t=1}^T P(\pi_t | x; \theta)$$

where:
- $x = (x_1, \dots, x_T)$ represents the sequence of $T$ acoustic feature frames emitted by the Transformer encoder.
- $y = (y_1, \dots, y_U)$ is the reference sequence of $U$ target characters ($U \le T$).
- $\pi = (\pi_1, \dots, \pi_T)$ is a valid CTC path over the vocabulary $\mathcal{V} \cup \{\epsilon\}$ (where $\epsilon$ is the CTC blank token).
- $\mathcal{B}$ is the collapse mapping that removes sequential duplicate characters and blanks (e.g., $\mathcal{B}(\text{a } \epsilon \text{ a a}) = \text{a a}$).

Gradients with respect to frame logits are computed using the forward-backward algorithm in $O(T \cdot U)$ time.

---

## 2. Minimum Word Error Rate (MWER) Sequence Loss

Implemented in Stage 14 and Stage 15 to directly optimize expected edit distance:

$$\mathcal{L}_{\text{MWER}}(\theta) = \sum_{y \in \mathcal{H}} \hat{P}(y | x) \left[ \mathcal{W}(y, y^*) - \hat{\mathbb{E}}_{y' \in \mathcal{H}}[\mathcal{W}(y', y^*)] \right]$$

where:
- $\mathcal{H} = \{y^{(1)}, \dots, y^{(N)}\}$ is the $N$-best hypothesis beam extracted via beam search decoding.
- $\mathcal{W}(y, y^*)$ is the word-level Levenshtein edit distance between candidate $y$ and reference transcript $y^*$.
- $\hat{P}(y | x)$ is the normalized posterior probability over the $N$-best beam:
  $$\hat{P}(y | x) = \frac{\exp(P_{\text{CTC}}(y | x) / \tau)}{\sum_{y' \in \mathcal{H}} \exp(P_{\text{CTC}}(y' | x) / \tau)}$$
  with temperature parameter $\tau = 1.0$.
- The sequence objective is combined with the standard CTC loss to ensure gradient stability:
  $$\mathcal{L}_{\text{Combined}} = (1 - \lambda) \mathcal{L}_{\text{CTC}} + \lambda \mathcal{L}_{\text{MWER}}$$
  with $\lambda \in \{0.00, 0.05, 0.10, 0.20\}$ in Stage 15.

---

## 3. Consistency-Regularized CTC (CR-CTC)

Implemented in Stage 13 (`CR-01`):

$$\mathcal{L}_{\text{CR-CTC}} = \mathcal{L}_{\text{CTC}}(x, y) + \gamma D_{\text{KL}}\left( P(\cdot | x; \theta) \parallel P(\cdot | \tilde{x}; \theta) \right)$$

where $\tilde{x}$ is a stochastically perturbed forward pass obtained by enabling dropout ($p=0.15$) and temporal masking (SpecAugment), and $D_{\text{KL}}$ penalizes divergences in frame-level posterior distributions.

---

## 4. Intermediate CTC Loss (Inter-CTC)

Implemented in Stage 13 (`I-01`):

$$\mathcal{L}_{\text{Inter-CTC}} = (1 - \alpha) \mathcal{L}_{\text{CTC}}^{(24)}(y | x) + \alpha \mathcal{L}_{\text{CTC}}^{(12)}(y | x)$$

An auxiliary linear prediction head was attached to layer 12 of the 24-layer Transformer stack, injecting direct character-level gradients into the intermediate representations with $\alpha = 0.3$.
