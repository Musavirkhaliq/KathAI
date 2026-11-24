# Lecture 18: Fine-Tuning Dynamics, Learning Rate Schedules, and Layer-Wise Decay

---

## 1. The Delicate Dynamics of Pretrained Speech Adaptation

When fine-tuning a 317M parameter self-supervised checkpoint like XLS-R on low-resource Kashmiri speech, the optimization problem is fundamentally different from training from scratch:
- The network already possesses powerful acoustic representations learned over 436,000 hours of speech.
- Applying a uniform learning rate $\eta$ across all layers creates an optimization pathology: high learning rates destroy pretrained representations in bottom layers (catastrophic forgetting), while low learning rates prevent the randomly initialized linear projection head from learning Kashmiri character mappings.

To resolve this tension, EduCasheer applies two foundational scheduling principles:
1. **Layer-wise Learning Rate Decay (LLRD)**
2. **Linear Warmup with Cosine Annealing**

---

## 2. Layer-wise Learning Rate Decay (LLRD)

In a 24-layer Transformer, lower layers capture low-level acoustic physics (formants, pitch periods, spectral envelopes), while upper layers capture high-level phonotactics and language-specific representations.

LLRD scales the learning rate exponentially across layers using decay factor $\xi \in (0, 1)$ (typically $\xi = 0.75$):

$$\eta_l = \eta_{\text{base}} \cdot \xi^{L - l}$$

where $L = 24$ is the top layer index, and $l \in \{1, 2, \dots, L\}$ is the layer index.

### Concrete Learning Rate Profile ($\eta_{\text{base}} = 3 \times 10^{-4}, \xi = 0.75$):
- **Classification Head**: $\eta_{\text{head}} = 3.0 \times 10^{-4}$
- **Layer 24 (Top Transformer Layer)**: $\eta_{24} = 3.0 \times 10^{-4} \times (0.75)^0 = 3.0 \times 10^{-4}$
- **Layer 23**: $\eta_{23} = 3.0 \times 10^{-4} \times (0.75)^1 = 2.25 \times 10^{-4}$
- **Layer 12 (Middle Layer)**: $\eta_{12} = 3.0 \times 10^{-4} \times (0.75)^{12} \approx 9.5 \times 10^{-6}$
- **Layer 1 (Bottom Layer)**: $\eta_1 = 3.0 \times 10^{-4} \times (0.75)^{23} \approx 3.0 \times 10^{-7}$
- **Feature Extractor (CNN)**: $\eta_{\text{cnn}} = 0.0$ (strictly frozen).

This gradient distribution allows the top layers to adapt rapidly to Kashmiri orthography while protecting bottom-layer acoustic primitives from degradation.

---

## 3. Warmup + Cosine Annealing Learning Rate Schedule

To prevent catastrophic gradient shocks during initial steps, the learning rate follows a two-phase schedule over total steps $S$:

### Phase 1: Linear Warmup ($s \le S_{\text{warmup}}$):
$$\eta(s) = \eta_{\max} \cdot \frac{s}{S_{\text{warmup}}}$$
Typically, $S_{\text{warmup}} = 0.10 \times S$ (first 10% of training steps).

### Phase 2: Cosine Annealing ($S_{\text{warmup}} < s \le S$):
$$\eta(s) = \eta_{\min} + \frac{1}{2} (\eta_{\max} - \eta_{\min}) \left( 1 + \cos\left( \pi \frac{s - S_{\text{warmup}}}{S - S_{\text{warmup}}} \right) \right)$$
where $\eta_{\min} \approx 0.01 \times \eta_{\max}$.

Cosine decay provides smooth deceleration without sharp step drops, allowing weights to settle gently into flat, generalizable minima.
