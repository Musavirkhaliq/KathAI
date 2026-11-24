# Lecture 23: Acoustic Temperature Calibration and Shallow Fusion Decoding

---

## 1. The Log-Linear Shallow Fusion Framework

During inference, decoding searches for the hypothesis $\hat{\mathbf{w}}$ maximizing the composite log-linear shallow fusion score:

$$\hat{\mathbf{w}} = \arg\max_{\mathbf{w}} \left( \log P_{\text{CTC}}(\mathbf{w} \mid \mathbf{X}, T) + \alpha \log P_{\text{LM}}(\mathbf{w}) + \beta |\mathbf{w}| \right)$$

This objective balances three distinct mathematical forces:
1. **$\log P_{\text{CTC}}(\mathbf{w} \mid \mathbf{X}, T)$**: Acoustic evidence conditioned on calibrated temperature $T$.
2. **$\alpha \log P_{\text{LM}}(\mathbf{w})$**: Syntactic prior weighted by language model weight $\alpha \in [0.1, 1.0]$.
3. **$\beta |\mathbf{w}|$**: Word insertion penalty compensating for character-level brevity bias ($\beta \in [0.5, 2.5]$).

---

## 2. Acoustic Temperature Calibration ($T = 1.08$)

In deep neural networks trained with cross-entropy or CTC loss, output distributions suffer from **overconfidence**:
- Logits are excessively large, causing Softmax probabilities to concentrate into near $1.0$ or $0.0$ values.
- In beam search, overconfident CTC probabilities prematurely prune valid linguistic branches before the language model has an opportunity to evaluate their syntactic validity.

### Temperature Scaling Formulation:
Logits $\mathbf{z}_t$ are calibrated by scalar temperature parameter $T > 1.0$:

$$y_k^t(T) = \frac{\exp(z_k^t / T)}{\sum_{j=1}^{|\Omega'|} \exp(z_j^t / T)}$$

- In Stage 23 of EduCasheer, temperature was optimized via negative log-likelihood (NLL) minimization on the disjoint validation split, yielding optimal **$T = 1.08$**.
- Setting $T = 1.08$ softens emission distributions, preserving candidate diversity in the beam search lattice and unlocking optimal shallow fusion synergy.

---

## 3. Grid Search Optimization of $\alpha$ and $\beta$

The parameters $\alpha$ (LM weight) and $\beta$ (Word insertion penalty) cannot be tuned on the test set without severe contamination. They must be calibrated via a **2D Grid Search on the held-out validation set**:

```
Alpha Sweep: [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
Beta Sweep:  [0.50, 0.80, 1.00, 1.20, 1.50, 2.00]
```

### The Two Historic Operating Regimes in the Repository:
1. **Configuration A (Balanced)**: $\alpha = 0.30, \beta = 1.00, T = 1.08, \text{beam} = 64$.
   - Yields balanced performance across clean and conversational audio.
2. **Configuration B (Aggressive LM)**: $\alpha = 0.50, \beta = 1.50, T = 1.08, \text{beam} = 100$.
   - Optimizes formal broadcast news and Quranic commentary, but slightly over-corrects colloquial names in spontaneous dialects.
   - Used by the certified `E22-01` benchmark to achieve **24.12% disjoint WER**.
