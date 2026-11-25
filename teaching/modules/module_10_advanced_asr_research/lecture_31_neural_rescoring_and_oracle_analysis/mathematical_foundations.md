# Mathematical Foundations: Oracle Bounds and Rescoring Interpolation

---

## 1. Mathematical Upper and Lower Bounds of Beam Search

For any hypothesis list $\mathcal{H}_W$ generated with beam width $W$:

$$\mathcal{W}(\mathbf{w}_{\text{oracle}}, \mathbf{y}^*) \le \mathcal{W}(\mathbf{w}_{\text{1-best}}, \mathbf{y}^*) \le \mathcal{W}(\mathbf{w}_{\text{worst}}, \mathbf{y}^*)$$

As beam width $W \to \infty$:

$$\lim_{W \to \infty} \text{WER}_{\text{oracle}}(W) = \text{WER}_{\text{lattice-oracle}}$$

---

## 2. Linear Model Interpolation and Cross-Entropy Optimization

The weights $\boldsymbol{\lambda} = (\alpha_1, \alpha_2, \beta)$ are optimized on the validation set using Minimum Error Rate Training (MERT) or Rank-Loss optimization:

$$\mathcal{L}_{\text{rank}}(\boldsymbol{\lambda}) = \sum_{i=1}^M \max\left( 0, 1 - \mathcal{S}_{\boldsymbol{\lambda}}(\mathbf{w}_{\text{best}}^{(i)}) + \mathcal{S}_{\boldsymbol{\lambda}}(\mathbf{w}_{\text{other}}^{(i)}) \right)$$
