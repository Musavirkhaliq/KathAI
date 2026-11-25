# Solutions: Lecture 17 (The Complete PyTorch CTC Training Loop)

---

## Solution to Exercise 1
Let $\mathcal{L}_k = \frac{1}{|\mathcal{B}_k|} \sum_{i \in \mathcal{B}_k} \ell(x_i, y_i)$.
If all micro-batches have equal size $B$, then the total loss across the combined batch is:

$$\mathcal{L}_{\text{total}} = \frac{1}{K \cdot B} \sum_{k=1}^K \sum_{i \in \mathcal{B}_k} \ell(x_i, y_i) = \frac{1}{K} \sum_{k=1}^K \left( \frac{1}{B} \sum_{i \in \mathcal{B}_k} \ell(x_i, y_i) \right) = \frac{1}{K} \sum_{k=1}^K \mathcal{L}_k$$

By linearity of differentiation:

$$\nabla_{\mathbf{W}} \mathcal{L}_{\text{total}} = \frac{1}{K} \sum_{k=1}^K \nabla_{\mathbf{W}} \mathcal{L}_k$$

Accumulating gradients computes precisely this summation. $\blacksquare$

---

## Solution to Exercise 2
1. Effective batch size:

$$B_{\text{eff}} = 2 \text{ GPUs} \times 8 \text{ samples/GPU} \times 4 \text{ accum steps} = \mathbf{64 \text{ utterances}}$$

2. Optimizer updates per epoch:

$$N_{\text{steps}} = \frac{32,000}{64} = \mathbf{500 \text{ updates per epoch}}$$
