# Solutions: Lecture 20 (Numerical Stability and OOM Recovery)

---

## Solution to Exercise 1
Formula: $M = B \times H \times T^2 \times L \times 4 \text{ bytes}$.
1. For $T = 250$:

$$M_{250} = 8 \times 16 \times 250^2 \times 24 \times 4 = 128 \times 62,500 \times 96 = 768,000,000 \text{ bytes} \approx \mathbf{0.768 \text{ GB}}$$

2. For $T = 1000$:

$$M_{1000} = 8 \times 16 \times 1000^2 \times 24 \times 4 = 128 \times 1,000,000 \times 96 = 12,288,000,000 \text{ bytes} \approx \mathbf{12.288 \text{ GB}}$$

3. Ratio:

$$\frac{M_{1000}}{M_{250}} = \left( \frac{1000}{250} \right)^2 = 4^2 = \mathbf{16\times}$$

---

## Solution to Exercise 2
1. If $\mathbf{g}_t$ contains NaN, the Adam momentum update computes:

$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \text{NaN} = \text{NaN}$$

2. The weight update computes:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \frac{\text{NaN}}{\sqrt{\text{NaN}} + \epsilon} = \text{NaN}$$

3. Once a weight matrix becomes NaN, all subsequent forward passes produce NaN outputs ($\mathbf{x} \mathbf{W} = \text{NaN}$), permanently destroying the model.
