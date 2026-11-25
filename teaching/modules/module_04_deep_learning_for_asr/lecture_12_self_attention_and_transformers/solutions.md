# Solutions: Lecture 12 (Self-Attention and Transformers)

---

## Solution to Exercise 1
1. Number of frames $T$:

$$T = \frac{15 \times 16000}{320} = \frac{240000}{320} = \mathbf{750 \text{ frames}}$$

2. Matrix dimensions:
   `[Batch, Heads, T, T]` = `[8, 16, 750, 750]`.
3. Elements per layer:

$$N_{\text{elem}} = 8 \times 16 \times 750 \times 750 = 128 \times 562,500 = 72,000,000 \text{ floats}$$

   Memory per layer in FP32 (4 bytes per float):

$$M_{\text{layer}} = 72,000,000 \times 4 \text{ bytes} = 288,000,000 \text{ bytes} \approx \mathbf{288 \text{ MB}}$$

   For 24 layers:

$$M_{\text{total}} = 24 \times 288 \text{ MB} = \mathbf{6,912 \text{ MB}} \approx \mathbf{6.91 \text{ GB}}$$

---

## Solution to Exercise 2
In Softmax:

$$P_j = \frac{e^{z_j}}{\sum_{k} e^{z_k}}$$

If $z_j = -10^9$, $e^{-10^9} \to 0$ in IEEE 754 floating point. Thus $P_j = 0$.
The gradient of Softmax is:

$$\frac{\partial P_i}{\partial z_j} = P_i (\delta_{ij} - P_j)$$

Since $P_j = 0$, for all $i \neq j$ the term is 0, and for $i = j$ it is $0 \times (1 - 0) = 0$.
Hence no gradient updates flow to masked padding positions. $\blacksquare$
