# Mathematical Foundations: Gradient Clipping and Accumulation

---

## 1. Gradient Norm Clipping Invariant

Let $\mathbf{g} = \nabla_{\mathbf{W}} \mathcal{L}$ be the concatenated gradient vector across all model parameters.
Given maximum norm threshold $M$ (typically $M = 1.0$):

$$\|\mathbf{g}\|_2 = \sqrt{\sum_{i} g_i^2}$$

The clipped gradient $\tilde{\mathbf{g}}$ is:

$$\tilde{\mathbf{g}} = \begin{cases}

\mathbf{g} & \text{if } \|\mathbf{g}\|_2 \le M \\
\frac{M}{\|\mathbf{g}\|_2} \mathbf{g} & \text{if } \|\mathbf{g}\|_2 > M
\end{cases}

$$

This guarantees:

$$\|\tilde{\mathbf{g}}\|_2 \le M$$

while preserving the exact directional angle of the gradient vector:

$$\frac{\tilde{\mathbf{g}}}{\|\tilde{\mathbf{g}}\|} = \frac{\mathbf{g}}{\|\mathbf{g}\|}$$
