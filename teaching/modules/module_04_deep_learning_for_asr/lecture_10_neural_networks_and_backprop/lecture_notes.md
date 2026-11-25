# Lecture 10: Deep Neural Networks and Temporal Backpropagation

---

## 1. The Fundamental Architecture of Deep Neural Acoustic Models

In modern speech recognition, deep neural networks parameterize non-linear vector functions mapping acoustic sequences to linguistic representations. A deep feedforward layer applies an affine transformation followed by an element-wise non-linear activation:

$$\mathbf{h}^{(l)} = \sigma\left( \mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)} \right)$$

where $\mathbf{W}^{(l)} \in \mathbb{R}^{d_l \times d_{l-1}}$ is the weight matrix, $\mathbf{b}^{(l)} \in \mathbb{R}^{d_l}$ is the bias vector, and $\sigma(\cdot)$ is an activation function (GELU, ReLU, or Swish).

In speech models like XLS-R, the standard activation is the **Gaussian Error Linear Unit (GELU)**:

$$\text{GELU}(x) = x \Phi(x) = x P(X \le x) \approx 0.5 x \left( 1 + \tanh\left( \sqrt{\frac{2}{\pi}} \left( x + 0.044715 x^3 \right) \right) \right)$$

GELU introduces smooth non-monotonicity and probabilistic gating, outperforming hard ReLU in deep acoustic transformers.

---

## 2. Temporal Backpropagation and Jacobian-Vector Products

Training deep models over sequences of length $T$ involves computing the gradient of a scalar sequence loss $\mathcal{L}$ with respect to all parameter matrices $\mathbf{W}^{(l)}$.

By the multivariate chain rule, the gradient at layer $l$ satisfies:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(l-1)}} = \left( \frac{\partial \mathbf{h}^{(l)}}{\partial \mathbf{h}^{(l-1)}} \right)^T \frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(l)}} = \left( \mathbf{W}^{(l)} \right)^T \left( \frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(l)}} \odot \sigma'(\mathbf{z}^{(l)}) \right)$$

where $\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}$, and $\odot$ represents the Hadamard (element-wise) product.

For a sequence of $L$ layers:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(0)}} = \left[ \prod_{l=1}^L \left( \mathbf{W}^{(l)} \right)^T \text{diag}(\sigma'(\mathbf{z}^{(l)})) \right] \frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(L)}}$$

### The Vanishing / Exploding Gradient Dilemma:
Let $\lambda_{\max}$ be the maximum singular value of the weight matrices.
- If $\lambda_{\max} < 1$, gradients decay exponentially with depth $L$: $\|\frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(0)}}\| \to 0$ (**Vanishing Gradient**).
- If $\lambda_{\max} > 1$, gradients grow exponentially: $\|\frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(0)}}\| \to \infty$ (**Exploding Gradient**).

---

## 3. Normalization Techniques: LayerNorm vs. BatchNorm in Speech

Speech sequences exhibit variable lengths within every batch (e.g., 2.5s to 15.0s).
- **Batch Normalization (BatchNorm)**: Computes mean and variance across the batch dimension. In speech, because sequences are zero-padded to batch max length, padding frames severely distort running mean $\mu_B$ and running variance $\sigma_B^2$, leading to training instability.
- **Layer Normalization (LayerNorm)**: Computes mean and variance independently for each time frame across the feature channel dimension $D$:

$$\mu_t = \frac{1}{D} \sum_{i=1}^D x_{t, i}, \quad \sigma_t^2 = \frac{1}{D} \sum_{i=1}^D (x_{t, i} - \mu_t)^2$$
$$\hat{\mathbf{x}}_t = \frac{\mathbf{x}_t - \mu_t}{\sqrt{\sigma_t^2 + \epsilon}} \odot \boldsymbol{\gamma} + \boldsymbol{\beta}$$

  LayerNorm is completely independent of batch size and padding tokens, making it the universal standard for speech transformers.
