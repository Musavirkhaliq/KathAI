# Mathematical Foundations: Gumbel-Softmax Quantization and Contrastive Bounds

---

## 1. Gumbel-Softmax Differentiable Discretization

Directly sampling a discrete codebook index $i = \arg\max_k (l_k)$ is non-differentiable ($\frac{\partial i}{\partial l_k} = 0$).

The **Gumbel-Softmax** trick introduces differentiable sampling by adding standard Gumbel noise $g_k \sim \text{Gumbel}(0, 1)$ to logits $l_k$:

$$g_k = -\log(-\log u_k), \quad u_k \sim \mathcal{U}(0, 1)$$

The continuous relaxation vector $\mathbf{p} \in \Delta^{V-1}$ is computed via annealed Softmax:

$$p_k = \frac{\exp((l_k + g_k) / \tau)}{\sum_{j=1}^V \exp((l_j + g_j) / \tau)}$$

- As temperature $\tau \to \infty$, $\mathbf{p}$ approaches a uniform distribution.
- As temperature $\tau \to 0$, $\mathbf{p}$ converges to a one-hot categorical sample, while remaining fully differentiable during backpropagation.

---

## 2. Codebook Diversity Loss ($\mathcal{L}_d$)

To prevent the codebook from collapsing into a small subset of frequently used entries, a diversity loss maximizes the entropy of the average softmax distribution across a batch of $T$ frames:

$$\bar{p}_g = \frac{1}{T} \sum_{t=1}^T p_{t, g}$$
$$\mathcal{L}_d = \frac{1}{G} \sum_{g=1}^G \left( \log V - H(\bar{p}_g) \right) = \frac{1}{G} \sum_{g=1}^G \left( \log V + \sum_{k=1}^V \bar{p}_{g, k} \log \bar{p}_{g, k} \right)$$

where $G$ is the number of codebook groups and $V$ is the number of entries per group.
