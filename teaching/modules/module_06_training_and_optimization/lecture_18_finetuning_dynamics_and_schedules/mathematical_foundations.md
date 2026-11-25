# Mathematical Foundations: Layer-Wise Decay and Cosine Dynamics

---

## 1. LLRD Parameter Grouping Formulation

Let the network parameters be partitioned into $L+1$ parameter groups $\mathcal{G}_0, \mathcal{G}_1, \dots, \mathcal{G}_L$:
- $\mathcal{G}_0$: Linear CTC projection head.
- $\mathcal{G}_l$: Parameters of Transformer layer $l$ ($l \in \{1, \dots, L\}$).

The optimizer update rule for parameter $\mathbf{w} \in \mathcal{G}_l$ under AdamW is:

$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t$$
$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta_l(t) \left( \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} + \lambda \mathbf{w}_t \right)$$

where $\eta_l(t) = \eta(t) \cdot \xi^{L - l}$ and $\lambda$ is the decoupled weight decay factor ($0.01$).
