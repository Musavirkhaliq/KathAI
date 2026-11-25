# Mathematical Foundations: Convergence Bounds and Checkpoint Hashing

---

## 1. Convergence Asymptotics under Cosine Decay

Let loss function $\mathcal{L}(\theta)$ be $L$-smooth and satisfy the Polyak-Łojasiewicz (PŁ) inequality with constant $\mu > 0$:

$$\frac{1}{2} \|\nabla \mathcal{L}(\theta)\|_2^2 \ge \mu (\mathcal{L}(\theta) - \mathcal{L}^*)$$

Under step-dependent learning rate $\eta_t = \eta_{\max} \cos^2\left(\frac{\pi t}{2 S}\right)$:

$$\mathbb{E}[\mathcal{L}(\theta_S) - \mathcal{L}^*] \le \mathcal{O}\left( \exp\left( -\mu \sum_{t=1}^S \eta_t \right) \right) + \mathcal{O}\left( \frac{\sigma^2}{\mu S} \right)$$

Cosine decay guarantees that the stochastic gradient variance term $\frac{\sigma^2}{\mu S}$ decays to zero as $\eta_t \to 0$ at the end of training.
