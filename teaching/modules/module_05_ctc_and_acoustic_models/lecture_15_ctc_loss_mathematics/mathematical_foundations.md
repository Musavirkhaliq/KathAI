# Mathematical Foundations: CTC Backward Variables and Gradient Derivations

---

## 1. The Backward Variable $\beta_t(s)$

The backward variable $\beta_t(s)$ represents the total probability mass of partial alignments from time $t$ to $T$ given state $s$ at time $t$:

### Base Case ($t = T$):
$$\beta_T(2U + 1) = 1, \quad \beta_T(2U) = 1, \quad \beta_T(s) = 0 \quad \forall s < 2U$$

### Recursive Step ($t < T$):
$$\beta_t(s) = \begin{cases}
\left( \beta_{t+1}(s) y_{\mathbf{l}'_s}^{t+1} + \beta_{t+1}(s+1) y_{\mathbf{l}'_{s+1}}^{t+1} \right) & \text{if } \mathbf{l}'_s = \epsilon \text{ or } \mathbf{l}'_s = \mathbf{l}'_{s+2} \\
\left( \beta_{t+1}(s) y_{\mathbf{l}'_s}^{t+1} + \beta_{t+1}(s+1) y_{\mathbf{l}'_{s+1}}^{t+1} + \beta_{t+1}(s+2) y_{\mathbf{l}'_{s+2}}^{t+1} \right) & \text{otherwise}
\end{cases}$$

---

## 2. Derivation of the Logit Gradient

Using Bayes' rule, the posterior probability of state $s$ at time $t$ is:
$$P(\pi_t = \mathbf{l}'_s \mid \mathbf{X}, \mathbf{l}) = \frac{\alpha_t(s) \beta_t(s)}{P(\mathbf{l} \mid \mathbf{X})}$$

Summing over all occurrences of character $k$:
$$\gamma_t(k) = \frac{1}{P(\mathbf{l} \mid \mathbf{X})} \sum_{s \in \text{lab}(\mathbf{l}', k)} \alpha_t(s) \beta_t(s)$$

Since $\frac{\partial \log P(\mathbf{l} \mid \mathbf{X})}{\partial y_k^t} = \frac{\gamma_t(k)}{y_k^t}$, applying the Softmax derivative $\frac{\partial y_k^t}{\partial z_j^t} = y_k^t (\delta_{kj} - y_j^t)$ yields:

$$\frac{\partial \mathcal{L}_{\text{CTC}}}{\partial z_j^t} = -\sum_k \frac{\partial \log P}{\partial y_k^t} \frac{\partial y_k^t}{\partial z_j^t} = -\sum_k \frac{\gamma_t(k)}{y_k^t} y_k^t (\delta_{kj} - y_j^t) = y_j^t \sum_k \gamma_t(k) - \gamma_t(j) = y_j^t - \gamma_t(j) \quad \blacksquare$$
