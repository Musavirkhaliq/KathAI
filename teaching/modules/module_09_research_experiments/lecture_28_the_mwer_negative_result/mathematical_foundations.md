# Mathematical Foundations: Risk Minimization and Policy Gradient Variance

---

## 1. Expected Risk Minimization Formulation

Let sequence hypothesis space be $\mathcal{H}$ with ground-truth reference $\mathbf{y}^*$ and task loss function $L(\mathbf{w}, \mathbf{y}^*)$.
The risk objective is:

$$\mathcal{R}(\theta) = \sum_{\mathbf{w} \in \mathcal{H}} P_\theta(\mathbf{w} \mid \mathbf{X}) L(\mathbf{w}, \mathbf{y}^*)$$

By the REINFORCE / log-derivative trick:

$$\nabla_\theta \mathcal{R}(\theta) = \sum_{\mathbf{w} \in \mathcal{H}} \nabla_\theta P_\theta(\mathbf{w} \mid \mathbf{X}) L(\mathbf{w}, \mathbf{y}^*) = \sum_{\mathbf{w} \in \mathcal{H}} P_\theta(\mathbf{w} \mid \mathbf{X}) \nabla_\theta \log P_\theta(\mathbf{w} \mid \mathbf{X}) L(\mathbf{w}, \mathbf{y}^*)$$

Introducing baseline $\bar{L} = \sum_{\mathbf{w}'} P_\theta(\mathbf{w}') L(\mathbf{w}', \mathbf{y}^*)$:

$$\nabla_\theta \mathcal{R}(\theta) = \mathbb{E}_{\mathbf{w} \sim P_\theta} \left[ \nabla_\theta \log P_\theta(\mathbf{w} \mid \mathbf{X}) (L(\mathbf{w}, \mathbf{y}^*) - \bar{L}) \right]$$

---

## 2. Gradient Variance Analysis

The variance of the Monte Carlo gradient estimate over batch size $B$ is:

$$\text{Var}(\hat{\mathbf{g}}) = \frac{1}{B} \mathbb{E}\left[ \| \nabla_\theta \log P_\theta(\mathbf{w}) (L(\mathbf{w}, \mathbf{y}^*) - \bar{L}) \|_2^2 \right]$$

In low-resource domains with small beams ($|\mathcal{H}| \le 8$), the empirical variance scales as $\mathcal{O}\left(\frac{1}{|\mathcal{H}|}\right)$, destabilizing AdamW momentum accumulators.
