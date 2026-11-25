# Mathematical Foundations: Low-Rank Decomposition and SVD Projections

---

## 1. Low-Rank Matrix Factorization

Let $\mathbf{W}_0 \in \mathbb{R}^{d \times k}$ with rank $k$.
The task-specific update $\Delta \mathbf{W}$ is constrained to have rank $r$:

$$\Delta \mathbf{W} = \mathbf{B} \mathbf{A}, \quad \mathbf{B} \in \mathbb{R}^{d \times r}, \mathbf{A} \in \mathbb{R}^{r \times k}, \quad r \ll \min(d, k)$$

### Forward Pass:
$$\mathbf{h} = \mathbf{W}_0 \mathbf{x} + \frac{\alpha}{r} \mathbf{B} \mathbf{A} \mathbf{x}$$

### Initialization Invariant:
$\mathbf{A}$ is initialized from Gaussian distribution $\mathcal{N}\left(0, \frac{1}{r}\right)$, and $\mathbf{B}$ is initialized to strictly **zero**:

$$\mathbf{B} = \mathbf{0} \implies \Delta \mathbf{W} = \mathbf{0} \cdot \mathbf{A} = \mathbf{0}$$

At step 0 of fine-tuning, the adapted model behaves identically to the original base model, preventing initial perturbation.
