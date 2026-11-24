# Mathematical Foundations: Numerical Precision and Logit Invariants

---

## 1. IEEE 754 Floating-Point Precision Bounds

In single precision (FP32), machine epsilon is $\epsilon_{\text{mach}} \approx 1.19 \times 10^{-7}$.
For non-deterministic GPU kernel execution (e.g., cuDNN atomic adds across threads), slight reordering of floating-point additions ($a + b + c \neq (a + b) + c$) introduces minor numerical drift:

$$\| \mathbf{z}_{\text{run1}} - \mathbf{z}_{\text{run2}} \|_\infty \le \mathcal{O}(L \cdot \epsilon_{\text{mach}})$$
where $L = 24$ is the network depth.
The certified tolerance threshold is established at $\tau = 10^{-5}$.
