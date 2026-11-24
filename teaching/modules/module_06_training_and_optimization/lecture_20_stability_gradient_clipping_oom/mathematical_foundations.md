# Mathematical Foundations: Attention Memory Complexity and Slicing

---

## 1. Peak Memory Footprint Formulation

Let a speech batch have $B$ sequences padded to maximum frame length $T_{\max}$, hidden dimension $D$, and $L$ Transformer layers.
The peak activation memory for the forward-backward pass is:

$$\mathcal{M}_{\text{peak}} \approx \mathcal{M}_{\text{weights}} + B \cdot T_{\max} \cdot D \cdot L \cdot c_1 + B \cdot H \cdot T_{\max}^2 \cdot L \cdot c_2$$

where:
- $c_1 \approx 4$ bytes (intermediate activations).
- $c_2 \approx 4$ bytes (attention score matrices).
- $H$ is the number of attention heads.

Notice the quadratic term $B \cdot H \cdot T_{\max}^2 \cdot L$: doubling the max duration quadruples attention memory!
