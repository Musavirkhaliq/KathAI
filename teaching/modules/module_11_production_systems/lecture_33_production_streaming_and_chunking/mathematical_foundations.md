# Mathematical Foundations: Overlap-Add Logit Stitching and Latency Bounds

---

## 1. Linear Cross-Fade Window Formulation

Let overlap region have duration $T_{\text{ov}}$ frames.
For frame index $\tau \in \{0, \dots, T_{\text{ov}}-1\}$:

$$w(\tau) = \frac{\tau}{T_{\text{ov}} - 1}$$

The stitched logit for class $k$ is:

$$z_k^{\text{stitched}}(\tau) = (1 - w(\tau)) z_k^{(1)}(\tau) + w(\tau) z_k^{(2)}(\tau)$$

---

## 2. Latency Budget Analysis

End-to-end streaming latency $L_{\text{total}}$ consists of:

$$L_{\text{total}} = T_{\text{chunk}} + T_{\text{network}} + T_{\text{inference}} + T_{\text{decode}}$$

For Method C2:
- $T_{\text{chunk}} = 1.0 \text{ s}$ (streaming buffer step)
- $T_{\text{inference}} \approx 0.08 \text{ s}$
- $T_{\text{decode}} \approx 0.05 \text{ s}$
- Total end-to-end latency: $\approx \mathbf{1.15 \text{ seconds}}$, well within live interactive broadcasting tolerances.
