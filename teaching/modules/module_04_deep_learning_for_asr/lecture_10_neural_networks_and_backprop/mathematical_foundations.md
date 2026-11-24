# Mathematical Foundations: Gradient Flow and Residual Stability

---

## 1. Residual Connection Gradient Preservation

Consider a residual building block:
$$\mathbf{h}^{(l)} = \mathbf{h}^{(l-1)} + \mathcal{F}(\mathbf{h}^{(l-1)}, \mathcal{W}^{(l)})$$

The Jacobian with respect to the input activation is:
$$\frac{\partial \mathbf{h}^{(l)}}{\partial \mathbf{h}^{(l-1)}} = \mathbf{I} + \frac{\partial \mathcal{F}}{\partial \mathbf{h}^{(l-1)}}$$

For any deep network spanning layer $m$ to layer $n$ ($n > m$):
$$\mathbf{h}^{(n)} = \mathbf{h}^{(m)} + \sum_{i=m}^{n-1} \mathcal{F}(\mathbf{h}^{(i)}, \mathcal{W}^{(i+1)})$$

The gradient with respect to $\mathbf{h}^{(m)}$ is:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(m)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(n)}} \frac{\partial \mathbf{h}^{(n)}}{\partial \mathbf{h}^{(m)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(n)}} \left( \mathbf{I} + \frac{\partial}{\partial \mathbf{h}^{(m)}} \sum_{i=m}^{n-1} \mathcal{F} \right)$$

Because of the identity matrix term $\mathbf{I}$, the gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(m)}}$ does not vanish even if the residual branch derivatives $\frac{\partial \mathcal{F}}{\partial \mathbf{h}^{(i)}}$ are close to zero.
