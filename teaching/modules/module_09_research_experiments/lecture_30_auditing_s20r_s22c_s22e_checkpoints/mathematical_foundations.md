# Mathematical Foundations: Tensor Hash Invariants and Bit-Level Parity

---

## 1. Tensor-Level Deterministic Hashing

Directly hashing a PyTorch `.pt` file can produce different SHA-256 hashes even when model weights are mathematically identical (due to zip container timestamps, serialization order, or PyTorch versions).

To guarantee **bit-level mathematical parity**, tensor hashing computes SHA-256 over the sorted contiguous memory bytes of each individual state dict tensor:

$$H_{\text{model}} = \text{SHA256}\left( \bigoplus_{k \in \text{sorted}(\mathcal{K})} \text{bytes}(k) \circ \text{bytes}(\mathbf{W}_k.\text{contiguous}()) \right)$$

where $\mathcal{K}$ is the set of parameter keys in the state dict, and $\bigoplus$ represents sequential cryptographic hashing.

---

## 2. Numerical Equivalence Bound

Two model checkpoints $\theta_A$ and $\theta_B$ are deemed **computationally equivalent** if:
$$\max_{k \in \mathcal{K}} \|\mathbf{W}_k^{(A)} - \mathbf{W}_k^{(B)}\|_\infty < \epsilon$$
where $\epsilon = 10^{-7}$ for FP32 weights.
