# Mathematical Foundations: Forensic Timestamp and Hash Verification

---

## 1. Cryptographic Lineage Invariant

Let an experiment artifact $A$ consist of:
1. Model Checkpoint Weights: $W \in \mathbb{R}^P$ with SHA-256 hash $H(W)$.
2. Training Manifest: $M$ with hash $H(M)$.
3. Configuration File: $C$ with hash $H(C)$.
4. Source Code Git Tree: Commit hash $G$.

The cryptographic execution tuple is:
$$\mathcal{T} = \langle H(W), H(M), H(C), G \rangle$$

An experimental claim is **empirically substantiated** if and only if:
1. Checkpoint file $W$ exists physically on disk and matches hash $H(W)$.
2. Evaluating $W$ with configuration $C$ on manifest $M$ deterministically reproduces the claimed log metric within numerical floating-point epsilon:
   $$|\text{Eval}(W, M, C) - \text{Metric}_{\text{claimed}}| < 10^{-4}$$
