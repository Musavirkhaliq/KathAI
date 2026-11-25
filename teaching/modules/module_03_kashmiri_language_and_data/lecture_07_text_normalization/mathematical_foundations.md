# Mathematical Foundations: String Transformation Invariants and Idempotence

---

## 1. Mathematical Properties of a Certified Normalizer

Let $\Sigma^*$ be the set of all finite character sequences over Unicode alphabet $\Sigma$.
A text normalizer is a mapping $\mathcal{N}: \Sigma^* \to \Sigma_c^*$, where $\Sigma_c \subset \Sigma$ is the certified canonical alphabet ($|\Sigma_c| = 47$).

A certified normalizer must satisfy three strict mathematical properties:

### A. Idempotence:
Applying the normalizer repeatedly after the first pass produces no additional changes:

$$\mathcal{N}(\mathcal{N}(s)) = \mathcal{N}(s) \quad \forall s \in \Sigma^*$$

### B. Commutativity with Concatenation (Modulo Whitespace):
For strings $s_1, s_2 \in \Sigma^*$:

$$\mathcal{N}(s_1 \circ s_2) = \mathcal{N}(s_1) \circ \text{" "} \circ \mathcal{N}(s_2)$$

### C. Partition Invariance (Homoglyph Equivalence Classes):
Let $\sim$ be the equivalence relation where $c_1 \sim c_2$ if $c_1$ and $c_2$ are visual or phonetic homoglyphs.

$$\forall c_1, c_2 \in \Sigma, \quad c_1 \sim c_2 \implies \mathcal{N}(c_1) = \mathcal{N}(c_2)$$
