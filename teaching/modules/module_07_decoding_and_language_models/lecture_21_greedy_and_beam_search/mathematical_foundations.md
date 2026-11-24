# Mathematical Foundations: Prefix Beam Search and Path Compression

---

## 1. Prefix State Transition Equations

Let $\ell$ be a prefix string over alphabet $\Omega$, and $c \in \Omega$ be a character emission.
The complete dynamic update equations for time frame $t$ given frame emission probabilities $\mathbf{y}^t$ are:

### Blank Extension:
$$P_b(\ell, t) = \left( P_b(\ell, t-1) + P_{nb}(\ell, t-1) \right) y_\epsilon^t$$

### Non-Blank Extensions:
For each symbol $c \in \Omega$:

$$\ell^+ = \ell \circ c$$

If $c = \text{end}(\ell)$:
$$P_{nb}(\ell, t) \mathrel{+}= P_{nb}(\ell, t-1) \cdot y_c^t$$
$$P_{nb}(\ell^+, t) \mathrel{+}= P_b(\ell, t-1) \cdot y_c^t$$

If $c \neq \text{end}(\ell)$:
$$P_{nb}(\ell^+, t) \mathrel{+}= \left( P_b(\ell, t-1) + P_{nb}(\ell, t-1) \right) y_c^t$$

---

## 2. Pruning Complexity
For beam width $W$ and vocabulary size $|\Omega|$:
- Number of candidate prefix extensions per frame: $W \times (|\Omega| + 1)$.
- Sorting/Selection via max-heap or partial selection: $\mathcal{O}(W \cdot |\Omega| + W \log W)$.
- Overall time complexity for sequence $T$: $\mathcal{O}(T \cdot W \cdot |\Omega|)$.
