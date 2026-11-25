# Mathematical Foundations: Graph Partitioning and Leakage Verification

---

## 1. Graph Bipartition Optimization Problem

Let $S = \{s_1, \dots, s_K\}$ be the set of speakers, and $D(s)$ be the total speech duration of speaker $s$:

$$D(s) = \sum_{u \in U(s)} \text{duration}(u)$$

We seek an assignment variable $z_s \in \{0, 1\}$ (where $1$ denotes Test set and $0$ denotes Train set) to satisfy:

$$\min_{z} \left| \sum_{s \in S} z_s D(s) - 0.10 \sum_{s \in S} D(s) \right|$$

subject to the strict disjointness condition:

$$U_{\text{train}} \cap U_{\text{test}} = \emptyset$$

---

## 2. Contamination Invariant Proof

Let $\phi(u)$ denote the speaker attribution function mapping utterance $u$ to speaker ID $s$.
The overlap contamination metric $C$ is defined as:

$$C = \frac{| \{ s \in S \mid s \in \phi(U_{\text{train}}) \land s \in \phi(U_{\text{test}}) \} |}{|\phi(U_{\text{test}})|}$$

A certified disjoint split has the invariant:

$$C \equiv 0.000$$
