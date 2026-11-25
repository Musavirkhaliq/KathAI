# Mathematical Foundations: Dynamic Time Warping and Sequence Distances

---

## 1. Dynamic Time Warping Formulation

Let sequence $\mathbf{X} = (\mathbf{x}_1, \dots, \mathbf{x}_N)$ and $\mathbf{Y} = (\mathbf{y}_1, \dots, \mathbf{y}_M)$ have local distance metric $d(\mathbf{x}_i, \mathbf{y}_j) = \|\mathbf{x}_i - \mathbf{y}_j\|_2$.

A warping path $W = (w_1, w_2, \dots, w_K)$ with $w_k = (i_k, j_k)$ must satisfy:
1. **Boundary Conditions**: $w_1 = (1, 1)$ and $w_K = (N, M)$.
2. **Monotonicity**: $i_1 \le i_2 \le \dots \le i_K$ and $j_1 \le j_2 \le \dots \le j_K$.
3. **Step Size Invariant**: $w_{k+1} - w_k \in \{(1, 0), (0, 1), (1, 1)\}$.

The optimal path minimizes cumulative cost:

$$\text{DTW}(\mathbf{X}, \mathbf{Y}) = \min_{W} \sum_{k=1}^K d(w_k)$$

Solved via dynamic programming in $\mathcal{O}(NM)$ operations.
