# Mathematical Foundations: Levenshtein Backtracking and Alignment Paths

---

## 1. Backtracking the Optimal Edit Sequence

The dynamic programming matrix $d(i, j)$ computes the minimal distance. To identify the specific operations, an optimal path is backtracked from cell $(N, M)$ to $(0, 0)$:

$$\text{Next Cell} = \arg\min \begin{cases}
d(i-1, j-1) & (\text{Diagonal: Match or Sub}) \\
d(i-1, j)   & (\text{Up: Deletion}) \\
d(i, j-1)   & (\text{Left: Insertion})
\end{cases}$$

---

## 2. Alignment Matrix Invariants

Let $C, S, D, I$ be the counts extracted from the optimal path:
$$N_{\text{ref}} = C + S + D$$
$$N_{\text{hyp}} = C + S + I$$
$$\text{Path Length } K \le N_{\text{ref}} + N_{\text{hyp}}$$
$$\max(N_{\text{ref}}, N_{\text{hyp}}) \le K \le N_{\text{ref}} + N_{\text{hyp}}$$
