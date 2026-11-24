# Lecture 24: Levenshtein Distance, Word Error Rate (WER), and Character Error Rate (CER)

---

## 1. The Mathematical Definition of Edit Distance

Evaluating speech recognition requires measuring the minimum string editing operations required to transform a model hypothesis sequence $\mathbf{H} = (h_1, \dots, h_M)$ into a ground-truth reference sequence $\mathbf{R} = (r_1, \dots, r_N)$.

The **Levenshtein Distance** $D(N, M)$ defines an optimal alignment using three primitive atomic operations:
1. **Substitution ($S$)**: Replacing reference token $r_i$ with distinct hypothesis token $h_j$.
2. **Deletion ($D$)**: Deleting reference token $r_i$ (omitted in hypothesis).
3. **Insertion ($I$)**: Inserting extra hypothesis token $h_j$ (absent in reference).

### Dynamic Programming Matrix Recursion:
Let $d(i, j)$ be the minimum edit operations to transform prefix $\mathbf{R}_{1:i}$ to $\mathbf{H}_{1:j}$:
$$d(i, 0) = i \quad \forall i \in \{0, \dots, N\} \quad (\text{all deletions})$$
$$d(0, j) = j \quad \forall j \in \{0, \dots, M\} \quad (\text{all insertions})$$

For $i \ge 1, j \ge 1$:
$$d(i, j) = \min \begin{cases}
d(i-1, j) + 1 & (\text{Deletion}) \\
d(i, j-1) + 1 & (\text{Insertion}) \\
d(i-1, j-1) + \mathbb{I}(r_i \neq h_j) & (\text{Match / Substitution})
\end{cases}$$

where $\mathbb{I}(\cdot)$ is the indicator function returning $1$ if $r_i \neq h_j$ and $0$ if $r_i = h_j$.
The execution complexity for sequence lengths $N$ and $M$ is $\mathcal{O}(NM)$.

---

## 2. Word Error Rate (WER) and Character Error Rate (CER)

Given cumulative counts across an evaluation dataset:
- Total Substitutions: $S$
- Total Deletions: $D$
- Total Insertions: $I$
- Total Reference Tokens: $N = S + D + C$ (where $C$ is correct matches)

The standard error rate metric is:
$$\text{Error Rate} = \frac{S + D + I}{N} = \frac{S + D + I}{S + D + C}$$

### Why WER Can Exceed 100%:
Unlike standard classification accuracy bounded in $[0, 1]$, WER has **no upper bound**:
$$\text{WER} \in [0, +\infty)$$
If a model hallucinates an excessively verbose hypothesis (e.g., $N = 5$ reference words, but hypothesis contains $25$ insertions), then:
$$\text{WER} = \frac{0 + 0 + 25}{5} = \frac{25}{5} = \mathbf{500\%}$$

### Micro-Averaged vs. Macro-Averaged WER:
- **Micro-Averaged WER (Scientific Standard)**: Sums all edit operations across the entire test corpus before dividing by total reference words:
  $$\text{WER}_{\text{micro}} = \frac{\sum_{k=1}^K (S_k + D_k + I_k)}{\sum_{k=1}^K N_k}$$
- **Macro-Averaged WER (Flawed Metric)**: Computes WER independently for each utterance and averages the percentages:
  $$\text{WER}_{\text{macro}} = \frac{1}{K} \sum_{k=1}^K \frac{S_k + D_k + I_k}{N_k}$$
  Macro-averaging creates catastrophic distortion: a 1-word utterance with 1 insertion scores $200\%$, skewing the entire benchmark mean. EduCasheer strictly mandates **micro-averaged WER**.
