# Solutions to Exercises: Lecture 02

## Solution to Problem 1: Conv1d Receptive Field & Stride

### Part 1: Receptive Field Derivation
The backward recursion formula for the receptive field $R_\ell$ of layer $\ell$ with kernel $K_\ell$ and cumulative stride $S_{\ell-1}^{\text{cum}} = \prod_{j=1}^{\ell-1} S_j$ is:

$$R_\ell = R_{\ell-1} + (K_\ell - 1) \cdot S_{\ell-1}^{\text{cum}} \quad \text{with } R_0 = 1, S_0^{\text{cum}} = 1$$

Evaluating across all 7 layers:
- Layer 1: $R_1 = 1 + (10 - 1) \times 1 = 10$
- Layer 2: $R_2 = 10 + (3 - 1) \times 5 = 20$
- Layer 3: $R_3 = 20 + (3 - 1) \times 10 = 40$
- Layer 4: $R_4 = 40 + (3 - 1) \times 20 = 80$
- Layer 5: $R_5 = 80 + (3 - 1) \times 40 = 160$
- Layer 6: $R_6 = 160 + (2 - 1) \times 80 = 240$
- Layer 7: $R_7 = 240 + (2 - 1) \times 160 = 400 \text{ samples}$

### Part 2: Sample Index Range for Frame $t=2$
The first frame ($t=1$, 0-indexed $t=0$) spans input samples $[0, 399]$ (a total of 400 samples).
Because the cumulative stride across all 7 layers is $S_{\text{total}} = 320$, the second frame ($t=2$, 0-indexed $t=1$) is shifted forward by exactly 320 samples:

$$n_{\text{start}} = 1 \times 320 = 320$$
$$n_{\text{end}} = n_{\text{start}} + R_{\text{total}} - 1 = 320 + 400 - 1 = 719$$

The second frame spans sample indices **$[320, 719]$** (inclusive).

### Part 3: Number of Valid CTC Frames for $N = 32,150$ Samples
Using the exact frame formula:

$$T = \left\lfloor \frac{N - 400}{320} \right\rfloor + 1 = \left\lfloor \frac{32,150 - 400}{320} \right\rfloor + 1 = \left\lfloor \frac{31,750}{320} \right\rfloor + 1 = \lfloor 99.21875 \rfloor + 1 = 99 + 1 = 100 \text{ frames}$$

The encoder produces exactly **100 valid CTC frames**.

---

## Solution to Problem 2: CTC Alignment Combinatorics

### Part 1: Alignments for $\mathbf{y} = (c_1, c_2)$ at $T=4$
Let $c_1$ represent character "ك" and $c_2$ represent character "ٲ".
We search for all sequences $(\pi_1, \pi_2, \pi_3, \pi_4) \in (\Omega')^4$ such that $\mathcal{B}(\boldsymbol{\pi}) = (c_1, c_2)$.
Because $c_1 \neq c_2$, intermediate blanks are optional. Valid alignments are:

1. $(c_1, c_2, \epsilon, \epsilon)$
2. $(c_1, \epsilon, c_2, \epsilon)$
3. $(c_1, \epsilon, \epsilon, c_2)$
4. $(\epsilon, c_1, c_2, \epsilon)$
5. $(\epsilon, c_1, \epsilon, c_2)$
6. $(\epsilon, \epsilon, c_1, c_2)$
7. $(c_1, c_1, c_2, \epsilon)$
8. $(c_1, c_1, \epsilon, c_2)$
9. $(c_1, c_1, c_1, c_2)$
10. $(c_1, c_2, c_2, \epsilon)$
11. $(c_1, \epsilon, c_2, c_2)$
12. $(c_1, c_2, c_2, c_2)$
13. $(\epsilon, c_1, c_2, c_2)$
14. $(\epsilon, c_1, c_1, c_2)$

Total cardinality $|\mathcal{B}^{-1}(\mathbf{y})| = \mathbf{14}$ paths.

### Part 2: Alignments for $\mathbf{y}_{\text{dup}} = (c_1, c_1)$ at $T=4$
Because both tokens are identical, they **must be separated by at least one blank token $\epsilon$**. If no blank intervenes, consecutive $c_1$ tokens collapse into a single character: $\mathcal{B}(c_1, c_1) = c_1$.

The required pattern is $c_1 \to \dots \to \epsilon \to \dots \to c_1$.
Enumerating all valid alignments of length $T=4$:

1. $(c_1, \epsilon, c_1, \epsilon)$
2. $(c_1, \epsilon, \epsilon, c_1)$
3. $(\epsilon, c_1, \epsilon, c_1)$
4. $(c_1, c_1, \epsilon, c_1)$
5. $(c_1, \epsilon, c_1, c_1)$

Total cardinality $|\mathcal{B}^{-1}(\mathbf{y}_{\text{dup}})| = \mathbf{5}$ paths.

### Part 3: Combinatorial Comparison
$|\mathcal{B}^{-1}(\mathbf{y})| = 14$ whereas $|\mathcal{B}^{-1}(\mathbf{y}_{\text{dup}})| = 5$.
Repeated identical characters eliminate all direct transition paths without blanks (such as $(c_1, c_1, c_1, c_1)$, $(c_1, c_1, c_1, c_2)$, etc.). The mandatory insertion of at least one $\epsilon$ at $t_{\text{split}}$ reduces the degrees of freedom for frame allocation from $\binom{T}{U}$ to $\binom{T}{U+1}$.

---

## Solution to Problem 3: Shallow Fusion Brevity Bias Calibration

### Part 1: Cause of Extreme Truncation
When $\beta = 0$, the language model score for a candidate sequence $\mathbf{w}$ is:

$$\alpha \sum_{k=1}^K \log P_{\text{LM}}(w_k \mid w_{\lt k})$$

Because every $\log P_{\text{LM}} \le 0$, each emitted word reduces the total hypothesis score by an average of $\alpha \times \bar{H}_{\text{LM}} = 0.50 \times 4.2 = 2.1 \text{ nats}$. The decoder maximizes score by emitting as few words as possible, frequently truncating long utterances after 2 or 3 words.

### Part 2: Minimum Theoretical Value of $\beta$
To neutralize the average negative log-probability penalty per word:

$$\beta_{\text{neutral}} = \alpha \times \bar{H}_{\text{LM}} = 0.50 \times 4.2 = \mathbf{2.10}$$

Setting $\beta \ge 2.10$ provides sufficient positive reward per word to counteract language model brevity bias.

### Part 3: Adjustment when $\alpha$ Doubles
If $\alpha$ doubles from $0.50$ to $1.00$, the average language model penalty per word doubles:

$$\text{Penalty} = 1.00 \times 4.2 = 4.20 \text{ nats}$$

To maintain sequence length balance, $\beta$ must also double to:

$$\beta_{\text{new}} = 1.00 \times 4.2 = \mathbf{4.20}$$

---

## Solution to Problem 4: Speaker Leakage Graph Partitioning

### Part 1: Probability of Speaker Leakage
Let a speaker have 10 files. We sample 20 files without replacement for the test set from 100 total files.
The probability that *none* of the speaker's 10 files are chosen for the test set is:

$$P(\text{all 10 in train}) = \frac{\binom{90}{20}}{\binom{100}{20}} = \frac{\frac{90!}{20! 70!}}{\frac{100!}{20! 80!}} = \frac{80 \times 79 \times \dots \times 71}{100 \times 99 \times \dots \times 91} \approx 0.0898 \text{ (8.98\%)}$$

The probability that *all 10* are in the test set is $\frac{\binom{10}{10}\binom{90}{10}}{\binom{100}{20}} \approx 0.0000$.
Therefore, the probability that the speaker appears in **both** train and test is:

$$P(\text{leakage}) = 1 - P(\text{all in train}) - P(\text{all in test}) \approx 1 - 0.0898 - 0.0000 = \mathbf{0.9102 \text{ (91.02\%)}}$$

There is a $91.0\%$ probability that any given speaker is leaked across both splits!

### Part 2: Mechanism of Artificial Generalization
When speaker leakage occurs, the acoustic model tunes its convolutional filters to the specific formants, pitch ($F_0$), vocal tract length, and room reverberation of those 10 speakers. The model achieves low WER on the test set by recognizing the *speaker's voice profile* rather than decoding speech sounds.

### Part 3: Bipartite Graph Partitioning Algorithm
1. Construct a bipartite graph $G = (V_{\text{spk}}, V_{\text{audio}}, E)$ where edge $(s, a) \in E$ indicates that audio file $a$ was spoken by speaker $s$.
2. Compute connected components $C_1, \dots, C_M$ of $G$.
3. Partition the components: assign entire connected components to either $\mathcal{D}_{\text{train}}$ or $\mathcal{D}_{\text{test}}$, such that:

$$\left( \bigcup_{C \in \mathcal{D}_{\text{train}}} V_{\text{spk}}(C) \right) \cap \left( \bigcup_{C \in \mathcal{D}_{\text{test}}} V_{\text{spk}}(C) \right) = \emptyset$$

This guarantees strictly zero speaker overlap.
