# Solutions: Lecture 31 (Neural Rescoring and Oracle Analysis)

---

## Solution to Exercise 1
1. 1-Best:

$$\text{Sentence 1: } E=2, \quad \text{Sentence 2: } E=1 \implies \text{WER}_{\text{1-best}} = \frac{2 + 1}{4 + 6} = \frac{3}{10} = \mathbf{30.0\%}$$

2. Oracle:

$$\text{Sentence 1 min: } E=1, \quad \text{Sentence 2 min: } E=0 \implies \text{WER}_{\text{oracle}} = \frac{1 + 0}{10} = \frac{1}{10} = \mathbf{10.0\%}$$

3. Potential gain:

$$30.0\% - 10.0\% = \mathbf{20.0\% \text{ absolute gain!}}$$

---

## Solution to Exercise 2
An $N$-best list of size 100 represents only 100 discrete paths through the search space. A compact lattice or WFST graph can encode millions of alternate word alignments in a compressed graph structure. Rescoring lattices directly allows the neural model to evaluate paths that were pruned from the top 100 list during first-pass decoding.
