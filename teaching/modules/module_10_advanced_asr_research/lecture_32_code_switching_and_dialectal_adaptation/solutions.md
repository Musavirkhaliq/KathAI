# Solutions: Lecture 32 (Code-Switching and Dialectal Adaptation)

---

## Solution to Exercise 1
1. Parameters in $\mathbf{W}_Q$ and $\mathbf{W}_V$:

$$2 \times (1024 \times 1024) = 2 \times 1,048,576 = \mathbf{2,097,152 \text{ parameters}}$$

2. LoRA parameters for rank $r = 8$:
   Per matrix: $\mathbf{A} \in \mathbb{R}^{8 \times 1024}, \mathbf{B} \in \mathbb{R}^{1024 \times 8} \implies (8 \times 1024) \times 2 = 16,384 \text{ params}$.
   For both $\mathbf{W}_Q$ and $\mathbf{W}_V$:

$$2 \times 16,384 = \mathbf{32,768 \text{ trainable parameters per layer}}$$

3. Percentage reduction:

$$\frac{32,768}{2,097,152} \times 100 = \mathbf{1.56\% \text{ of original parameters! (A 98.44\% reduction)}}$$

---

## Solution to Exercise 2
1. Because the vocabulary is strictly character-level, every letter in `"کَمپیوٗٹَر"` exists in `configs/vocab.json`. The tokenizer spells it out without generating `<unk>` tokens.
2. A language model trained only on archaic literature will have zero counts for `"کَمپیوٗٹَر"`, forcing the model to back off to unigram continuation probabilities, heavily penalizing the hypothesis during beam search.
