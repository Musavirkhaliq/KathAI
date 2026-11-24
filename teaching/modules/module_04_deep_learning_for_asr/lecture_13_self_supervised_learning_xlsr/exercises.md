# Exercises: Lecture 13 (Self-Supervised Learning and XLS-R)

---

## Exercise 1: Distractor Probability Analysis
In the wav2vec 2.0 contrastive loss with 100 distractors ($K=100$) and temperature $\tau = 0.1$:
1. Suppose the model is completely untrained, such that $\text{sim}(\mathbf{c}, \mathbf{q}_{\text{pos}}) = \text{sim}(\mathbf{c}, \mathbf{q}_{\text{neg}}) = 0.0$.
   What is the initial theoretical contrastive loss $\mathcal{L}_m$?
2. Suppose after pretraining, $\text{sim}(\mathbf{c}, \mathbf{q}_{\text{pos}}) = 0.85$ and the average distractor similarity is $0.10$.
   Calculate the resulting contrastive loss.

---

## Exercise 2: Codebook Entropy Proof
Let a codebook have $V = 320$ entries.
1. What is the maximum theoretical entropy $H_{\max}$ of the codebook distribution?
2. If all frames collapse to a single codebook index ($p_1 = 1.0, p_k = 0$ for $k > 1$), what is the diversity loss $\mathcal{L}_d$?
