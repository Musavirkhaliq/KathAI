# Exercises: Lecture 17 (The Complete PyTorch CTC Training Loop)

---

## Exercise 1: Gradient Accumulation Equivalence
Prove that accumulating the gradient of $\frac{1}{K} \sum_{k=1}^K \mathcal{L}_k$ over $K$ steps before updating weights is mathematically identical to performing a single backward pass on the combined batch $\mathcal{B} = \bigcup_{k=1}^K \mathcal{B}_k$.

---

## Exercise 2: Effective Batch Size Calculation
A researcher trains on 2 NVIDIA RTX 3090 GPUs.
- Per-GPU batch size: 8 utterances
- Gradient accumulation steps: 4
1. What is the effective batch size per optimizer step?
2. If the training manifest contains 32,000 utterances, how many optimizer updates occur per epoch?
