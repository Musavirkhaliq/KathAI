# Exercises: Lecture 12 (Self-Attention and Transformers)

---

## Exercise 1: Attention Memory Calculation
An XLS-R Transformer layer processes an audio clip of 15 seconds at 16 kHz.
1. Given downsampling factor 320, how many acoustic frames $T$ enter the Transformer?
2. What are the dimensions of the attention score matrix $\mathbf{Q} \mathbf{K}^T$ for 16 attention heads with batch size $B = 8$?
3. Calculate the memory in megabytes (MB) required to store the attention weight matrices for all 24 layers during the forward pass in 32-bit floating point precision.

---

## Exercise 2: Masked Attention Invariance
In self-attention, padding frames are masked by setting their attention scores to $-10^9$ before Softmax.
Prove mathematically that if $z_j = -10^9$, then $P_j = \text{Softmax}(\mathbf{z})_j \approx 0$ and $\frac{\partial \mathcal{L}}{\partial z_j} \approx 0$.
