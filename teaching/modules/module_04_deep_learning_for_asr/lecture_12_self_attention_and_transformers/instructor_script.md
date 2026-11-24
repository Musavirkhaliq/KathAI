# Instructor Delivery Script: Lecture 12 (Multi-Head Self-Attention for Speech)

## Timetable (90 Minutes)
```
00:00 - 00:20 | From Sequential RNNs to Parallel Self-Attention
00:20 - 00:45 | Mathematical Proof: Dot Product Variance and the 1/sqrt(dk) Scale Factor
00:45 - 01:10 | Multi-Head Decomposition in XLS-R (16 Heads, 1024 Dimensions)
01:10 - 01:25 | Code Walkthrough: Scaled Dot Product Attention
01:25 - 01:30 | Q&A and Synthesis
```

---

## Instructor Notes & Dialogue

### 00:20 - 00:45 | The 1/sqrt(dk) Proof
- Write on the board:
  $$z = \sum_{i=1}^{64} q_i k_i$$
- Ask the students:
  > "If $q$ and $k$ are standard normal vectors, what is the standard deviation of $z$?"
- Wait for a student to say $\sqrt{64} = 8$.
- Plot the Softmax sigmoid on the board:
  > "If your inputs are $+8$ and $-8$, what does Softmax output?
  > $[0.9997, 0.0003]$.
  > What is the gradient of Softmax at 0.9997? It is essentially zero!
  > If you omit $\frac{1}{\sqrt{d_k}}$, your Transformer dies on day one because backpropagation cannot pass through the saturated Softmax.
  > Dividing by $\sqrt{d_k}$ is the simple mathematical lever that keeps attention alive."
