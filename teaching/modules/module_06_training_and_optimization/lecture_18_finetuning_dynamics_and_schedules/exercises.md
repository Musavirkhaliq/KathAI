# Exercises: Lecture 18 (Fine-Tuning Dynamics and Schedules)

---

## Exercise 1: LLRD Layer Ratio Calculation
For a 24-layer Transformer with base learning rate $\eta_{\text{base}} = 5 \times 10^{-4}$ and decay factor $\xi = 0.80$:
1. Calculate the exact learning rate for Layer 20, Layer 10, and Layer 1.
2. What is the ratio of learning rate between Layer 24 and Layer 1 ($\frac{\eta_{24}}{\eta_1}$)?

---

## Exercise 2: Cosine Step Evaluation
A training run has $S = 10,000$ steps with $S_{\text{warmup}} = 1,000$ steps, $\eta_{\max} = 3 \times 10^{-4}$, and $\eta_{\min} = 1 \times 10^{-6}$.
Calculate the learning rate at:
1. Step $s = 500$ (during warmup).
2. Step $s = 5,500$ (halfway through cosine decay).
