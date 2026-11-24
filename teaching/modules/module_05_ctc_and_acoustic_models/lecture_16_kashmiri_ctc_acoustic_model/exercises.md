# Exercises: Lecture 16 (Building the Kashmiri CTC Acoustic Model)

---

## Exercise 1: Parameter Ratio Calculation
Given the parameter counts derived in the lecture:
1. What percentage of the model's total parameters reside in the 24 Transformer layers?
2. What percentage of parameters reside in the task-specific CTC classification head?
3. What does this ratio tell us about the risk of overfitting in the projection head vs. the backbone?

---

## Exercise 2: Blank Bias Softmax Shift
Suppose $d_{\text{vocab}} = 47$.
1. If $\mathbf{b} = \mathbf{0}$, what is $P(\epsilon)$?
2. If $b_0 = 3.0$ and $b_k = 0$ for all $k \ge 1$, calculate $P(\epsilon)$ when $\mathbf{h} = \mathbf{0}$.
