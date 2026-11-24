# Exercises: Lecture 10 (Deep Neural Networks and Backpropagation)

---

## Exercise 1: GELU Derivative
1. Write the mathematical definition of GELU using the standard Gaussian CDF $\Phi(x)$.
2. Compute the first derivative $\frac{d}{dx}\text{GELU}(x)$ with respect to $x$.
3. Evaluate the derivative at $x = 0$ and $x = -2.0$. Explain why GELU does not suffer from dead neurons like ReLU.

---

## Exercise 2: LayerNorm Invariance Proof
Prove that LayerNorm is invariant to:
1. Scaling of the input vector by positive scalar $\alpha > 0$ ($\text{LN}(\alpha \mathbf{x}) = \text{LN}(\mathbf{x})$).
2. Constant shift of the input vector by scalar $c \in \mathbb{R}$ ($\text{LN}(\mathbf{x} + c \mathbf{1}) = \text{LN}(\mathbf{x})$).
