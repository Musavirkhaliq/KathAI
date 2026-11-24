# Exercises: Lecture 34 (MLOps and Golden Logit Parity Testing)

---

## Exercise 1: Tolerance Threshold Analysis
1. Explain why setting the parity threshold to $\tau = 0.0$ (exact bit-for-bit equality) causes automated tests to fail across different GPU hardware architectures (e.g., NVIDIA Ampere vs. Hopper).
2. What makes $\tau = 10^{-5}$ an optimal tolerance for FP32 logits?

---

## Exercise 2: Continuous Integration Pipeline Design
Design an automated GitHub Actions CI workflow for an ASR repository that triggers on every pull request.
List the five test stages that must execute before code can be merged into `main`.
