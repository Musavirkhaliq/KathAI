# Exercises: Lecture 25 (Scientific Benchmarking Methodology)

---

## Exercise 1: Composite Metric Calculation
A certified model evaluation produces the following domain metrics:
- Domain 1: $N_1 = 4,000 \text{ words}, \text{WER}_1 = 12.5\%$
- Domain 2: $N_2 = 6,000 \text{ words}, \text{WER}_2 = 22.0\%$
- Domain 3: $N_3 = 10,000 \text{ words}, \text{WER}_3 = 38.0\%$
1. Calculate the total edit errors $E_1, E_2, E_3$ in each domain.
2. Calculate the composite micro-averaged WER across the entire 20,000-word benchmark.
3. What is the macro-averaged WER across the 3 domains?

---

## Exercise 2: Auditing Invariant Violations
An external lab claims to reproduce EduCasheer's 24.12% WER using a smaller model, but their report states:
*"Audio was resampled using bilinear interpolation, and decoder alpha was tuned separately for each domain to achieve lowest error."*
Identify which protocol invariants were violated and explain why their claim is invalid.
