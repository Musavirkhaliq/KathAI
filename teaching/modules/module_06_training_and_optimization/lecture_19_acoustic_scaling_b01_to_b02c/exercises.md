# Exercises: Lecture 19 (Empirical Scaling Analysis)

---

## Exercise 1: Extrapolating Required Data
Using the fitted power-law model $\text{WER}(H) = 18.5 + 85.0 \cdot H^{-0.42}$:
1. Predict the expected disjoint WER if training data is expanded to 500 hours.
2. How many hours would theoretically be required to achieve $\text{WER} = 20.0\%$?
3. Discuss why achieving $15.0\%$ WER is mathematically impossible under this acoustic model without language model enhancements.

---

## Exercise 2: Domain Scaling Discrepancy
Comparing B-01 to B-02C:
- Studio clean WER improved from $28.4\%$ to $14.2\%$ (a relative reduction of $50\%$).
- WAAZ reverberant WER improved from $74.2\%$ to $55.15\%$ (a relative reduction of only $25.7\%$).
Explain the physical and acoustic reasons why spontaneous reverberant speech scales more slowly than studio speech.
