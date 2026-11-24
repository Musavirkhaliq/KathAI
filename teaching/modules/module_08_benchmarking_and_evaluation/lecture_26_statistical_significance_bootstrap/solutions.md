# Solutions: Lecture 26 (Statistical Significance and Bootstrap Resampling)

---

## Solution to Exercise 1
1. The 95% confidence interval is: **$[23.45\%, 24.81\%]$**.
2. No! The competitor's score of $23.90\%$ falls squarely inside our 95% confidence interval $[23.45\%, 24.81\%]$. Because the point estimate lies within the range of sampling variation, there is no statistically significant difference without conducting a direct paired bootstrap comparison.

---

## Solution to Exercise 2
1. Under Bonferroni correction for $m = 8$ hypotheses:
   $$\alpha^* = \frac{\alpha}{m} = \frac{0.05}{8} = \mathbf{0.00625}$$
2. For $p = 0.012$:
   Since $0.012 > 0.00625$, the result is **not statistically significant** after Bonferroni correction. It could readily arise as a false positive due to multiple hypothesis testing.
