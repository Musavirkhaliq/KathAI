# Exercises: Lecture 26 (Statistical Significance and Bootstrap Resampling)

---

## Exercise 1: Confidence Interval Interpretation
A bootstrap test on Model `E22-01` produces $B = 10,000$ resampled WER estimates.
The sorted estimates yield:
- Index 250 (2.5%): $\text{WER} = 23.45\%$
- Index 5,000 (50.0%): $\text{WER} = 24.12\%$
- Index 9,750 (97.5%): $\text{WER} = 24.81\%$
1. State the 95% confidence interval for the true model WER.
2. If a competing laboratory claims their model achieves 23.90% WER, can we claim our model is statistically inferior to theirs at $p < 0.05$ based solely on this interval? Explain.

---

## Exercise 2: The Bonferroni Correction for Multiple Comparisons
Suppose a researcher evaluates 8 different model variants against a baseline model on the same test set.
To maintain an overall false positive rate of $\alpha = 0.05$ across all 8 tests:
1. What adjusted significance threshold $\alpha^*$ must be used for each individual test under the Bonferroni correction?
2. If a model variant achieves $p = 0.012$, is it statistically significant after Bonferroni correction?
