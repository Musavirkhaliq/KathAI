# Quiz: Lecture 26 (Statistical Significance and Bootstrap Resampling)

---

### Question 1:
Why are parametric statistical tests (like Student's t-test) inappropriate for comparing ASR Word Error Rates?
- [A] Speech error distributions are heavily non-normal, right-skewed, and exhibit variable variance across sentence lengths.
- [B] They take too long to run.
- [C] Speech cannot be stored as numbers.
- [D] They only work for images.

**Correct Answer**: **[A]**  
*Explanation*: Speech errors violate the normality and homoscedasticity assumptions of parametric tests.

---

### Question 2:
How is a sample constructed in paired bootstrap resampling?
- [A] By recording new speech files in a studio.
- [B] By sampling $M$ utterances uniformly at random with replacement from the existing test corpus.
- [C] By deleting all short sentences.
- [D] By averaging the audio waveforms.

**Correct Answer**: **[B]**  
*Explanation*: Random sampling with replacement generates the empirical bootstrap distribution.

---

### Question 3:
How many bootstrap resamples ($B$) are standardly evaluated to achieve stable p-value estimation?
- [A] 10
- [B] 100
- [C] 10,000
- [D] 1,000,000

**Correct Answer**: **[C]**  
*Explanation*: $B = 10,000$ provides tight resolution for confidence intervals and p-values down to $\alpha = 0.01$.

---

### Question 4:
What does a 95% confidence interval of $[23.45\%, 24.81\%]$ for model WER signify?
- [A] The model will fail 5% of the time.
- [B] Under repeated sampling from the population of speakers, 95% of estimated intervals will contain the true population WER.
- [C] The model has 23 layers.
- [D] The audio sample rate is 24 kHz.

**Correct Answer**: **[B]**  
*Explanation*: Confidence intervals quantify the precision of empirical point estimates.

---

### Question 5:
What correction must be applied to the significance threshold $\alpha$ when conducting multiple pairwise model comparisons?
- [A] Temperature scaling
- [B] Bonferroni correction ($\alpha^* = \alpha / m$)
- [C] Sinc interpolation
- [D] Levenshtein penalty

**Correct Answer**: **[B]**  
*Explanation*: Bonferroni correction controls the family-wise error rate across multiple hypothesis tests.
