# Statistical Significance and Hypothesis Testing

This document details the paired statistical hypothesis tests, non-parametric bootstrap resampling, and confidence intervals across primary model comparisons.

---

## 1. Statistical Methodology

Because speech recognition evaluation exhibits heavy sentence-level variance and non-normal error distributions, standard un-paired t-tests are invalid. We employ three complementary statistical frameworks:
1. **Paired Non-Parametric Bootstrap Resampling (10,000 resamples)**: Measures the empirical distribution of $\Delta \text{WER}$ by drawing utterances with replacement:
   $$\Delta \text{WER}^{(b)} = \text{WER}(M_1, \mathcal{S}^{(b)}) - \text{WER}(M_2, \mathcal{S}^{(b)})$$
2. **Wilcoxon Signed-Rank Test**: Paired non-parametric test assessing whether utterance-level error differentials are symmetrically distributed around zero.
3. **Paired Student's t-test**: Parametric comparison of paired sentence error deltas.

---

## 2. Statistical Comparisons Across Primary Models

### Comparison 1: B-01 vs B-02C (Acoustic Scaling on `val.jsonl`, 6,728 utts)
- **Observed $\Delta \text{WER}$**: $-6.02\%$ absolute (33.86% $\rightarrow$ 27.84%)
- **Bootstrap 95% Confidence Interval**: $[-6.33\%, -5.72\%]$
- **Bootstrap $p$-value**: $p < 0.0001$
- **Wilcoxon $p$-value**: $p < 0.0001$
- **Verdict**: **HIGHLY STATISTICALLY SIGNIFICANT**. Acoustic scaling from 92.9h to 165.3h delivers indisputable improvements.

### Comparison 2: B-02A vs B-02B (Adding Quranic Speech on `val.jsonl`)
- **Observed $\Delta \text{WER}$**: $-0.03\%$ absolute (29.42% $\rightarrow$ 29.39%)
- **Bootstrap 95% Confidence Interval**: $[-0.24\%, +0.16\%]$
- **Bootstrap $p$-value**: $p = 0.3764$
- **Verdict**: **NEUTRAL (Statistically Indistinguishable)**. General secular recognition was unchanged, though religious terminology recall improved by +1.52%.

### Comparison 3: B-02C vs E-03 (Foundation vs MWER on Canonical `dev_assessment.jsonl`)
- **Observed $\Delta \text{WER}$**: $-0.006\%$ absolute (24.47% $\rightarrow$ 24.46%; 4,122 vs 4,121 errors; $\Delta = 1$ word)
- **Bootstrap 95% Confidence Interval**: $[-0.142\%, +0.130\%]$
- **Bootstrap $p$-value**: $p = 0.963$
- **Wilcoxon $p$-value**: $p = 0.964$
- **Verdict**: **STATISTICALLY IDENTICAL**. E-03 does not meaningfully outperform B-02C. Historical reports claiming a ~3% gap were distorted by the unresampled evaluation bug.

### Comparison 4: BASE-A vs Stage 21B Champion (Decoder Calibration & Ranker)
- **Observed $\Delta \text{WER}$**: $-0.166\%$ absolute (24.46% $\rightarrow$ 24.30%; 4,121 vs 4,093 errors; $\Delta = -28$ words)
- **Bootstrap 95% Confidence Interval**: $[-0.402\%, +0.072\%]$
- **Bootstrap $p$-value**: $p = 0.0847$
- **Paired t-test $p$-value**: $p = 0.317$
- **Wilcoxon $p$-value**: $p = 0.0848$
- **Verdict**: **MARGINALLY SIGNIFICANT / INCONCLUSIVE**. Because the 95% confidence interval spans across zero ($+0.072\%$), the 28-word reduction cannot be claimed as a statistically certified breakthrough at $\alpha = 0.05$. It represents a modest, positive empirical trend.
