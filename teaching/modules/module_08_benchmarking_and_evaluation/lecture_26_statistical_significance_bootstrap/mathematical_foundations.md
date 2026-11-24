# Mathematical Foundations: Bootstrap Consistency and P-Value Bounds

---

## 1. Asymptotic Consistency of Empirical Bootstrap

Let $\hat{\theta}_n$ be an estimator based on sample size $n$ with true parameter $\theta_0$.
According to Bickel and Freedman (1981), if the underlying statistics possess finite variance:

$$\sup_{x} \left| P^*\left( \sqrt{n}(\theta^* - \hat{\theta}_n) \le x \right) - P\left( \sqrt{n}(\hat{\theta}_n - \theta_0) \le x \right) \right| \xrightarrow{P} 0 \quad \text{as } n \to \infty$$

The bootstrap distribution $P^*$ asymptotically mirrors the true sampling distribution of the test statistic.

---

## 2. Paired Wilcoxon Signed-Rank Test

For paired utterance differences $d_i = \text{WER}_A(u_i) - \text{WER}_B(u_i)$:
1. Exclude ties ($d_i = 0$).
2. Rank absolute differences $|d_i|$ from smallest to largest with ranks $R_i$.
3. Compute sum of positive ranks $W^+$ and negative ranks $W^-$:
   $$W = \min(W^+, W^-)$$
For $N > 20$, the test statistic $z$ is approximately standard normal:
$$z = \frac{W - \frac{N(N+1)}{4}}{\sqrt{\frac{N(N+1)(2N+1)}{24}}} \sim \mathcal{N}(0, 1)$$
