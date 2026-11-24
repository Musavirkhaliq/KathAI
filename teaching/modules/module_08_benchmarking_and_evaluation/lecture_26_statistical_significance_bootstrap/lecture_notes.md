# Lecture 26: Statistical Significance, Confidence Intervals, and Bootstrap Resampling

---

## 1. The Myth of the 0.5% WER Improvement

In machine learning research literature, papers frequently claim state-of-the-art superiority based on tiny margin improvements:
*"Model A achieved 24.12% WER, outperforming Model B's 24.40% WER by 0.28%."*

Without statistical testing, this claim is scientifically meaningless:
- Is a 0.28% difference a genuine structural algorithmic breakthrough?
- Or is it merely random statistical noise arising from the specific sample of test speakers?

In low-resource speech recognition with finite test sets, **hypothesis testing and confidence interval estimation are scientifically mandatory**.

---

## 2. Why Parametric Tests (t-tests) Fail for Speech Recognition

Standard statistical hypothesis tests (such as Student's paired $t$-test) make strict distributional assumptions:
1. Data points must be normally distributed: $X \sim \mathcal{N}(\mu, \sigma^2)$.
2. Observations must have constant variance (homoscedasticity).

Word error rates on speech sentences violate every parametric assumption:
- Error distributions are **heavily right-skewed** (most clean sentences have 0 errors, while noisy sentences have high error spikes).
- Sentence lengths vary from 1 word to 40 words, causing variance to fluctuate wildly.

Consequently, speech researchers must rely on **non-parametric statistical methods**:
1. **Paired Bootstrap Resampling**
2. **Paired Wilcoxon Signed-Rank Test**

---

## 3. The Non-Parametric Paired Bootstrap Test

Proposed by Efron (1979) and adapted to speech by Bisani and Ney (2004), the paired bootstrap estimates the empirical sampling distribution without making parametric assumptions.

### Algorithm:
Given test corpus of $M$ utterances and two competing models $A$ and $B$:
1. For bootstrap iteration $b = 1, \dots, B$ (typically $B = 10,000$):
   - Sample $M$ utterances **uniformly at random with replacement** from the test corpus: $\mathcal{S}^{(b)} = (u_1^*, \dots, u_M^*)$.
   - Evaluate micro-averaged WER for both models on sample $\mathcal{S}^{(b)}$: $\text{WER}_A^{(b)}$ and $\text{WER}_B^{(b)}$.
   - Calculate the delta: $\Delta^{(b)} = \text{WER}_B^{(b)} - \text{WER}_A^{(b)}$.
2. Compute the **empirical two-tailed $p$-value**:
   $$p = 2 \cdot \min\left( \frac{1}{B} \sum_{b=1}^B \mathbb{I}(\Delta^{(b)} \le 0), \frac{1}{B} \sum_{b=1}^B \mathbb{I}(\Delta^{(b)} \ge 0) \right)$$
3. If $p < 0.05$, the difference is statistically significant at the $95\%$ confidence level.

---

## 4. Bootstrap Confidence Intervals (Percentile Method)

To construct a $95\%$ confidence interval for Model A's true error rate:
1. Sort the $B$ bootstrap estimates in ascending order:
   $$\text{WER}_{(1)} \le \text{WER}_{(2)} \le \dots \le \text{WER}_{(B)}$$
2. The $95\%$ confidence interval is given by the $2.5\text{th}$ and $97.5\text{th}$ percentiles:
   $$\text{CI}_{95\%} = \left[ \text{WER}_{(\lfloor 0.025 B \rfloor)}, \text{WER}_{(\lfloor 0.975 B \rfloor)} \right]$$

For the certified EduCasheer model `E22-01`:
$$\text{WER}_{\text{composite}} = 24.12\% \quad (\text{95\% CI: } [23.45\%, 24.81\%])$$
