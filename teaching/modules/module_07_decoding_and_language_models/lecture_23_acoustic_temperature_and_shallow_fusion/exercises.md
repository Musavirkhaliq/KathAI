# Exercises: Lecture 23 (Acoustic Temperature and Shallow Fusion)

---

## Exercise 1: Temperature Calibration Derivative
Let $y_k(T) = \frac{e^{z_k / T}}{\sum_j e^{z_j / T}}$.
1. Derive the derivative of probability $y_k$ with respect to temperature $T$: $\frac{\partial y_k}{\partial T}$.
2. Show that if $z_k > \bar{z}$ (where $\bar{z} = \sum_j y_j z_j$), increasing temperature decreases the probability of token $k$.

---

## Exercise 2: Word Insertion Penalty Threshold
Given two hypotheses for an ambiguous utterance:
- Hypothesis 1: 2 words, $\log P_{\text{AM}} = -8.0, \log P_{\text{LM}} = -4.0$
- Hypothesis 2: 3 words, $\log P_{\text{AM}} = -10.5, \log P_{\text{LM}} = -4.5$
Let $\alpha = 0.40$.
Calculate the critical threshold $\beta^*$ above which Hypothesis 2 (the longer hypothesis) is selected over Hypothesis 1.
