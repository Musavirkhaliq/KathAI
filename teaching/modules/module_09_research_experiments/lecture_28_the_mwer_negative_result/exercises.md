# Exercises: Lecture 28 (The MWER Negative Result)

---

## Exercise 1: Expected Error Calculation
An acoustic model generates an $N$-best list of 3 candidate hypotheses:
- Hypothesis 1: Probability $P_1 = 0.60$, Edit errors $\mathcal{W}_1 = 2$ words
- Hypothesis 2: Probability $P_2 = 0.30$, Edit errors $\mathcal{W}_2 = 0$ words
- Hypothesis 3: Probability $P_3 = 0.10$, Edit errors $\mathcal{W}_3 = 5$ words
1. Calculate the baseline expected word error $\bar{\mathcal{W}}$.
2. Calculate the risk gradient factor $( \mathcal{W}_i - \bar{\mathcal{W}} )$ for each hypothesis.
3. Which hypothesis receives a positive gradient update encouraging its probability to increase?

---

## Exercise 2: The Generalization Trade-off
Explain why optimizing Maximum Likelihood on raw frames preserves multi-domain generalization better than optimizing Minimum Word Error Rate on clean in-domain text.
