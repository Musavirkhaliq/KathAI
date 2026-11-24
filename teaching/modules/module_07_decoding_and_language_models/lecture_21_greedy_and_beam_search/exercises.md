# Exercises: Lecture 21 (Greedy vs. Beam Search)

---

## Exercise 1: Prefix Path Merging
Given prefix $\ell = \text{"ک"}$, with $P_b(\ell) = 0.30$ and $P_{nb}(\ell) = 0.20$.
At time step $t$, the emissions are:
$y_\epsilon^t = 0.40$, $y_{\text{ک}}^t = 0.30$, $y_{\text{ٲ}}^t = 0.20$.
1. Calculate the updated probabilities $P_b(\text{"ک"})$ and $P_{nb}(\text{"ک"})$.
2. Calculate the updated probability $P_{nb}(\text{"کک"})$.
3. Calculate the updated probability $P_{nb}(\text{"کٲ"})$.

---

## Exercise 2: Beam Width Sensitivity
In an ASR benchmark across 500 Kashmiri sentences:
- Greedy ($W=1$): $\text{WER} = 29.8\%$
- Beam $W=10$: $\text{WER} = 26.4\%$
- Beam $W=64$: $\text{WER} = 24.8\%$
- Beam $W=200$: $\text{WER} = 24.7\%$
Plot the WER vs. Beam Width curve and identify the optimal operational sweet spot balancing accuracy and real-time latency.
