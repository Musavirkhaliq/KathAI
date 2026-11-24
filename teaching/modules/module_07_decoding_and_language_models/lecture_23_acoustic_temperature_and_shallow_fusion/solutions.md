# Solutions: Lecture 23 (Acoustic Temperature and Shallow Fusion)

---

## Solution to Exercise 1
1. Let $S = \sum_j e^{z_j / T}$.
   $$\frac{\partial y_k}{\partial T} = \frac{\frac{\partial}{\partial T}(e^{z_k / T}) S - e^{z_k / T} \frac{\partial S}{\partial T}}{S^2}$$
   Using $\frac{\partial}{\partial T} e^{z_k / T} = e^{z_k / T} \left( -\frac{z_k}{T^2} \right)$:
   $$\frac{\partial y_k}{\partial T} = \frac{-z_k e^{z_k / T} S - e^{z_k / T} \sum_j (-z_j e^{z_j / T})}{T^2 S^2} = -\frac{y_k}{T^2} \left( z_k - \sum_j y_j z_j \right) = \mathbf{-\frac{y_k}{T^2} (z_k - \bar{z})}$$
2. If $z_k > \bar{z}$, then $(z_k - \bar{z}) > 0$. Since $y_k > 0$ and $T^2 > 0$, the derivative $\frac{\partial y_k}{\partial T} < 0$. Thus, increasing temperature systematically reduces the probability of above-average (overconfident) logits. $\blacksquare$

---

## Solution to Exercise 2
Scores:
$$\mathcal{S}(H_1) = -8.0 + 0.40(-4.0) + 2\beta = -8.0 - 1.6 + 2\beta = -9.6 + 2\beta$$
$$\mathcal{S}(H_2) = -10.5 + 0.40(-4.5) + 3\beta = -10.5 - 1.8 + 3\beta = -12.3 + 3\beta$$
For $H_2$ to win:
$$\mathcal{S}(H_2) > \mathcal{S}(H_1) \implies -12.3 + 3\beta > -9.6 + 2\beta$$
$$\beta > -9.6 + 12.3 \implies \mathbf{\beta^* = 2.70}$$
For $\beta > 2.70$, the insertion bonus overcomes the acoustic penalty and selects the longer word sequence.
