# Solutions: Lecture 16 (Building the Kashmiri CTC Acoustic Model)

---

## Solution to Exercise 1
1. Transformer layers: $\frac{302.2}{317} \times 100 = \mathbf{95.3\%}$.
2. Classification head: $\frac{0.048}{317} \times 100 = \mathbf{0.015\%}$.
3. The projection head has negligible capacity ($< 0.02\%$), meaning catastrophic overfitting cannot occur in the linear head itself; overfitting occurs when the 302M parameters in the Transformer backbone distort universal speech features.

---

## Solution to Exercise 2
1. For $\mathbf{b} = \mathbf{0}$:
   $$P(\epsilon) = \frac{e^0}{\sum_{i=1}^{47} e^0} = \frac{1}{47} \approx \mathbf{0.0213 \text{ (2.13\%)}}$$
2. For $b_0 = 3.0$ and $b_{1:46} = 0$:
   $$e^{b_0} = e^{3.0} \approx 20.0855$$
   $$\sum_{k=1}^{46} e^{b_k} = 46 \times 1.0 = 46.0$$
   $$P(\epsilon) = \frac{20.0855}{20.0855 + 46.0} = \frac{20.0855}{66.0855} \approx \mathbf{0.3039 \text{ (30.4\%)}}$$
   (Setting $b_0 = 5.0$ yields $P(\epsilon) \approx 76.3\%$).
