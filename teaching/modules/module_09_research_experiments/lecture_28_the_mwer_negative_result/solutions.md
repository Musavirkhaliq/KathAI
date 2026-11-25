# Solutions: Lecture 28 (The MWER Negative Result)

---

## Solution to Exercise 1
1. Expected word error:

$$\bar{\mathcal{W}} = (0.60 \times 2) + (0.30 \times 0) + (0.10 \times 5) = 1.20 + 0.0 + 0.50 = \mathbf{1.70 \text{ errors}}$$

2. Gradient factors $(\mathcal{W}_i - \bar{\mathcal{W}})$:
   - $H_1$: $2 - 1.70 = \mathbf{+0.30}$
   - $H_2$: $0 - 1.70 = \mathbf{-1.70}$
   - $H_3$: $5 - 1.70 = \mathbf{+3.30}$
3. Notice that for $H_2$, the error is below average, so $(\mathcal{W}_2 - \bar{\mathcal{W}}) = -1.70 < 0$. Under gradient descent ($-\eta \nabla$), this negative sign increases the logit of $H_2$, encouraging the zero-error hypothesis.

---

## Solution to Exercise 2
1. Frame-level Maximum Likelihood CTC loss acts as a universal acoustic regularizer: it forces every time frame to emit the correct phonetic class regardless of whether that class affects the final word boundary.
2. MWER evaluates only the discrete string edits of the top beam candidates. When trained on clean speech, it allows the network to overfit to clean acoustic trajectories while ignoring acoustic subtleties in noisy frames, degrading performance when tested on reverberant audio.
