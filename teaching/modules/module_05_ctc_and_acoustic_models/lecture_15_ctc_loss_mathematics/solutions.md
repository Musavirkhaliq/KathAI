# Solutions: Lecture 15 (CTC Loss Mathematics)

---

## Solution to Exercise 1
1. Extended sequence: $\mathbf{l}' = (\epsilon, c_1, \epsilon, c_2, \epsilon)$. Length $U' = 2(2) + 1 = \mathbf{5}$.
2. For `"دَد"`, state 2 is $c_1$ and state 4 is $c_1$.
   If $\alpha_t(4)$ received a transition directly from $\alpha_{t-1}(2)$, the resulting alignment sub-path would emit $(c_1, c_1)$ without an intervening blank. Under collapse operator $\mathcal{B}$, consecutive identical characters merge: $\mathcal{B}(c_1, c_1) = c_1$, which collapses the two words into a single letter. To preserve two distinct letters, passing through the intervening blank at state 3 ($\epsilon$) is mandatory.

---

## Solution to Exercise 2
1. Logit gradient (for target character $k =$ "ب"):

$$\frac{\partial \mathcal{L}}{\partial z_k^t} = y_k^t - \gamma_t(k) = 0.95 - 0.98 = \mathbf{-0.03}$$

2. Gradient descent update rule:

$$z_{\text{new}} = z_{\text{old}} - \eta \frac{\partial \mathcal{L}}{\partial z} = z_{\text{old}} - \eta (-0.03) = z_{\text{old}} + 0.03 \eta$$

   The logit will **increase**, pushing `y_ب^t` closer to $1.0$.
