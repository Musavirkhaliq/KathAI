# Solutions: Lecture 02 (The Anatomy of Modern ASR Pipelines)

---

## Solution to Exercise 1

### Part 1: $\mathbf{y} = (c_1, c_2)$ with $c_1 \neq c_2$, $T = 3$
The possible alignments of length 3 that collapse to $(c_1, c_2)$ are:
1. $(c_1, c_1, c_2) \to \text{dedup } (c_1, c_2)$
2. $(c_1, c_2, c_2) \to \text{dedup } (c_1, c_2)$
3. $(c_1, \epsilon, c_2) \to \text{dedup } (c_1, \epsilon, c_2) \to (c_1, c_2)$
4. $(\epsilon, c_1, c_2) \to (c_1, c_2)$
5. $(c_1, c_2, \epsilon) \to (c_1, c_2)$

Total valid paths = **5**.

### Part 2: $\mathbf{y} = (c_1, c_1)$ (Identical Characters), $T = 3$
If two consecutive tokens are identical, they cannot be adjacent in time without collapsing into a single token under $\mathcal{B}_{\text{dedup}}$.
Thus, they **must** be separated by at least one non-$c_1$ token (i.e., $\epsilon$):
- Alignment: $(c_1, \epsilon, c_1) \to \mathcal{B} = (c_1, c_1)$.
Any other arrangement fails:
- $(c_1, c_1, \epsilon) \to (c_1)$
- $(\epsilon, c_1, c_1) \to (c_1)$
- $(c_1, c_1, c_1) \to (c_1)$

Total valid paths = **1** (namely, $(c_1, \epsilon, c_1)$).

### Part 3: Proof
Under the definition of $\mathcal{B}$, any consecutive repetition of $c_1$ without an intermediate $\epsilon$ yields $\mathcal{B}(c_1, c_1) = c_1$. Hence, to preserve two distinct emissions of $c_1$ within length $T=3$, the sequence must contain $c_1$ at positions $t_1$ and $t_2$ with $t_2 > t_1 + 1$ and $\pi_t = \epsilon$ for all $t_1 < t < t_2$. For $T=3$, the only possible indices are $t_1=1, t_2=3$, forcing $\pi_2 = \epsilon$. $\blacksquare$

---

## Solution to Exercise 2

1. Total downsampling factor $D$:
   $$D = \prod_{i=1}^7 S_i = 5 \times 2 \times 2 \times 2 \times 2 \times 2 \times 2 = 5 \times 64 = 320$$

2. Total samples $N = 12.4 \times 16,000 = 198,400$.
   Using the exact convolutional receptive field formula:
   $$T_\text{frames} \approx \left\lfloor \frac{198400}{320} \right\rfloor = 620 \text{ frames}$$

3. At 22,050 Hz without resampling:
   $$N_{22k} = 12.4 \times 22,050 = 273,420 \text{ samples}$$
   $$T_\text{frames} \approx \left\lfloor \frac{273420}{320} \right\rfloor = 854 \text{ frames}$$
   This is an artificial **37.7% inflation** in temporal frame count, causing phonemes to stretch across more frames than the model was trained to observe, leading to rampant deletion and insertion errors.

---

## Solution to Exercise 3

The shallow fusion score function is:
$$\mathcal{S}(H) = \log P_\text{CTC}(H) + \alpha \log P_\text{LM}(H)$$

1. For $\alpha = 0.20$:
   $$\mathcal{S}(H_1) = -4.2 + 0.20 \times (-8.5) = -4.2 - 1.7 = -5.90$$
   $$\mathcal{S}(H_2) = -5.8 + 0.20 \times (-3.1) = -5.8 - 0.62 = -6.42$$
   Since $-5.90 > -6.42$, **Hypothesis $H_1$ wins**.

2. Critical threshold $\alpha^*$:
   $$\mathcal{S}(H_1) = \mathcal{S}(H_2)$$
   $$-4.2 + \alpha^* (-8.5) = -5.8 + \alpha^* (-3.1)$$
   $$-4.2 + 5.8 = \alpha^* (8.5 - 3.1)$$
   $$1.6 = \alpha^* (5.4) \implies \alpha^* = \frac{1.6}{5.4} \approx 0.2963$$
   For $\alpha > 0.296$, the language model dominates and selects $H_2$.
