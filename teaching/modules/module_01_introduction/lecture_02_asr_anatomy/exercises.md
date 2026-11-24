# Exercises: Lecture 02 (The Anatomy of Modern ASR Pipelines)

---

## Exercise 1: Theoretical Analysis of the CTC Collapse Operator
Let target word be $\mathbf{y} = (c_1, c_2)$ with $c_1 \neq c_2$.
Suppose $T = 3$, and the alphabet contains only $\{c_1, c_2, \epsilon\}$.
1. Enumerate all valid alignment paths $\boldsymbol{\pi} \in (\Omega')^3$ such that $\mathcal{B}(\boldsymbol{\pi}) = \mathbf{y}$.
2. Now let $\mathbf{y} = (c_1, c_1)$ (two consecutive identical characters).
   Enumerate all valid alignment paths $\boldsymbol{\pi} \in (\Omega')^3$ such that $\mathcal{B}(\boldsymbol{\pi}) = \mathbf{y}$.
3. Prove why emitting two consecutive identical characters requires an intervening blank symbol $\epsilon$ when $T = 3$.

---

## Exercise 2: Receptive Field and Frame Calculation
The XLS-R feature extractor comprises 7 1D convolutional layers with strides:
$$S = (5, 2, 2, 2, 2, 2, 2)$$
1. Calculate the total temporal downsampling factor $D$.
2. For an audio clip of 12.4 seconds sampled at 16,000 Hz, calculate the exact number of acoustic frames $T_\text{frames}$ passed to the Transformer backbone.
3. If the audio sampling rate is erroneously 22,050 Hz and not resampled, how many frames are produced, and what is the temporal duration mismatch in the CTC alignment?

---

## Exercise 3: Shallow Fusion Sensitivity
Given candidate hypotheses with CTC log-probabilities and LM log-probabilities:
- Hypothesis $H_1$: $\log P_\text{CTC} = -4.2$, $\log P_\text{LM} = -8.5$, Length $|H_1| = 3$ words
- Hypothesis $H_2$: $\log P_\text{CTC} = -5.8$, $\log P_\text{LM} = -3.1$, Length $|H_2| = 3$ words

1. For $\alpha = 0.20, \beta = 0.0$, which hypothesis wins?
2. Find the critical threshold $\alpha^*$ at which the decoder switches preferences from $H_1$ to $H_2$.
