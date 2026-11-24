# Solutions: Lecture 19 (Empirical Scaling Analysis)

---

## Solution to Exercise 1
1. For $H = 500 \text{ hours}$:
   $$\text{WER}(500) = 18.5 + 85.0 \times (500)^{-0.42} = 18.5 + 85.0 \times 0.0739 = 18.5 + 6.28 = \mathbf{24.78\%}$$
2. For $\text{WER} = 20.0\%$:
   $$20.0 = 18.5 + 85.0 \cdot H^{-0.42} \implies 1.5 = 85.0 \cdot H^{-0.42}$$
   $$H^{-0.42} = \frac{1.5}{85.0} \approx 0.01765 \implies H = (0.01765)^{-1/0.42} \approx (0.01765)^{-2.381} \approx \mathbf{14,750 \text{ hours}}$$
3. Achieving $15.0\%$ is impossible because the asymptotic irreducible error floor is $\text{WER}_\infty = 18.5\%$. Even with infinite acoustic training data ($H \to \infty$), acoustic ambiguities prevent error from dropping below 18.5% without language model decoding.

---

## Solution to Exercise 2
1. Reverberation introduces room impulse response convolutions ($y[n] = x[n] * h_{\text{room}}[n]$) that smear phonemes across adjacent time frames.
2. Spontaneous sermon speech contains non-standard grammar, overlapping speech, crowd murmurs, and loudspeaker clipping.
3. Unless the training data contains substantial proportions of reverberant audio, the model's capacity scales primarily on clean speech features.
