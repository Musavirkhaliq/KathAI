# Exercises: Lecture 08 (Speech Corpora Collection)

---

## Exercise 1: Quantile SNR Calculation
An audio recording has the following energy percentiles:
- 10th percentile energy ($P_{10}$): $2.4 \times 10^{-5}$
- 95th percentile energy ($P_{95}$): $1.8 \times 10^{-2}$
1. Calculate the estimated SNR in decibels.
2. Based on the 12 dB threshold, should this recording be included in the clean training manifest?

---

## Exercise 2: Clipping Distortion Analysis
A mobile recording of 10 seconds (160,000 samples) contains 4,800 samples that hit full-scale ceiling ($\pm 1.0$).
1. Calculate the percentage of clipped samples.
2. Explain the mathematical consequence of clipping in the frequency domain.
