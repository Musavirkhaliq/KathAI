# Solutions: Lecture 08 (Speech Corpora Collection)

---

## Solution to Exercise 1
1. Estimated SNR:

$$\text{SNR} = 10 \log_{10}\left( \frac{P_{95}}{P_{10}} \right) = 10 \log_{10}\left( \frac{1.8 \times 10^{-2}}{2.4 \times 10^{-5}} \right) = 10 \log_{10}(750) \approx 10 \times 2.875 = \mathbf{28.75 \text{ dB}}$$

2. Since $28.75 \text{ dB} \ge 12.0 \text{ dB}$, the recording is accepted into the clean training set.

---

## Solution to Exercise 2
1. Percentage clipped:

$$\frac{4,800}{160,000} \times 100 = \mathbf{3.0\%}$$

2. Hard digital clipping abruptly truncates the peaks of the continuous waveform, creating sharp non-linear discontinuities. In the frequency domain, this acts like convolution with high-frequency square wave harmonics, generating spurious broadband high-frequency noise that masks subtle consonant formants.
