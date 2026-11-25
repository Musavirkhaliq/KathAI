# Mathematical Foundations: WADA-SNR and Acoustic Energy Estimation

---

## 1. Mathematical Formulation of WADA-SNR

WADA-SNR (Waveform Amplitude Distribution Analysis) estimates SNR from a single-channel speech signal without requiring explicit speech/silence segmentation.

Let clean speech amplitude $s$ follow a **Gamma distribution** with shape parameter $\gamma = 0.5$:

$$P_s(s) = \frac{1}{\sqrt{2\pi \sigma_s s}} e^{-\frac{s}{2\sigma_s}}$$

and let additive background noise $n$ follow a **Gaussian distribution**:

$$P_n(n) = \frac{1}{\sqrt{2\pi \sigma_n^2}} e^{-\frac{n^2}{2\sigma_n^2}}$$

The observed signal is $x = s + n$. The WADA algorithm computes the **ratio of the arithmetic mean of the absolute value to the geometric mean**:

$$\Psi(x) = \frac{\frac{1}{N} \sum_{i=1}^N |x_i|}{\left( \prod_{i=1}^N |x_i| \right)^{1/N}}$$

Since $\Psi(x)$ is a strictly monotonic function of the ratio $\frac{\sigma_s^2}{\sigma_n^2}$, the true SNR in decibels is obtained by inverting the pre-computed calibration function:

$$\text{SNR}_{\text{dB}} = f_{\text{calib}}^{-1}(\Psi(x))$$
