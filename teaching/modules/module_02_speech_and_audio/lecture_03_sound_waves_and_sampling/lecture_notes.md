# Lecture 03: Sound Waves, Physics of Speech, and Digital Sampling

---

## 1. The Physics of Acoustic Propagation

Sound is a mechanical disturbance propagating through an elastic medium (such as air) as a longitudinal pressure wave. When a human speaks, vocal fold vibration and turbulent airflow create localized fluctuations in ambient atmospheric pressure $P_0 \approx 101.325 \text{ kPa}$.

The instantaneous acoustic sound pressure $p(t)$ is defined as the deviation from ambient pressure:

$$p_{\text{total}}(t) = P_0 + p(t)$$

The physical wave equation governing acoustic propagation in three-dimensional lossless air is:

$$\nabla^2 p - \frac{1}{c^2} \frac{\partial^2 p}{\partial t^2} = 0$$

where $c \approx 343 \text{ m/s}$ is the speed of sound in dry air at $20^\circ\text{C}$.

---

## 2. The Source-Filter Model of Speech Production

Speech production is modeled through Gunnar Fant's **Source-Filter Theory**:

$$S(f) = E(f) \cdot H(f) \cdot R(f)$$

```
[ Glottal Source E(f) ] ──▶ [ Vocal Tract Filter H(f) ] ──▶ [ Lip Radiation R(f) ] ──▶ Radiated Speech S(f)
```

1. **Glottal Source $E(f)$**:
   - For voiced sounds (vowels, nasals), vocal cords open and close periodically at fundamental frequency $F_0$.
   - The glottal pulse has a spectral roll-off of $-12 \text{ dB/octave}$.
2. **Vocal Tract Filter $H(f)$**:
   - The pharynx, oral cavity, and nasal cavity act as an acoustic tube resonator with varying cross-sectional area.
   - Resonances of this tube are called **formants** ($F_1, F_2, F_3, \dots$).
   - Formant frequencies determine vowel identity:
     - $F_1$ correlates inversely with vowel height (tongue height).
     - $F_2$ correlates with vowel backness (tongue advancement).
3. **Lip Radiation $R(f)$**:
   - Radiation into free space acts as a high-pass spatial filter with a $+6 \text{ dB/octave}$ boost.
   - The net speech spectrum rolls off at approximately $-6 \text{ dB/octave}$.

---

## 3. The Mathematics of Continuous-to-Discrete Sampling

An analog microphone converts pressure fluctuations $p(t)$ into a continuous voltage $v(t)$. To process this signal computationally, it must undergo **Analog-to-Digital Conversion (ADC)** consisting of two distinct operations: **sampling** in time and **quantization** in amplitude.

### The Ideal Sampling Model
Mathematically, ideal sampling is modeled as multiplication of the continuous signal $x(t)$ with a periodic Dirac delta comb:

$$p_\delta(t) = \sum_{n=-\infty}^\infty \delta(t - n T_s)$$

where $T_s$ is the sampling interval, and $f_s = \frac{1}{T_s}$ is the sampling frequency.

The sampled continuous signal is:

$$x_s(t) = x(t) \cdot p_\delta(t) = \sum_{n=-\infty}^\infty x(n T_s) \delta(t - n T_s)$$

Taking the Continuous-Time Fourier Transform (CTFT) of $x_s(t)$ reveals the frequency-domain consequence of sampling:

$$X_s(f) = X(f) * P_\delta(f) = \frac{1}{T_s} \sum_{k=-\infty}^\infty X\left(f - \frac{k}{T_s}\right) = f_s \sum_{k=-\infty}^\infty X(f - k f_s)$$

Sampling in time creates an infinite sequence of spectral replicas (images) shifted by integer multiples of the sampling frequency $f_s$.

---

## 4. The Nyquist-Shannon Sampling Theorem and Aliasing

### Theorem:
If a continuous-time signal $x(t)$ contains no frequencies higher than $B$ Hz (i.e., $X(f) = 0$ for all $|f| > B$), it can be completely and uniquely recovered from its discrete samples $x[n] = x(n T_s)$ if and only if:

$$f_s \ge 2B$$

The frequency $f_N = \frac{f_s}{2}$ is defined as the **Nyquist frequency**.

### Aliasing Catastrophe:
If the signal contains spectral components exceeding $f_N$ (i.e., $f > f_s / 2$), those frequencies fold back (alias) into the baseband:

$$f_{\text{alias}} = |f - k f_s| \le \frac{f_s}{2}$$

In speech recognition, high-frequency fricative energy (e.g., Kashmiri /s/, /ʃ/, /t͡s/) at 9 kHz aliased into an 8 kHz sampled signal masquerades as 7 kHz energy, permanently corrupting formant and phonetic trajectories.

---

## 5. Amplitude Quantization and Dynamic Range

Continuous voltage is mapped to a discrete set of $2^B$ levels for a $B$-bit linear Pulse Code Modulation (PCM) representation.

The quantization step size $\Delta$ for full-scale voltage range $[-V_{\max}, +V_{\max}]$ is:

$$\Delta = \frac{2 V_{\max}}{2^B}$$

Under the standard assumption of uniformly distributed quantization noise $e[n] \sim \mathcal{U}\left(-\frac{\Delta}{2}, \frac{\Delta}{2}\right)$:

$$\sigma_e^2 = \mathbb{E}[e^2] = \frac{1}{\Delta} \int_{-\Delta/2}^{\Delta/2} e^2 de = \frac{\Delta^2}{12}$$

For a full-scale sinusoidal signal with amplitude $V_{\max}$, signal power is:

$$\sigma_x^2 = \frac{V_{\max}^2}{2} = \frac{(2^{B-1} \Delta)^2}{2} = \frac{2^{2B-2} \Delta^2}{2} = 2^{2B-3} \Delta^2$$

The theoretical **Signal-to-Quantization-Noise Ratio (SQNR)** is:

$$\text{SQNR} = 10 \log_{10} \left( \frac{\sigma_x^2}{\sigma_e^2} \right) = 10 \log_{10} \left( \frac{2^{2B-3} \Delta^2}{\Delta^2 / 12} \right) = 10 \log_{10} \left( \frac{12 \cdot 2^{2B}}{8} \right)$$
$$\text{SQNR} \approx 6.02 B + 1.76 \text{ dB}$$

### Practical Significance:
- **16-bit Linear PCM**: Yields $\text{SQNR} \approx 98.1 \text{ dB}$, which exceeds the dynamic range of human speech in natural room acoustic environments.
- EduCasheer standardizes all audio on **16-bit 16 kHz Mono WAV**.
