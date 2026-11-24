# Lecture 05: Fourier Analysis, Spectrograms, and Mel Filterbanks

---

## 1. The Mathematical Bridge: From Time Domain to Frequency Domain

A speech waveform $x(t)$ represents pressure amplitude fluctuations over time. However, the human auditory system does not perceive raw pressure directly; the cochlea performs a biological spectral decomposition into distinct frequency bands along the basilar membrane.

The **Continuous-Time Fourier Transform (CTFT)** decomposes a continuous time-domain signal $x(t) \in L^2(\mathbb{R})$ into a continuous continuum of complex sinusoids:

$$X(f) = \int_{-\infty}^\infty x(t) e^{-j 2\pi f t} dt$$

The inverse transform reconstructs the time-domain waveform:
$$x(t) = \int_{-\infty}^\infty X(f) e^{j 2\pi f t} df$$

### The Discrete Fourier Transform (DFT)
For a discrete sequence of $N$ samples $x[n] = (x[0], x[1], \dots, x[N-1])$:

$$X[k] = \sum_{n=0}^{N-1} x[n] e^{-j \frac{2\pi}{N} k n}, \quad k = 0, 1, \dots, N-1$$

The frequency corresponding to bin index $k$ is:
$$f_k = \frac{k f_s}{N}$$
The **Fast Fourier Transform (FFT)**, discovered by Cooley and Tukey, evaluates this summation in $\mathcal{O}(N \log N)$ operations rather than $\mathcal{O}(N^2)$.

---

## 2. Short-Time Fourier Transform (STFT) and the Uncertainty Principle

Speech is non-stationary: its spectral properties change dynamically as the articulators move. However, over short intervals of $20 - 30 \text{ ms}$, speech is **quasi-stationary**.

The **Short-Time Fourier Transform (STFT)** analyzes speech by sliding an analysis window $w[n]$ across the sequence:

$$X[m, k] = \sum_{n=0}^{N-1} x[n + m H] w[n] e^{-j \frac{2\pi}{N} k n}$$

where:
- $m$ is the time frame index.
- $H$ is the hop size (frame shift, typically 10 ms = 160 samples at 16 kHz).
- $w[n]$ is a tapered window function (e.g., Hann or Hamming) designed to eliminate spectral leakage from boundary discontinuities.

### The Gabor-Heisenberg Uncertainty Principle
Time resolution $\Delta t$ and frequency resolution $\Delta f$ cannot be simultaneously minimized:

$$\Delta t \cdot \Delta f \ge \frac{1}{4\pi}$$

- **Narrowband Spectrogram (Large $N$, e.g., 50 ms)**: High frequency resolution (resolves individual glottal harmonics), poor time resolution (smears rapid plosive bursts).
- **Wideband Spectrogram (Small $N$, e.g., 20 ms)**: High time resolution (resolves vertical glottal pulses and plosive closures), poor frequency resolution (smears harmonics into formant bands).
- In ASR, **wideband analysis ($20-25 \text{ ms}$ window, $10 \text{ ms}$ hop)** is standard to capture formant transitions accurately.

---

## 3. Window Functions and Spectral Leakage

Rectangular windowing truncates the signal abruptly, equivalent to convolving the true spectrum with a sinc function:
$$W_{\text{rect}}(f) = \frac{\sin(\pi f N)}{\pi f}$$
The high sidelobes ($-13 \text{ dB}$) cause **spectral leakage**, where intense low-frequency energy (e.g., $F_0$) contaminates weak high-frequency formants.

### The Hann and Hamming Windows
To suppress sidelobes, tapered cosine windows are applied:
- **Hann Window**:
  $$w_{\text{Hann}}[n] = 0.5 - 0.5 \cos\left( \frac{2\pi n}{N-1} \right)$$
  First sidelobe attenuation: $-31.5 \text{ dB}$; roll-off: $-18 \text{ dB/octave}$.
- **Hamming Window**:
  $$w_{\text{Hamming}}[n] = 0.54 - 0.46 \cos\left( \frac{2\pi n}{N-1} \right)$$
  First sidelobe attenuation: $-42.7 \text{ dB}$; roll-off: $-6 \text{ dB/octave}$.

---

## 4. Psychoacoustics and Mel Filterbanks

Human pitch perception is non-linear: our ears have exquisite frequency resolution below 1 kHz, but logarithmic resolution above 1 kHz.

The **Mel Scale** maps physical frequency $f$ (Hz) to perceived pitch $m$ (mels):
$$m = 2595 \log_{10}\left( 1 + \frac{f}{700} \right) = 1127 \ln\left( 1 + \frac{f}{700} \right)$$

To construct an 80-channel Mel filterbank:
1. Define 82 points equally spaced on the mel scale between $0 \text{ Hz}$ and $f_s / 2 = 8000 \text{ Hz}$.
2. Convert mel points back to linear frequencies $f_m$.
3. Construct triangular overlapping bandpass filters $H_m[k]$:
   $$H_m[k] = \begin{cases} 
   0 & \text{if } k < f_{m-1} \\
   \frac{k - f_{m-1}}{f_m - f_{m-1}} & \text{if } f_{m-1} \le k \le f_m \\
   \frac{f_{m+1} - k}{f_{m+1} - f_m} & \text{if } f_m \le k \le f_{m+1} \\
   0 & \text{if } k > f_{m+1}
   \end{cases}$$

The **Log-Mel Filterbank Energy** $S[m]$ is computed by integrating the power spectrum and taking the logarithm:
$$S[m] = \ln \left( \sum_{k=0}^{N/2} |X[k]|^2 H_m[k] \right)$$

---

## 5. Raw Waveforms vs. Filterbanks: The Self-Supervised Revolution

For thirty years, Log-Mel filterbanks (and MFCCs) were the universal input representation for ASR. However, hand-engineered filterbanks discard two critical sources of acoustic information:
1. **Phase Spectrum**: Discarded completely during magnitude computation $|X[k]|^2$.
2. **Fixed Frequency Banding**: The rigid triangular filters prevent the network from learning non-linear auditory representations tailored to specific linguistic distinctions (such as Kashmiri central vowel transitions).

In modern self-supervised architectures (e.g., **XLS-R**), the acoustic model operates directly on **raw 1D waveforms**:
- A 7-layer 1D convolutional feature extractor learns parameterized temporal filters directly from backpropagation.
- As we prove in Lecture 11, the first convolutional layer acts as a data-driven filterbank with learnable center frequencies and bandwidths.
