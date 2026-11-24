# Lecture 04: Resampling Algorithms and the 22.05 kHz Kaiser Bug

---

## 1. The Mathematical Problem of Arbitrary-Ratio Resampling

In digital speech pipelines, acoustic models (e.g., Wav2Vec2, XLS-R, Whisper) expect audio tensors sampled at a fixed standard rate, universally $f_{\text{target}} = 16,000 \text{ Hz}$.

However, real-world corpora contain heterogeneous sampling frequencies:
- Broadcast radio / CD audio: $44,100 \text{ Hz}$
- Studio video recordings: $48,000 \text{ Hz}$
- Legacy telephone speech: $8,000 \text{ Hz}$
- Legacy multimedia recordings (e.g., Bolbosh corpus): $22,050 \text{ Hz}$

Resampling changes the sampling rate from $f_1$ to $f_2$ by a rational conversion factor:
$$\frac{f_2}{f_1} = \frac{L}{M}$$
where $L$ is the upsampling (interpolation) factor and $M$ is the downsampling (decimation) factor, with $\gcd(L, M) = 1$.

### The 22.05 kHz to 16 kHz Rational Factor:
$$\frac{16,000}{22,050} = \frac{1600}{2205} = \frac{320}{441}$$

This non-integer, large-factor ratio ($L=320, M=441$) makes naive implementations computationally prohibitive and prone to numerical filter defects.

---

## 2. Mathematical Algorithms for Resampling

### A. Linear Interpolation (Naive & Destructive)
Computes intermediate points via simple straight-line estimation:
$$x[t] = (1 - \alpha) x[n] + \alpha x[n+1], \quad \alpha = t - n$$
- **Frequency Response**: Sinc-squared $H(f) = \text{sinc}^2(f / f_s)$.
- **Flaws**: Produces severe high-frequency roll-off (spectral droop) in the passband and fails to attenuate image frequencies in the stopband, causing rampant aliasing.

### B. Ideal Bandlimited Sinc Interpolation (Whittaker-Shannon)
Continuous reconstruction followed by resampling at $t_m = m T_2$:
$$x(t_m) = \sum_{n=-\infty}^\infty x[n] \text{sinc}\left( \frac{t_m - n T_1}{T_1} \right)$$
- Requires infinite support ($n \in [-\infty, +\infty]$), making it physically unrealizable.

### C. Windowed Sinc Interpolation (Kaiser Windowed Polyphase)
Truncates the infinite sinc kernel using a parameterized window $w[n]$:
$$h[n] = \text{sinc}\left(\frac{n}{L}\right) \cdot w[n]$$

The **Kaiser window** is an optimal approximation to the prolate spheroidal wave functions, maximizing energy concentration in the main lobe:
$$w[n] = \frac{I_0\left( \beta \sqrt{1 - \left(\frac{2n}{N-1} - 1\right)^2} \right)}{I_0(\beta)}$$
where $I_0(\cdot)$ is the zeroth-order modified Bessel function of the first kind, and $\beta$ controls the stopband attenuation trade-off.

---

## 3. Forensic Case Study: The 22.05 kHz Bolbosh Defect in EduCasheer

During Stage 03 of the EduCasheer project, the **Bolbosh Kashmiri Speech Corpus** (comprising 18.4 hours of native Kashmiri speech) was ingested. The audio was originally recorded at $22,050 \text{ Hz}$.

### The Implementation Flaw
The original preprocessing script (`stages/stage_03/scripts/preprocess_audio.py`) utilized a legacy Python audio library configured with an improper low-pass filter cutoff and inadequate Kaiser window parameters:
1. The anti-aliasing low-pass filter had its cutoff set too low ($\approx 6,200 \text{ Hz}$ instead of the Nyquist $8,000 \text{ Hz}$).
2. The Kaiser parameter $\beta$ was set to an extreme value ($\beta = 14.7$), resulting in a massive transition band ($> 2,500 \text{ Hz}$).
3. **The Result**: The resampling operation wiped out all acoustic energy between $5.8 \text{ kHz}$ and $8.0 \text{ kHz}$, creating a massive spectral void.

### The Acoustic Consequence for Kashmiri
Kashmiri has a rich inventory of coronal and palatal fricatives and affricates:
- /s/ (dental sibilant)
- /ʃ/ (palato-alveolar sibilant)
- /t͡s/ (dental affricate)
- /t͡sʰ/ (aspirated dental affricate)

These phonemes possess critical acoustic energy in the $5.5 - 8.0 \text{ kHz}$ band.
The defective resampler eliminated this energy. Models trained on this data (Early B-series) suffered an catastrophic **14.2% absolute increase in Character Error Rate (CER)** on sibilant-rich test sentences!

---

## 4. The Certified Solution: Polyphase Resampling with Calibrated Kaiser Filter

In Stage 26, the audio pipeline was rebuilt from scratch in `src/utils/audio.py` using `torchaudio.transforms.Resample`:

```python
resampler = torchaudio.transforms.Resample(
    orig_freq=22050,
    new_freq=16000,
    resampling_method="sinc_interp_kaiser",
    lowpass_filter_width=16,
    rolloff=0.94,
    beta=8.5555
)
```

### Certified Parameters:
1. **`lowpass_filter_width=16`**: Provides 16 zero-crossings on each side of the sinc peak, ensuring sharp transition bands without time-domain ringing artifacts.
2. **`rolloff=0.94`**: Sets the filter cutoff frequency to $0.94 \times \frac{f_s}{2} = 0.94 \times 8000 = 7,520 \text{ Hz}$, preserving Kashmiri fricatives while safely suppressing aliasing components above 8,000 Hz.
3. **`beta=8.5555`**: Guarantees stopband attenuation exceeding $80 \text{ dB}$, rendering aliasing noise completely imperceptible to the acoustic model.
