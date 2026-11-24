# Quiz: Lecture 03 (Sound Waves, Speech Physics, and Digital Sampling)

---

### Question 1:
According to the Nyquist-Shannon sampling theorem, what is the minimum sampling rate required to perfectly reconstruct a speech signal band-limited to 7.5 kHz?
- [A] 7.5 kHz
- [B] 15.0 kHz
- [C] 16.0 kHz
- [D] 30.0 kHz

**Correct Answer**: **[B]**  
*Explanation*: $f_s \ge 2 B = 2 \times 7.5 \text{ kHz} = 15.0 \text{ kHz}$. (In practice, 16 kHz is used to provide a transition band for anti-aliasing filters).

---

### Question 2:
What spectral effect occurs when a continuous signal is multiplied by a periodic Dirac impulse train in the time domain?
- [A] The spectrum is attenuated by 6 dB per octave.
- [B] The baseband spectrum is replicated infinitely at integer multiples of the sampling frequency.
- [C] All phase information is destroyed.
- [D] The signal converts into discrete Fourier series harmonics.

**Correct Answer**: **[B]**  
*Explanation*: Multiplication in time corresponds to convolution in frequency with a Dirac comb, yielding $X_s(f) = f_s \sum_k X(f - k f_s)$.

---

### Question 3:
What physical property of the human vocal tract is primarily responsible for the formant frequencies observed in vowel acoustics?
- [A] The mass and stiffness of the vocal folds.
- [B] The subglottal lung pressure.
- [C] The acoustic resonances determined by the cross-sectional shape and length of the pharyngeal and oral cavities.
- [D] The viscosity of air molecules inside the nasal cavity.

**Correct Answer**: **[C]**  
*Explanation*: Formants are acoustic resonances of the vocal tract tube, governed by the positioning of the tongue, jaw, and lips.

---

### Question 4:
What is the theoretical Signal-to-Quantization-Noise Ratio (SQNR) for a 16-bit linear PCM audio signal?
- [A] ~48 dB
- [B] ~72 dB
- [C] ~98 dB
- [D] ~128 dB

**Correct Answer**: **[C]**  
*Explanation*: $\text{SQNR} \approx 6.02 \times 16 + 1.76 = 96.32 + 1.76 = 98.08 \text{ dB}$.

---

### Question 5:
If a sound wave contains energy at 10 kHz and is sampled at 16 kHz without an analog anti-aliasing low-pass filter, at what frequency will that energy appear in the discrete signal?
- [A] 2 kHz
- [B] 6 kHz
- [C] 8 kHz
- [D] 10 kHz

**Correct Answer**: **[B]**  
*Explanation*: $f_{\text{alias}} = |f - f_s| = |10 - 16| = 6 \text{ kHz}$.
