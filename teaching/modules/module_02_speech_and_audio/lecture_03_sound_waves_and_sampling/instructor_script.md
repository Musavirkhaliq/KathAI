# Instructor Delivery Script: Lecture 03 (Sound Waves, Speech Physics, and Digital Sampling)

## Timetable & Pedagogical Flow (90 Minutes Total)

```
00:00 - 00:20 | Physical Reality: Pressure Waves and the Source-Filter Model
00:20 - 00:45 | Mathematical Proof: The Dirac Comb and Spectral Replication
00:45 - 01:10 | Nyquist Theorem and the Aliasing Demonstration
01:10 - 01:25 | Quantization Noise, SQNR Derivation (6.02B + 1.76 dB)
01:25 - 01:30 | Takeaways & Link to the 22.05 kHz Bolbosh Failure
```

---

## Detailed Delivery Script

### 00:00 - 00:20 | Physical Reality
- **Opening Hook**: Tap the desk with a pen.
  > "What is that sound? A compression wave in air molecules. When we train a 300-million-parameter neural network, it never sees words; it sees floating point numbers representing millipascal pressure changes.
  > Today, we bridge physics to digital arrays. If your sampling is mathematically defective, your downstream Transformer is learning noise."
- Draw the **Source-Filter Model** on the Center Board.
  - Show the glottal source vibrating at $F_0 \approx 120 \text{ Hz}$ for males, $210 \text{ Hz}$ for females.
  - Show the vocal tract tube modulating the harmonics into formants $F_1, F_2, F_3$.
  - Highlight Kashmiri central vowels: the tongue retracts to the center of the oral cavity, creating subtle $F_2$ shifts between $1300\text{ Hz}$ and $1600\text{ Hz}$.

### 00:20 - 00:45 | Mathematical Derivation of Sampling
- **Board Action**: Write the Fourier transform of the Dirac comb.

$$\mathcal{F}\left\{\sum_{n} \delta(t - n T_s)\right\} = \frac{1}{T_s} \sum_k \delta\left(f - \frac{k}{T_s}\right)$$

- Show that multiplication in time is convolution in frequency:

$$X_s(f) = X(f) * P(f) = f_s \sum_k X(f - k f_s)$$

- Draw the baseband spectrum $X(f)$ centered at 0.
- Draw the spectral copies at $\pm f_s, \pm 2f_s$.
- Move $f_s$ closer to 0 until the copies overlap.
- Highlight the overlap in red chalk: **THIS IS ALIASING**.

### 00:45 - 01:10 | Aliasing in ASR
- Ask students: *"If a Kashmiri speaker produces a voiceless dental sibilant /s/ at 9,000 Hz, and our ADC operates at 16,000 Hz, what does the model hear?"*
- Walk through the calculation:

$$f_{\text{alias}} = |9000 - 16000| = 7000 \text{ Hz}$$

- The 9 kHz /s/ folds down into 7 kHz, directly over the /ʃ/ (sh) acoustic region!

### 01:10 - 01:30 | Quantization and Wrap-up
- Derive the $6.02 B + 1.76 \text{ dB}$ formula.
- Conclude with a teaser:
  > "Next lecture, we will examine what happens when an engineer downsamples 22.05 kHz speech to 16 kHz without an anti-aliasing low-pass filter. It corrupted 18 hours of Kashmiri speech in our repository."
