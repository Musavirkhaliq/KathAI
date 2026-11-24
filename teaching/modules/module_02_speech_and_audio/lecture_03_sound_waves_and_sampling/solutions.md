# Solutions: Lecture 03 (Sound Waves and Digital Sampling)

---

## Solution to Exercise 1

1. Fundamental period $T_0$:
   $$T_0 = \frac{1}{F_0} = \frac{1}{220 \text{ Hz}} \approx 0.004545 \text{ s} = 4.55 \text{ ms}$$

2. Harmonics of glottal source:
   $h_n = n \times F_0$.
   - $h_1 = 220 \text{ Hz}$
   - $h_2 = 440 \text{ Hz}$
   The closest harmonics to $F_1 = 340 \text{ Hz}$ are $h_1 (220 \text{ Hz}, \Delta = 120 \text{ Hz})$ and $h_2 (440 \text{ Hz}, \Delta = 100 \text{ Hz})$. $h_2$ is closest.

3. Differentiating /ɨ/ from /u/:
   - For /u/: $F_1 \approx 320 \text{ Hz}, F_2 \approx 800 \text{ Hz}$.
   - For /ɨ/: $F_1 \approx 340 \text{ Hz}, F_2 \approx 1450 \text{ Hz}$.
   If the signal is low-pass filtered at 1,000 Hz, the $F_2$ formant of /ɨ/ ($1450 \text{ Hz}$) is completely obliterated. Since both vowels have nearly identical $F_1$ ($320$ vs $340 \text{ Hz}$), the model loses the critical backness dimension and will confuse the two vowels.

---

## Solution to Exercise 2

Sampling rate $f_s = 16.0 \text{ kHz}$, Nyquist frequency $f_N = 8.0 \text{ kHz}$.
The apparent aliased frequency is given by $f_{\text{alias}} = |f - k f_s|$ such that $0 \le f_{\text{alias}} \le f_N$:

1. $f = 4.5 \text{ kHz}$:
   Since $4.5 < 8.0 \text{ kHz}$, no aliasing occurs: **$4.5 \text{ kHz}$**.

2. $f = 8.2 \text{ kHz}$:
   $|8.2 - 16.0| = \mathbf{7.8 \text{ kHz}}$.

3. $f = 13.5 \text{ kHz}$:
   $|13.5 - 16.0| = \mathbf{2.5 \text{ kHz}}$.

4. $f = 21.0 \text{ kHz}$:
   For $k=1$: $|21.0 - 16.0| = 5.0 \text{ kHz}$.
   Baseband alias is **$5.0 \text{ kHz}$**.

---

## Solution to Exercise 3

1. Dynamic range of the speech:
   $$\text{DR} = 92 \text{ dB} - 38 \text{ dB} = 54 \text{ dB}$$

2. Required SQNR:
   We require the quantization noise floor to be $15 \text{ dB}$ below the room background noise floor.
   Total dynamic range needed = $54 \text{ dB} + 15 \text{ dB} = 69 \text{ dB}$.
   Using $\text{SQNR} \approx 6.02 B + 1.76 \text{ dB}$:
   $$6.02 B + 1.76 \ge 69 \implies 6.02 B \ge 67.24 \implies B \ge 11.17$$
   The minimum integer bit depth is **$B = 12$ bits**. (16 bits provides 98 dB, offering 29 dB of comfortable headroom).
