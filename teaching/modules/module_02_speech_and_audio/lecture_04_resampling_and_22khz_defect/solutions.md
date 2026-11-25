# Solutions: Lecture 04 (Resampling Algorithms and the 22.05 kHz Defect)

---

## Solution to Exercise 1

1. Reducing ratios:
   - $44,100 \to 16,000$:

$$\frac{16000}{44100} = \frac{160}{441} \quad (\gcd(160, 441) = 1) \implies L = 160, M = 441$$

   - $48,000 \to 16,000$:

$$\frac{16000}{48000} = \frac{1}{3} \implies L = 1, M = 3$$

   - $22,050 \to 16,000$:

$$\frac{16000}{22050} = \frac{320}{441} \implies L = 320, M = 441$$

   - $8,000 \to 16,000$:

$$\frac{16000}{8000} = \frac{2}{1} \implies L = 2, M = 1$$

2. Computational difference:
   $48 \text{ kHz} \to 16 \text{ kHz}$ requires integer decimation by $M=3$ with no upsampling ($L=1$). A simple 3-tap FIR filter downsamples directly.
   $22.05 \text{ kHz} \to 16 \text{ kHz}$ requires $L=320$ polyphase branches and filtering across a complex 441-point decimation grid, requiring $320 \times$ more filter coefficient storage and memory lookups.

---

## Solution to Exercise 2

1. Kaiser parameter $\beta$:
   For $A = 85 \text{ dB} > 50 \text{ dB}$:

$$\beta = 0.1102 (A - 8.7) = 0.1102 (85 - 8.7) = 0.1102 \times 76.3 = \mathbf{8.408}$$

2. Estimated filter length $N$:

$$N \approx \frac{A - 8}{2.285 \Delta \omega} = \frac{85 - 8}{2.285 \times (0.05 \pi)} = \frac{77}{2.285 \times 0.15708} = \frac{77}{0.3589} \approx 214.5 \implies \mathbf{N = 215}$$

3. Latency at 16 kHz:
   For a linear-phase FIR filter of length $N = 215$, the group delay is:

$$\tau = \frac{N - 1}{2} = \frac{214}{2} = 107 \text{ samples}$$

   In milliseconds:

$$t_{\text{latency}} = \frac{107}{16000} \text{ s} \approx 0.00669 \text{ s} = \mathbf{6.69 \text{ ms}}$$

---

## Solution to Exercise 3

- Fricatives and affricates (/t͡s/, /ʃ/, /s/) rely on high-frequency turbulent noise produced by constriction in the anterior oral cavity, with acoustic spectral peaks between $5.5 \text{ kHz}$ and $8.0 \text{ kHz}$. When the resampler prematurely cut off frequencies above $5.8 \text{ kHz}$, the turbulent noise spectrum was obliterated, collapsing the acoustic evidence into generic low-frequency stops (/t/) or adjacent sibilants.
- Nasals and voiced stops (/m/, /b/) are low-frequency sounds characterized by voice bars and low formants ($F_1 < 500 \text{ Hz}$, $F_2 < 1500 \text{ Hz}$). The defective resampler left the band below $5.0 \text{ kHz}$ unaffected, preserving nasals and stops with high fidelity.
