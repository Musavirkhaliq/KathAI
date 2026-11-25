# Solutions: Lecture 05 (Fourier Analysis and Mel Filterbanks)

---

## Solution to Exercise 1

1. Frequency bin resolution:

$$\Delta f = \frac{f_s}{N_{\text{FFT}}} = \frac{16,000}{512} = \mathbf{31.25 \text{ Hz}}$$

2. Bin index for $F_1 = 480 \text{ Hz}$:

$$k = \text{round}\left( \frac{480}{31.25} \right) = \text{round}(15.36) = \mathbf{15}$$

   Actual frequency of bin 15: $15 \times 31.25 = 468.75 \text{ Hz}$.

3. Physical frequency of bin $k = 48$:

$$f_{48} = 48 \times 31.25 \text{ Hz} = \mathbf{1,500.0 \text{ Hz}}$$

---

## Solution to Exercise 2

1. Mel formula: $m = 2595 \log_{10}\left(1 + \frac{f}{700}\right)$
   - For $f = 300 \text{ Hz}$:

$$m = 2595 \log_{10}\left(1 + \frac{300}{700}\right) = 2595 \log_{10}(1.42857) \approx 2595 \times 0.1549 = \mathbf{401.98 \text{ mels}}$$

   - For $f = 1000 \text{ Hz}$:

$$m = 2595 \log_{10}\left(1 + \frac{1000}{700}\right) = 2595 \log_{10}(2.42857) \approx 2595 \times 0.38535 = \mathbf{1000.0 \text{ mels}}$$

   - For $f = 4000 \text{ Hz}$:

$$m = 2595 \log_{10}\left(1 + \frac{4000}{700}\right) = 2595 \log_{10}(6.71428) \approx 2595 \times 0.827 = \mathbf{2146.0 \text{ mels}}$$

2. Derivative:

$$\frac{dm}{df} = \frac{2595}{\ln(10)} \cdot \frac{1}{f + 700} \approx \frac{1127}{f + 700}$$

   As frequency $f$ increases, $\frac{dm}{df}$ decreases monotonically toward zero. This proves that human frequency resolution becomes progressively coarser at higher frequencies.

---

## Solution to Exercise 3

1. To inspect vertical striations corresponding to individual glottal pulse openings, we require high temporal resolution (wideband spectrogram). The analyst must use $N = 128$ (8 ms).
2. To resolve individual harmonics separated by $\Delta f = F_0 = 120 \text{ Hz}$, the main lobe width of the window must be narrower than $120 \text{ Hz}$.
   For a Hann window, the main lobe width is $\Delta f_{\text{main}} = \frac{4 f_s}{N} \le 120 \text{ Hz}$:

$$N \ge \frac{4 \times 16,000}{120} = \frac{64,000}{120} \approx 533.3 \text{ samples}$$

   In time:

$$T_w \ge \frac{534}{16000} \approx \mathbf{33.3 \text{ ms}}$$
