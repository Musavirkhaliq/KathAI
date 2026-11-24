# Exercises: Lecture 05 (Fourier Analysis and Mel Filterbanks)

---

## Exercise 1: DFT Frequency Bin Resolution
An engineer configures an STFT with sampling rate $f_s = 16,000 \text{ Hz}$ and FFT size $N_{\text{FFT}} = 512$.
1. What is the frequency spacing $\Delta f$ between adjacent DFT bins?
2. Which discrete DFT bin index $k$ corresponds to the first formant of the Kashmiri vowel /ə/ ($F_1 \approx 480 \text{ Hz}$)?
3. What is the physical frequency represented by bin index $k = 48$?

---

## Exercise 2: Mel Scale Conversion
1. Convert the following physical frequencies (Hz) to perceived pitch (mels):
   - $f = 300 \text{ Hz}$
   - $f = 1,000 \text{ Hz}$
   - $f = 4,000 \text{ Hz}$
2. Calculate the derivative $\frac{dm}{df}$. What does this derivative tell us about the frequency sensitivity of human hearing as physical frequency increases?

---

## Exercise 3: Wideband vs. Narrowband Trade-off
An audio analyst is inspecting an unvoiced Kashmiri affricate /t͡s/ and a sustained vowel /iː/.
1. For inspecting the vertical striations of vocal fold vibration in /iː/, should the analyst use an analysis window of $N=128$ (8 ms) or $N=1024$ (64 ms)?
2. For resolving the exact individual harmonic lines of the glottal source ($F_0 = 120 \text{ Hz}$), what is the minimum window duration $T_w$ required?
