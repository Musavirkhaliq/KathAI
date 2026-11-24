# Instructor Delivery Script: Lecture 05 (Fourier Analysis, Spectrograms, and Mel Filterbanks)

## Timetable & Pedagogical Flow (90 Minutes Total)

```
00:00 - 00:20 | From Time to Frequency: The Discrete Fourier Transform
00:20 - 00:45 | Quasi-Stationarity and the Gabor-Heisenberg Uncertainty Principle
00:45 - 01:10 | Windowing Functions (Hann vs. Hamming) and Spectral Leakage
01:10 - 01:25 | Mel Filterbank Derivation: The Bridge from Biology to Features
01:25 - 01:30 | Why XLS-R Left Filterbanks Behind: Raw Waveforms & Takeaways
```

---

## Detailed Delivery Script

### 00:00 - 00:20 | Opening Hook & The DFT
- **Action**: Draw a single sinusoidal wave on the board.
  > "If I play a 440 Hz tuning fork, you hear an 'A'. The pressure wave is a pure sine.
  > But if I say the Kashmiri word 'کٲشُر', what is the pressure wave?
  > It is a chaotic, turbulent signal with hundreds of frequencies superimposed.
  > Jean-Baptiste Joseph Fourier gave us the mathematical lens to dismantle that signal into pure sinusoids.
  > Today, we study how that mathematical transform powers classical feature extraction, and why deep learning eventually learned to do it better."
- Write the DFT summation on the Center Board.
- Highlight the complex exponential $e^{-j \frac{2\pi}{N} k n} = \cos\left(\frac{2\pi}{N}kn\right) - j \sin\left(\frac{2\pi}{N}kn\right)$.
- Show that computing the DFT is projecting the signal onto an orthogonal basis of complex sinusoids.

### 00:20 - 00:45 | STFT & Heisenberg Gabor Trade-off
- Ask the class: *"Why can't we just compute one single Fourier Transform over a 10-second sentence?"*
- Guide them to the answer: A single FFT tells you *what* frequencies were present, but not *when* they happened.
- Introduce the STFT sliding window.
- Draw the Time-Frequency trade-off grid on the Left Board:
  - Tall thin rectangles: Narrowband (high frequency resolution $\Delta f$, poor time resolution $\Delta t$).
  - Short wide rectangles: Wideband (high time resolution $\Delta t$, poor frequency resolution $\Delta f$).
- State the Gabor uncertainty bound: $\Delta t \cdot \Delta f \ge \frac{1}{4\pi}$.

### 00:45 - 01:10 | Windowing and Leakage
- Demonstrate spectral leakage:
  - Show the rectangular window creating high sidelobes ($-13 \text{ dB}$).
  - Explain how a loud fundamental frequency at 150 Hz spills energy into higher frequency bins, masking weak formants.
  - Introduce the Hann and Hamming windows as gentle tapers that force the endpoints to zero.

### 01:10 - 01:30 | Mel Filterbanks to Raw Waveforms
- Derive the Mel warping formula.
- Draw the triangular overlapping filterbank from 0 to 8000 Hz.
- Explain why filters are narrow below 1 kHz and wide above 1 kHz.
- Conclude by previewing Module 04:
  > "Keep this triangular filterbank in your mind. Because in Lecture 11, when we inspect the first convolutional layer of XLS-R, you will see that the neural network learns filters that look astonishingly like these Mel filters, completely on its own from raw audio!"
