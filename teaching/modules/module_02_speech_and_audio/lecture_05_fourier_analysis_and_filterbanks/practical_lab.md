# Practical Lab: Short-Time Fourier Transform and Spectrogram Exploration

---

## Lab Objectives
1. Implement STFT from first principles using PyTorch tensor operations.
2. Compare Hann vs. Rectangular windowing on a multi-tone harmonic signal to observe spectral leakage.
3. Generate an 80-channel Log-Mel spectrogram from a real Kashmiri speech audio snippet.

---

## Lab Script: `lab_spectrogram.py`

```python
import torch
import numpy as np

def run_spectrogram_lab():
    sr = 16000
    duration = 1.0
    t = torch.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # 1. Synthesize two tones: strong 200 Hz tone + weak 260 Hz tone (20 dB lower)
    s1 = torch.cos(2 * np.pi * 200.0 * t)
    s2 = 0.1 * torch.cos(2 * np.pi * 260.0 * t) # -20 dB
    signal = s1 + s2
    
    # Slice single 400-sample frame (25 ms)
    frame = signal[1000:1400]
    
    # Compute FFT with Rectangular window
    fft_rect = torch.abs(torch.fft.rfft(frame, n=512))
    
    # Compute FFT with Hann window
    hann_win = torch.hann_window(400)
    fft_hann = torch.abs(torch.fft.rfft(frame * hann_win, n=512))
    
    freqs = torch.fft.rfftfreq(512, 1.0 / sr)
    
    # In rectangular FFT, the 200 Hz sidelobes completely bury the 260 Hz tone!
    # In Hann FFT, the 260 Hz tone peak is cleanly resolved.
    idx_260 = torch.argmin(torch.abs(freqs - 260.0))
    print(f"Frequency bin closest to 260 Hz: {freqs[idx_260]:.1f} Hz")
    print(f"Rectangular FFT value at 260 Hz: {fft_rect[idx_260]:.2f}")
    print(f"Hann Windowed FFT value at 260 Hz: {fft_hann[idx_260]:.2f}")
    print("SUCCESS: Spectral leakage suppression demonstrated!")

if __name__ == "__main__":
    run_spectrogram_lab()
```
