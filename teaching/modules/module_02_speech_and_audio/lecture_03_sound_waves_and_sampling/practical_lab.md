# Practical Lab: Audio Digitization, Aliasing, and Quantization

---

## Laboratory Goals
1. Generate synthetic continuous-time signals using dense micro-grids.
2. Observe time-domain waveform distortion and spectral aliasing under sub-Nyquist sampling.
3. Compute the empirical Signal-to-Quantization-Noise Ratio across 8-bit, 12-bit, and 16-bit linear PCM representations.

---

## Lab Script: `lab_sampling.py`

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

def run_lab():
    print("=== STEP 1: Aliasing Demonstration ===")
    # Dense continuous analog approximation (fs_analog = 200 kHz)
    fs_analog = 200000.0
    duration = 0.02
    t_analog = torch.linspace(0, duration, int(fs_analog * duration), endpoint=False)
    
    f_tone = 11000.0 # 11 kHz signal
    x_analog = torch.cos(2 * np.pi * f_tone * t_analog)
    
    # Sample at 16 kHz (Nyquist = 8 kHz)
    fs_digital = 16000.0
    step = int(fs_analog / fs_digital)
    t_digital = t_analog[::step]
    x_digital = x_analog[::step]
    
    # FFT of discrete signal
    fft_spec = torch.abs(torch.fft.rfft(x_digital))
    freqs = torch.fft.rfftfreq(len(x_digital), 1.0 / fs_digital)
    apparent_f = freqs[torch.argmax(fft_spec)].item()
    print(f"Injected Frequency: {f_tone} Hz")
    print(f"Aliased Baseband Peak: {apparent_f:.1f} Hz")
    print(f"Theoretical Fold: |11000 - 16000| = 5000 Hz")
    
    print("\n=== STEP 2: SQNR Empirical Verification ===")
    bits_to_test = [4, 8, 12, 16]
    for B in bits_to_test:
        # Quantize x_analog to B bits
        num_levels = 2 ** B
        delta = 2.0 / num_levels
        x_quant = torch.clamp(torch.round(x_analog / delta) * delta, -1.0, 1.0)
        
        # Calculate noise
        noise = x_analog - x_quant
        p_signal = torch.mean(x_analog ** 2)
        p_noise = torch.mean(noise ** 2)
        sqnr_empirical = 10 * torch.log10(p_signal / p_noise).item()
        sqnr_theory = 6.02 * B + 1.76
        
        print(f"Bits: {B:2d} | Empirical SQNR: {sqnr_empirical:6.2f} dB | Theoretical SQNR: {sqnr_theory:6.2f} dB")

if __name__ == "__main__":
    run_lab()
```
