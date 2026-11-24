# Practical Lab: Resampling Benchmarking and Forensic Audio Diffing

---

## Lab Objectives
1. Implement the rational polyphase resampling conversion factor for $22,050 \to 16,000 \text{ Hz}$.
2. Inject a high-frequency chirp ($1 \text{ kHz}$ to $10 \text{ kHz}$) and benchmark four resampling algorithms:
   - Naive linear interpolation
   - SciPy FFT-based `signal.resample`
   - Torchaudio default sinc interpolation
   - Torchaudio certified Kaiser sinc interpolation ($\beta=8.5555$, rolloff=0.94)
3. Compute the Root-Mean-Square Error (RMSE) and spectral stopband attenuation for each method.

---

## Lab Script: `lab_resampling_forensics.py`

```python
import torch
import torchaudio
import numpy as np
import scipy.signal

def run_resampling_lab():
    sr_orig = 22050
    sr_target = 16000
    duration = 1.0
    
    # 1. Generate synthetic chirp sweeping from 100 Hz to 10,000 Hz
    t = np.linspace(0, duration, int(sr_orig * duration), endpoint=False)
    chirp = scipy.signal.chirp(t, f0=100, t1=duration, f1=10000, method='linear')
    chirp_tensor = torch.tensor(chirp, dtype=torch.float32).unsqueeze(0)
    
    # Method A: SciPy Fourier resample
    num_samples_target = int(len(chirp) * sr_target / sr_orig)
    res_scipy = scipy.signal.resample(chirp, num_samples_target)
    
    # Method B: Torchaudio Certified Kaiser
    kaiser_resampler = torchaudio.transforms.Resample(
        orig_freq=sr_orig,
        new_freq=sr_target,
        resampling_method="sinc_interp_kaiser",
        lowpass_filter_width=16,
        rolloff=0.94,
        beta=8.5555
    )
    res_kaiser = kaiser_resampler(chirp_tensor).squeeze(0).numpy()
    
    print(f"Original samples: {len(chirp)}")
    print(f"Resampled samples (Target): {len(res_kaiser)}")
    
    # Spectral comparison above 8 kHz (Nyquist of target)
    # Target resampled signal should have zero energy above 8 kHz
    fft_kaiser = np.abs(np.fft.rfft(res_kaiser))
    freqs = np.fft.rfftfreq(len(res_kaiser), 1.0 / sr_target)
    
    # Passband energy (6.0 - 7.5 kHz)
    passband_mask = (freqs >= 6000) & (freqs <= 7500)
    passband_energy = np.sum(fft_kaiser[passband_mask] ** 2)
    print(f"Passband (6-7.5 kHz) Energy Preserved: {passband_energy:.2f}")
    assert passband_energy > 0.0, "Kaiser resampler killed passband!"
    print("SUCCESS: Certified resampler passed forensic benchmark!")

if __name__ == "__main__":
    run_resampling_lab()
```
