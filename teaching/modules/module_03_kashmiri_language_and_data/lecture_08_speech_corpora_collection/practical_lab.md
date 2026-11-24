# Practical Lab: Single-Channel SNR Estimation and Audio Auditing

---

## Lab Objectives
1. Implement single-channel quantile-based SNR estimation in PyTorch.
2. Build an automated clipping detection script.
3. Filter a directory of synthetic corrupted audio files into `clean/` and `quarantined/` manifests.

---

## Lab Script: `lab_audio_curation.py`

```python
import torch
import numpy as np

def run_curation_lab():
    sr = 16000
    duration = 3.0
    t = torch.linspace(0, duration, int(sr * duration))
    
    # Generate clean speech-like signal
    speech = torch.sin(2 * np.pi * 300 * t) * (torch.sin(2 * np.pi * 3 * t) > 0).float()
    
    # Corrupt with Gaussian noise (SNR ~ 6 dB)
    noise = torch.randn_like(speech) * 0.5
    noisy_audio = speech + noise
    
    # Quantile-based SNR estimation
    energy = noisy_audio ** 2
    noise_est = torch.quantile(energy, 0.10)
    signal_est = torch.quantile(energy, 0.95)
    snr = 10 * torch.log10(signal_est / noise_est).item()
    
    print(f"Estimated SNR: {snr:.2f} dB")
    should_keep = snr >= 12.0
    print(f"Curation Decision: {'KEEP' if should_keep else 'QUARANTINE'}")
    assert not should_keep, "Failed to quarantine noisy audio!"
    print("SUCCESS: Curation filter functioned as expected!")

if __name__ == "__main__":
    run_curation_lab()
```
