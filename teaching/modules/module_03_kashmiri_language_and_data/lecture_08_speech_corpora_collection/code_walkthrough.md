# Code Walkthrough: Audio Curation and SNR Filtering

---

## 1. Inspecting the Quality Filter (`src/data/curation.py`)

```python
import torch
import torchaudio

def assess_audio_quality(audio_path: str, min_snr: float = 12.0) -> dict:
    # [Docstring]
    Evaluates audio quality criteria for inclusion in training manifests.
    # [Docstring]
    waveform, sr = torchaudio.load(audio_path)
    
    # 1. Duration check
    duration = waveform.shape[-1] / sr
    if duration < 0.8 or duration > 20.0:
        return {"keep": False, "reason": f"Duration {duration:.2f}s out of bounds"}
        
    # 2. Clipping detection: samples near +/- 1.0
    clipped_samples = torch.sum(torch.abs(waveform) >= 0.99).item()
    clipping_ratio = clipped_samples / waveform.numel()
    if clipping_ratio > 0.01:
        return {"keep": False, "reason": f"Digital clipping {clipping_ratio*100:.1f}%"}
        
    # 3. Simple Energy-based SNR approximation
    energy = waveform ** 2
    noise_floor = torch.quantile(energy, 0.10)
    peak_energy = torch.quantile(energy, 0.95)
    
    snr_est = 10 * torch.log10((peak_energy + 1e-8) / (noise_floor + 1e-8)).item()
    if snr_est < min_snr:
        return {"keep": False, "reason": f"Low SNR: {snr_est:.1f} dB"}
        
    return {"keep": True, "snr": snr_est, "duration": duration}
```
