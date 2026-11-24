# Code Walkthrough: Digital Signal Verification

---

## 1. Verifying Audio Invariants in `src/utils/audio.py`

Let us examine the production audio loader in `src/utils/audio.py`:

```python
import torch
import torchaudio

def validate_audio_file(filepath: str, max_duration: float = 30.0) -> dict:
    # [Docstring]
    Performs forensic inspection of an incoming audio file to ensure
    it conforms to the strict acoustic standards required by XLS-R fine-tuning.
    # [Docstring]
    info = torchaudio.info(filepath)
    
    # 1. Check Sampling Rate
    is_16k = (info.sample_rate == 16000)
    
    # 2. Check Channel Count
    is_mono = (info.num_channels == 1)
    
    # 3. Check Duration
    duration = info.num_frames / info.sample_rate
    is_valid_len = (0.5 <= duration <= max_duration)
    
    # 4. Check Bit Depth
    is_16bit = (info.bits_per_sample == 16)
    
    return {
        "filepath": filepath,
        "sample_rate": info.sample_rate,
        "channels": info.num_channels,
        "duration_sec": duration,
        "bits_per_sample": info.bits_per_sample,
        "is_valid": is_16k and is_mono and is_valid_len and is_16bit
    }
```

---

## 2. Simulating Aliasing in PyTorch

```python
import torch
import numpy as np

def generate_aliased_signal():
    # [Docstring]
    Demonstrates aliasing by sampling a 9 kHz tone at 16 kHz.
    # [Docstring]
    f_signal = 9000.0  # 9 kHz tone (exceeds Nyquist of 16 kHz = 8 kHz)
    fs = 16000.0       # 16 kHz sampling rate
    duration = 0.05    # 50 ms
    
    t = torch.linspace(0, duration, int(fs * duration), endpoint=False)
    # Sampled signal without anti-aliasing filter:
    sampled_wave = torch.sin(2 * np.pi * f_signal * t)
    
    # Detect the apparent frequency via FFT
    fft_vals = torch.abs(torch.fft.rfft(sampled_wave))
    freqs = torch.fft.rfftfreq(len(t), d=1/fs)
    peak_freq = freqs[torch.argmax(fft_vals)].item()
    
    print(f"Original Frequency: {f_signal} Hz")
    print(f"Apparent Detected Frequency after 16 kHz sampling: {peak_freq:.1f} Hz")
    assert abs(peak_freq - 7000.0) < 100, "Aliasing folding failed!"

if __name__ == "__main__":
    generate_aliased_signal()
```
