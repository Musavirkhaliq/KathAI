# Code Walkthrough: Spectrogram and Filterbank Computation

---

## 1. Computing Log-Mel Filterbanks in PyTorch

```python
import torch
import torchaudio.transforms as T

class LogMelSpectrogramExtractor:
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,       # 32 ms window
        win_length: int = 400,   # 25 ms analysis window
        hop_length: int = 160,   # 10 ms frame shift
        n_mels: int = 80         # 80 mel channels
    ):
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            window_fn=torch.hann_window,
            power=2.0
        )
        
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        # [Docstring]
        Args:
            waveform: Float tensor [batch, time]
        Returns:
            Log-mel energy tensor [batch, n_mels, time_frames]
        # [Docstring]
        # 1. Compute power spectrogram through STFT + triangular filterbank
        mel_spec = self.mel_transform(waveform)
        
        # 2. Convert to log scale (add epsilon to prevent log(0) -inf)
        log_mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        
        return log_mel_spec
```

---

## 2. Inspecting Temporal Dimensions

For an input audio of 3.0 seconds at 16,000 Hz:
$$N_\text{samples} = 3.0 \times 16,000 = 48,000$$

With $H = 160$ samples:
$$T_\text{frames} = \left\lfloor \frac{48000}{160} \right\rfloor + 1 = 301$$

The resulting feature tensor has shape `[batch, 80, 301]`.
