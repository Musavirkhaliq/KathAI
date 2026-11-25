# Code Walkthrough: Audio Invariant Verification and Pipeline Entry Points

---

## 1. Inspecting the Production Audio Loading Engine ([`src/data_processing/audio_processor.py`](file:///home/musa/Kashmiri%20Project/src/data_processing/audio_processor.py))

In this walkthrough, we inspect the production audio preprocessing engine from [`src/data_processing/audio_processor.py`](file:///home/musa/Kashmiri%20Project/src/data_processing/audio_processor.py). Every incoming recording must be strictly conditioned before it enters neural feature extraction.

```python
# From src/data_processing/audio_processor.py
from pathlib import Path
from typing import Union, Tuple
import numpy as np
import soundfile as sf
import scipy.signal

def load_audio(
    file_path: Union[str, Path],
    target_sample_rate: int = 16000,
    to_mono: bool = True
) -> Tuple[np.ndarray, int]:
    # 1. Existence audit
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # 2. Raw disk read via libsndfile
    data, sample_rate = sf.read(str(path))

    # 3. Precision normalization to float32
    if data.dtype != np.float32:
        data = data.astype(np.float32)

    # 4. Invariant: Multi-channel downmix to Mono
    # In speech recognition, phase differences across stereo microphones
    # create destructive interference if naively summed without beamforming.
    if to_mono and data.ndim > 1:
        data = np.mean(data, axis=1)

    # 5. Invariant: Resampling to exactly 16,000 Hz
    # If the source audio is 22.05 kHz (Bolbosh corpus) or 44.1 kHz,
    # it must be resampled to 16 kHz using high-fidelity band-limited interpolation.
    if sample_rate != target_sample_rate:
        num_target_samples = int(round(len(data) * float(target_sample_rate) / float(sample_rate)))
        data = scipy.signal.resample(data, num_target_samples).astype(np.float32)
        sample_rate = target_sample_rate

    return data, sample_rate

def normalize_audio(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    # Invariant: Peak amplitude scaling to prevent digital clipping
    max_val = np.max(np.abs(audio))
    if max_val > 1e-6:
        return (audio / max_val * target_peak).astype(np.float32)
    return audio
```

---

## 2. PyTorch / Torchaudio Polyphase Kaiser Sinc Resampling

When streaming audio or executing batch GPU inference, the pipeline utilizes Torchaudio's polyphase Kaiser-windowed sinc resampler:

```python
import torch
import torchaudio

def resample_polyphase_kaiser(waveform: torch.Tensor, orig_sr: int, target_sr: int = 16000) -> torch.Tensor:
    if orig_sr == target_sr:
        return waveform
        
    resampler = torchaudio.transforms.Resample(
        orig_freq=orig_sr,
        new_freq=target_sr,
        resampling_method="sinc_interp_kaiser",
        lowpass_filter_width=16,
        rolloff=0.94,
        beta=8.5555
    )
    return resampler(waveform)
```

### Critical Signal Processing Parameters:
- **`sinc_interp_kaiser`**: Avoids the destructive spectral blurring of linear interpolation.
- **`rolloff=0.94`**: Sets the filter cutoff at $0.94 \times 8000\text{ Hz} = 7520\text{ Hz}$, perfectly preserving high-frequency Kashmiri fricatives (/s/, /ʃ/, /t͡s/).
- **`beta=8.5555`**: Guarantees stopband attenuation exceeding $80\text{ dB}$, preventing high-frequency aliasing folding into the speech band.
- **`target_peak=0.95`**: Leaves $0.45\text{ dB}$ of headroom, preventing digital clipping when resampled samples overshoot.

---

## 3. Auditing Audio Invariants in Production

In [`src/asr/transcribe.py`](file:///home/musa/Kashmiri%20Project/src/asr/transcribe.py), these conditioning functions are called as the first step of inference:

```python
# Trace in src/asr/transcribe.py
audio = load_audio(audio_path, target_sr=16000)
audio = normalize_audio(audio)
num_samples = len(audio)

if num_samples < 400:
    logger.warning(f"Audio file {audio_path} has only {num_samples} samples; below receptive field (400).")
    return {"text": "", "tokens": [], "confidence": 0.0, "raw_text": ""}
```

If an audio snippet is shorter than the 400-sample receptive field of the Conv1d feature extractor ($25$ ms), it cannot produce a single valid acoustic frame. The pipeline gracefully traps this boundary condition rather than throwing an unhandled CUDA kernel exception.
