# Code Walkthrough: Pipeline Inspection and Execution Trace

---

## 1. Primary Source Reference: `src/inference/pipeline.py`

In this walkthrough, we inspect the end-to-end inference execution flow from `src/inference/pipeline.py` and `src/utils/audio.py`.

### Audio Loading and Normalization
The entry point takes an arbitrary audio path or raw waveform:

```python
# From src/utils/audio.py
import torch
import torchaudio

def load_and_preprocess_audio(audio_path: str, target_sr: int = 16000) -> torch.Tensor:
    # [Docstring]
    Loads an audio file, converts it to single-channel (mono),
    and resamples to target_sr using polyphase sinc interpolation.
    # [Docstring]
    waveform, sr = torchaudio.load(audio_path)
    
    # 1. Convert to mono if multi-channel
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # 2. Resample if sampling rate mismatches
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=target_sr,
            resampling_method="sinc_interp_kaiser"
        )
        waveform = resampler(waveform)
        
    # 3. Peak normalization to prevent clipping
    max_val = torch.max(torch.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val
        
    return waveform.squeeze(0)  # Shape: [T_samples]
```

---

## 2. Acoustic Forward Pass and Logit Slicing

```python
# From src/inference/pipeline.py
@torch.inference_mode()
def extract_logits(model, audio_tensor: torch.Tensor, device: torch.device):
    # [Docstring]
    Passes raw 16kHz audio through the fine-tuned XLS-R backbone
    and emits frame-level CTC logits.
    # [Docstring]
    model.eval()
    audio_tensor = audio_tensor.unsqueeze(0).to(device)  # [1, T_samples]
    
    # Forward pass through feature extractor + Transformer
    outputs = model(audio_tensor)
    logits = outputs.logits  # Shape: [1, T_frames, vocab_size]
    
    # Apply acoustic temperature calibration
    T = 1.08
    scaled_logits = logits / T
    
    return scaled_logits.squeeze(0)  # Shape: [T_frames, 47]
```

### Tensor Shape Assertions:
1. For an input audio of 5.0 seconds at 16 kHz:
   $$T_\text{samples} = 5.0 \times 16000 = 80,000$$
2. The XLS-R convolutional feature encoder downsamples by a factor of 320:
   $$T_\text{frames} = \left\lfloor \frac{80000 - 400}{320} \right\rfloor + 1 = 249$$
3. Resulting logit tensor has exact shape: `[249, 47]`.
