# Code Walkthrough: Production Chunking (`src/inference/chunking.py`)

---

## 1. Inspecting Method C2 Audio Slicer

```python
import torch

def slice_audio_chunks(waveform: torch.Tensor, sr: int = 16000, chunk_sec: float = 8.0, overlap_sec: float = 1.0):
    chunk_samples = int(chunk_sec * sr)
    overlap_samples = int(overlap_sec * sr)
    hop_samples = chunk_samples - overlap_samples
    
    total_samples = waveform.shape[-1]
    chunks = []
    
    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunk = waveform[..., start:end]
        
        # Pad final chunk if necessary
        if chunk.shape[-1] < chunk_samples and total_samples > chunk_samples:
            pad_len = chunk_samples - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, pad_len))
            
        chunks.append({"waveform": chunk, "start_sample": start, "end_sample": end})
        if end == total_samples:
            break
        start += hop_samples
        
    return chunks
```
