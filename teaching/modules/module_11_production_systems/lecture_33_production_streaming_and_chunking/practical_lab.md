# Practical Lab: Benchmarking Real-Time Factor (RTF)

---

## Lab Objectives
1. Implement Method C2 chunking over a synthetic 60-second audio buffer.
2. Measure wall-clock execution time and compute empirical RTF.
3. Verify that boundary cross-fading eliminates duplicate character emissions.

---

## Lab Script: `lab_streaming_rtf.py`

```python
import time
import torch
from src.inference.chunking import slice_audio_chunks

def run_rtf_lab():
    sr = 16000
    duration = 60.0 # 60 seconds
    audio = torch.randn(1, int(sr * duration))
    
    t0 = time.time()
    chunks = slice_audio_chunks(audio, sr=sr, chunk_sec=8.0, overlap_sec=1.0)
    
    # Simulate forward pass processing
    for c in chunks:
        # Dummy compute: 5 ms per chunk
        time.sleep(0.005)
        
    elapsed = time.time() - t0
    rtf = elapsed / duration
    print(f"Total Chunks: {len(chunks)} | Processing Time: {elapsed:.3f}s")
    print(f"Calculated RTF: {rtf:.4f} (Real-time limit is 1.00)")
    assert rtf < 0.10, "RTF exceeds performance threshold!"
    print("SUCCESS: Streaming chunking benchmark validated!")

if __name__ == "__main__":
    run_rtf_lab()
```
