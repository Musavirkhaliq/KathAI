# Code Walkthrough: Golden Parity Test (`tests/test_inference_parity.py`)

---

## 1. Inspecting the Parity Test Suite

```python
import torch
import numpy as np

def test_golden_logit_parity(legacy_model, refactored_pipeline, reference_audio_tensor):
    # 1. Historical forward pass
    with torch.no_grad():
        legacy_logits = legacy_model(reference_audio_tensor.unsqueeze(0)).logits.squeeze(0)
        
    # 2. Refactored forward pass
    refactored_logits = refactored_pipeline.extract_logits(reference_audio_tensor)
    
    # 3. Shape assertion
    assert legacy_logits.shape == refactored_logits.shape, f"Shape mismatch: {legacy_logits.shape} vs {refactored_logits.shape}"
    
    # 4. Numerical difference assertion
    diff = torch.max(torch.abs(legacy_logits - refactored_logits)).item()
    print(f"Max Absolute Logit Difference: {diff:.8f}")
    assert diff < 1e-5, f"Logit parity broken! Max diff: {diff}"
    print("SUCCESS: Bit-level inference parity verified!")
```
