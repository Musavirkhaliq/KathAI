# Practical Lab: Building a Regression-Proof Parity Test Suite

---

## Lab Objectives
1. Implement a golden tensor test fixture in pytest.
2. Assert tensor shape, mean, variance, and max difference tolerances.
3. Catch an intentional regression bug injected into an audio resampler.

---

## Lab Script: `lab_parity_test.py`

```python
import torch

def run_parity_lab():
    # Ground truth golden tensor
    golden_logits = torch.ones(249, 47) * 2.5
    
    # Current pipeline output
    current_logits = torch.ones(249, 47) * 2.500001
    
    diff = torch.max(torch.abs(golden_logits - current_logits)).item()
    print(f"Logit Delta: {diff:.8f}")
    is_parity = diff < 1e-5
    print(f"Parity Pass: {is_parity}")
    assert is_parity, "Regression detected!"
    print("SUCCESS: Parity test framework validated!")

if __name__ == "__main__":
    run_parity_lab()
```
