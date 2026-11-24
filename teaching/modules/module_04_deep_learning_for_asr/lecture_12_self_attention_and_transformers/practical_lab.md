# Practical Lab: Multi-Head Attention and Variance Calibration

---

## Lab Objectives
1. Implement scaled dot-product attention in PyTorch.
2. Verify numerically that unscaled dot products have variance $d_k$, while scaled dot products have variance $1.0$.
3. Measure gradient magnitudes across Softmax with and without scaling.

---

## Lab Script: `lab_attention_scaling.py`

```python
import torch
import math

def run_attention_lab():
    B = 32
    T = 100
    d_k = 64
    
    # Generate random queries and keys with unit variance
    q = torch.randn(B, T, d_k)
    k = torch.randn(B, T, d_k)
    
    # 1. Unscaled dot products
    unscaled = torch.matmul(q, k.transpose(-1, -2))
    var_unscaled = unscaled.var().item()
    print(f"Unscaled Attention Logit Variance: {var_unscaled:.2f} (Expected ~ {d_k})")
    
    # 2. Scaled dot products
    scaled = unscaled / math.sqrt(d_k)
    var_scaled = scaled.var().item()
    print(f"Scaled Attention Logit Variance:   {var_scaled:.2f} (Expected ~ 1.00)")
    
    assert abs(var_scaled - 1.0) < 0.2, "Scaling failed to normalize variance!"
    print("SUCCESS: Scaled dot-product attention variance verified!")

if __name__ == "__main__":
    run_attention_lab()
```
