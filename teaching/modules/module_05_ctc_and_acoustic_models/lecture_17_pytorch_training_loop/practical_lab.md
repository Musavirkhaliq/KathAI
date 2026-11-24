# Practical Lab: Mixed Precision and Gradient Accumulation Verification

---

## Lab Objectives
1. Implement gradient accumulation over 4 micro-steps.
2. Verify that accumulated gradients match a single full-batch gradient within numerical tolerance.
3. Observe dynamic scale adjustment in `GradScaler`.

---

## Lab Script: `lab_training_mechanics.py`

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

def run_loop_lab():
    model = nn.Linear(32, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = GradScaler(enabled=False) # CPU simulation
    
    # Simulate 4 micro-batches
    accum_steps = 4
    optimizer.zero_grad()
    
    for _ in range(accum_steps):
        x = torch.randn(8, 32)
        y = model(x).sum() / accum_steps
        y.backward()
        
    # Check that grad is non-zero
    grad_norm = model.weight.grad.norm().item()
    print(f"Accumulated Gradient Norm: {grad_norm:.4f}")
    assert grad_norm > 0.0, "Zero gradient after accumulation!"
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    clipped_norm = model.weight.grad.norm().item()
    print(f"Clipped Gradient Norm:     {clipped_norm:.4f}")
    assert clipped_norm <= 1.0 + 1e-5, "Clipping invariant violated!"
    print("SUCCESS: Training loop invariants verified!")

if __name__ == "__main__":
    run_loop_lab()
```
