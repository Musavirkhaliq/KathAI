# Practical Lab: Learning Rate Schedules and LLRD Verification

---

## Lab Objectives
1. Implement LLRD parameter grouping for a multi-layer model.
2. Verify that optimizer parameter groups have strictly decreasing learning rates.
3. Plot the warmup + cosine annealing curve over 1,000 steps.

---

## Lab Script: `lab_llrd_schedule.py`

```python
import torch
import torch.nn as nn
import math

def run_schedule_lab():
    # Simulate 5-layer model
    layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])
    base_lr = 1e-3
    decay = 0.5
    
    groups = []
    for i, l in enumerate(reversed(layers)):
        groups.append({"params": l.parameters(), "lr": base_lr * (decay ** i)})
        
    opt = torch.optim.AdamW(groups)
    
    print("Assigned Learning Rates per Layer (Top to Bottom):")
    for idx, grp in enumerate(opt.param_groups):
        print(f"Layer Group {idx}: LR = {grp['lr']:.6f}")
        
    assert opt.param_groups[0]["lr"] > opt.param_groups[-1]["lr"], "LLRD failed!"
    print("SUCCESS: LLRD hierarchy validated!")

if __name__ == "__main__":
    run_schedule_lab()
```
