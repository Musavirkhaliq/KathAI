# Practical Lab: Gradient Flow and Activation Analysis

---

## Lab Objectives
1. Implement forward and backward passes of a 10-layer MLP in pure PyTorch.
2. Compare gradient norms across layers with and without residual skip connections.
3. Observe gradient vanishing when using sigmoid activations vs. stable flow with GELU + LayerNorm.

---

## Lab Script: `lab_gradient_flow.py`

```python
import torch
import torch.nn as nn

def run_gradient_experiment():
    depth = 12
    dim = 256
    batch_size = 16
    
    # Model without residuals (Plain MLP)
    plain_layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(depth)])
    
    x = torch.randn(batch_size, dim, requires_grad=True)
    h = x
    for layer in plain_layers:
        h = torch.tanh(layer(h))
        
    loss = h.sum()
    loss.backward()
    
    first_grad = plain_layers[0].weight.grad.norm().item()
    last_grad = plain_layers[-1].weight.grad.norm().item()
    print(f"Plain MLP - First Layer Grad Norm: {first_grad:.6f} | Last Layer: {last_grad:.6f}")
    print(f"Gradient Decay Ratio: {first_grad / last_grad:.6f}")
    assert first_grad / last_grad < 0.1, "Expected gradient vanishing in deep plain tanh network"
    print("SUCCESS: Gradient decay demonstrated!")

if __name__ == "__main__":
    run_gradient_experiment()
```
