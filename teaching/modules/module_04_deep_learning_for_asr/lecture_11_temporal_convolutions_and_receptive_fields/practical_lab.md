# Practical Lab: Receptive Field Calculation and Verification

---

## Lab Objectives
1. Implement automated receptive field tracking in PyTorch.
2. Verify empirical receptive field by computing gradient impulse responses from output neurons back to the input audio buffer.
3. Validate output shape assertions for arbitrary audio lengths.

---

## Lab Script: `lab_receptive_field.py`

```python
import torch
import torch.nn as nn

def run_rf_lab():
    # Build a 2-layer toy convolutional stack
    # Layer 1: K=10, S=5
    # Layer 2: K=3, S=2
    # Theoretical RF: R1 = 10, R2 = 10 + (3-1)*5 = 20 samples
    conv1 = nn.Conv1d(1, 16, kernel_size=10, stride=5, bias=False)
    conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=2, bias=False)
    
    # Initialize all weights to 1.0 to trace connections
    nn.init.constant_(conv1.weight, 1.0)
    nn.init.constant_(conv2.weight, 1.0)
    
    x = torch.zeros(1, 1, 100, requires_grad=True)
    out = conv2(conv1(x))
    
    # Backprop from first output frame center neuron
    out[0, 0, 0].backward()
    
    # Count non-zero gradient positions in input x
    grad = x.grad[0, 0]
    active_samples = torch.sum(grad != 0).item()
    print(f"Empirically measured active input samples: {active_samples}")
    print("Theoretical RF = 20 samples")
    assert active_samples == 20, f"Expected 20, got {active_samples}"
    print("SUCCESS: Empirical receptive field perfectly matches theory!")

if __name__ == "__main__":
    run_rf_lab()
```
