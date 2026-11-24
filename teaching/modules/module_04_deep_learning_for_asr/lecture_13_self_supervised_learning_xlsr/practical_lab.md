# Practical Lab: Gumbel-Softmax Vector Quantization

---

## Lab Objectives
1. Implement Gumbel-Softmax categorical sampling in PyTorch.
2. Observe how temperature $\tau$ controls the transition between continuous soft probabilities and discrete one-hot vectors.
3. Compute codebook perplexity and verify codebook diversity loss.

---

## Lab Script: `lab_gumbel_quantization.py`

```python
import torch
import torch.nn.functional as F

def run_gumbel_lab():
    vocab_size = 10
    logits = torch.randn(5, vocab_size)
    
    # Sample at high temperature tau = 2.0 (soft)
    sample_soft = F.gumbel_softmax(logits, tau=2.0, hard=False)
    print("Soft Gumbel Sample (tau=2.0, row 0 max):", sample_soft[0].max().item())
    
    # Sample at low temperature tau = 0.2 (near discrete)
    sample_hard = F.gumbel_softmax(logits, tau=0.2, hard=False)
    print("Sharp Gumbel Sample (tau=0.2, row 0 max):", sample_hard[0].max().item())
    
    assert sample_hard[0].max().item() > sample_soft[0].max().item(), "Low tau should be sharper!"
    print("SUCCESS: Gumbel-Softmax temperature annealing verified!")

if __name__ == "__main__":
    run_gumbel_lab()
```
