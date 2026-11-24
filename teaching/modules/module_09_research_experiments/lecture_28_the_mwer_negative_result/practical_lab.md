# Practical Lab: Simulating MWER Gradient Variance

---

## Lab Objectives
1. Implement empirical risk minimization using N-best beam search hypotheses.
2. Measure gradient variance under small vs. large beam sizes.
3. Observe how outlier sentences with high edit distance induce gradient instability.

---

## Lab Script: `lab_mwer_variance.py`

```python
import torch

def run_mwer_lab():
    torch.manual_seed(42)
    # Simulate 4 hypotheses with edit distances [0, 1, 4, 8]
    errors = torch.tensor([0.0, 1.0, 4.0, 8.0])
    logits = torch.randn(4, requires_grad=True)
    
    # Forward pass
    probs = torch.softmax(logits, dim=-1)
    expected_wer = torch.sum(probs * errors)
    expected_wer.backward()
    
    grad = logits.grad
    print(f"Computed MWER Gradient: {grad.numpy()}")
    print(f"Gradient Variance:      {torch.var(grad).item():.4f}")
    assert torch.var(grad) > 0.0, "Zero gradient variance!"
    print("SUCCESS: MWER gradient mechanics demonstrated!")

if __name__ == "__main__":
    run_mwer_lab()
```
