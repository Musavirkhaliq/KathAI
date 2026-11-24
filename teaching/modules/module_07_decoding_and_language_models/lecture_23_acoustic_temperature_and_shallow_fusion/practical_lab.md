# Practical Lab: Grid Searching Alpha, Beta, and Temperature

---

## Lab Objectives
1. Implement temperature scaling over raw CTC logits.
2. Execute a 2D grid search over $\alpha \in [0.1, 0.6]$ and $\beta \in [0.5, 2.0]$.
3. Identify the global minimum WER configuration on a toy validation split.

---

## Lab Script: `lab_tune_decoding.py`

```python
import numpy as np

def run_tuning_lab():
    alphas = [0.1, 0.3, 0.5]
    betas = [0.5, 1.0, 1.5]
    
    # Simulate a convex WER surface
    # Minimum at alpha = 0.3, beta = 1.0
    best_wer = 999.0
    best_cfg = None
    
    for a in alphas:
        for b in betas:
            # Simulated parabolic response surface
            wer = 25.0 + 15.0 * (a - 0.3)**2 + 10.0 * (b - 1.0)**2
            print(f"Alpha: {a:.2f} | Beta: {b:.2f} | Simulated WER: {wer:.2f}%")
            if wer < best_wer:
                best_wer = wer
                best_cfg = (a, b)
                
    print(f"\nOptimal Configuration: Alpha={best_cfg[0]}, Beta={best_cfg[1]} (WER = {best_wer:.2f}%)")
    assert best_cfg == (0.3, 1.0), "Grid search failed to find optimal point!"
    print("SUCCESS: Decoding hyperparameter tuning validated!")

if __name__ == "__main__":
    run_tuning_lab()
```
