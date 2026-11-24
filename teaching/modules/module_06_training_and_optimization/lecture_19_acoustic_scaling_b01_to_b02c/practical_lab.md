# Practical Lab: Fitting Speech Scaling Curves

---

## Lab Objectives
1. Load historical benchmark metrics from B-01, B-02, B-02C, and E22-01.
2. Implement non-linear curve fitting for power-law scaling exponents.
3. Extrapolate required dataset size to achieve an ambitious 15.0% disjoint WER target.

---

## Lab Script: `lab_scaling_curves.py`

```python
import numpy as np

def run_scaling_lab():
    hours = [20.4, 61.8, 112.5, 174.5]
    wers = [42.1, 32.8, 25.82, 24.12]
    
    wer_floor = 18.0
    x = np.log(hours)
    y = np.log(np.array(wers) - wer_floor)
    
    slope, intercept = np.polyfit(x, y, 1)
    alpha = -slope
    A = np.exp(intercept)
    
    print(f"Fitted Power-Law Exponent: alpha = {alpha:.4f}")
    print(f"Fitted Coefficient:        A     = {A:.2f}")
    assert 0.3 < alpha < 0.6, "Power-law exponent out of expected bounds!"
    print("SUCCESS: Scaling curve verified!")

if __name__ == "__main__":
    run_scaling_lab()
```
