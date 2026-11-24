# Code Walkthrough: Empirical Scaling Law Fitting

---

## 1. Fitting Power Law to Repository Benchmarks

```python
import numpy as np

def fit_scaling_law(hours, wers, wer_floor=18.0):
    # hours: [20.4, 61.8, 112.5, 174.5]
    # wers:  [42.1, 32.8, 25.82, 24.12]
    h = np.array(hours)
    w = np.array(wers) - wer_floor
    
    x = np.log(h)
    y = np.log(w)
    
    # Linear fit
    coeffs = np.polyfit(x, y, 1)
    alpha = -coeffs[0]
    A = np.exp(coeffs[1])
    
    return {"alpha": alpha, "A": A, "wer_floor": wer_floor}
```
