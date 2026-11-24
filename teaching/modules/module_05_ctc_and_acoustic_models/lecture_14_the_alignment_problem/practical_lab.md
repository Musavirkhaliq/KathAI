# Practical Lab: Aligning Speech Sequences via Dynamic Time Warping

---

## Lab Objectives
1. Implement DTW in pure NumPy.
2. Align two audio recordings of the Kashmiri word `"کٲشُر"` spoken at different speeds.
3. Visualize the optimal non-linear warping path through the alignment lattice.

---

## Lab Script: `lab_dtw_alignment.py`

```python
import numpy as np

def run_dtw_lab():
    # Synthesize two feature sequences: fast vs. slow speech
    t_fast = np.linspace(0, 1, 20)
    feat_fast = np.sin(2 * np.pi * t_fast)[:, None]
    
    t_slow = np.linspace(0, 1, 40)
    feat_slow = np.sin(2 * np.pi * t_slow)[:, None]
    
    N, M = len(feat_fast), len(feat_slow)
    D = np.full((N + 1, M + 1), np.inf)
    D[0, 0] = 0.0
    
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            cost = abs(feat_fast[i-1, 0] - feat_slow[j-1, 0])
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
            
    total_cost = D[N, M]
    print(f"Optimal Warping Path Cost: {total_cost:.4f}")
    assert total_cost < 2.0, "DTW failed to align temporal sequences!"
    print("SUCCESS: DTW alignment verified!")

if __name__ == "__main__":
    run_dtw_lab()
```
