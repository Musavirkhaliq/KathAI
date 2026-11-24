# Code Walkthrough: DTW Implementation in Python

---

## 1. Dynamic Time Warping Algorithm

```python
import numpy as np

def compute_dtw_cost(seq1: np.ndarray, seq2: np.ndarray) -> float:
    # seq1: [N, D], seq2: [M, D]
    N, M = len(seq1), len(seq2)
    cost_matrix = np.full((N + 1, M + 1), np.inf)
    cost_matrix[0, 0] = 0.0
    
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            dist = np.linalg.norm(seq1[i-1] - seq2[j-1])
            cost_matrix[i, j] = dist + min(
                cost_matrix[i-1, j],    # Insertion
                cost_matrix[i, j-1],    # Deletion
                cost_matrix[i-1, j-1]   # Match
            )
            
    return cost_matrix[N, M]
```
