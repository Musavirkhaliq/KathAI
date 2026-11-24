# Code Walkthrough: Paired Bootstrap Resampling (`src/evaluation/significance.py`)

---

## 1. Inspecting the Bootstrap Test Harness

```python
import numpy as np

def paired_bootstrap_test(refs, hyps_a, hyps_b, num_bootstrap=10000, alpha=0.05):
    # refs, hyps_a, hyps_b: lists of sentence strings of equal length M
    M = len(refs)
    deltas = np.zeros(num_bootstrap)
    
    # Pre-calculate utterance edit operations
    errors_a = []
    errors_b = []
    word_counts = []
    
    for r, ha, hb in zip(refs, hyps_a, hyps_b):
        words = len(r.split())
        ea = compute_errors(r, ha)
        eb = compute_errors(r, hb)
        errors_a.append(ea)
        errors_b.append(eb)
        word_counts.append(words)
        
    err_a = np.array(errors_a)
    err_b = np.array(errors_b)
    words = np.array(word_counts)
    
    for b in range(num_bootstrap):
        # Sample with replacement
        idx = np.random.randint(0, M, size=M)
        tot_w = np.sum(words[idx])
        wer_a = np.sum(err_a[idx]) / tot_w
        wer_b = np.sum(err_b[idx]) / tot_w
        deltas[b] = wer_b - wer_a
        
    # Two-tailed p-value
    p_val = 2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))
    is_significant = (p_val < alpha)
    
    ci_lower = np.percentile(deltas, 2.5)
    ci_upper = np.percentile(deltas, 97.5)
    
    return {
        "p_value": p_val,
        "is_significant": is_significant,
        "mean_delta": np.mean(deltas),
        "ci_95": (ci_lower, ci_upper)
    }
```
