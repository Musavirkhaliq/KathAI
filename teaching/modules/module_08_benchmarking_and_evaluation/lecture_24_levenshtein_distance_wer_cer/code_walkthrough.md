# Code Walkthrough: Production Metric Computation (`src/evaluation/metrics.py`)

---

## 1. Inspecting the Micro-Averaged WER Engine

```python
import numpy as np

def compute_levenshtein_ops(ref_tokens, hyp_tokens):
    # Returns (S, D, I, C)
    N, M = len(ref_tokens), len(hyp_tokens)
    dp = np.zeros((N + 1, M + 1), dtype=int)
    
    for i in range(N + 1):
        dp[i, 0] = i
    for j in range(M + 1):
        dp[0, j] = j
        
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            if ref_tokens[i-1] == hyp_tokens[j-1]:
                dp[i, j] = dp[i-1, j-1]
            else:
                dp[i, j] = 1 + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
                
    # Backtrack
    i, j = N, M
    S, D, I, C = 0, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_tokens[i-1] == hyp_tokens[j-1]:
            C += 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i, j] == dp[i-1, j-1] + 1:
            S += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i, j] == dp[i-1, j] + 1:
            D += 1
            i -= 1
        else:
            I += 1
            j -= 1
            
    return S, D, I, C

def compute_corpus_wer(references, hypotheses):
    tot_S, tot_D, tot_I, tot_N = 0, 0, 0, 0
    for r, h in zip(references, hypotheses):
        r_words = r.split()
        h_words = h.split()
        S, D, I, C = compute_levenshtein_ops(r_words, h_words)
        tot_S += S
        tot_D += D
        tot_I += I
        tot_N += len(r_words)
        
    wer = (tot_S + tot_D + tot_I) / max(1, tot_N)
    return wer, tot_S, tot_D, tot_I, tot_N
```
