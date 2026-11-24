# Code Walkthrough: Oracle WER Calculation (`src/evaluation/oracle.py`)

---

## 1. Computing Oracle WER across N-Best Candidates

```python
from src.evaluation.metrics import compute_levenshtein_ops

def compute_oracle_wer(nbest_hypotheses_per_utt, ground_truth_references):
    tot_oracle_errors = 0
    tot_words = 0
    
    for nbest_list, ref in zip(nbest_hypotheses_per_utt, ground_truth_references):
        ref_words = ref.split()
        tot_words += len(ref_words)
        
        min_err = float('inf')
        for hyp in nbest_list:
            hyp_words = hyp.split()
            S, D, I, C = compute_levenshtein_ops(ref_words, hyp_words)
            err = S + D + I
            if err < min_err:
                min_err = err
                
        tot_oracle_errors += min_err
        
    oracle_wer = tot_oracle_errors / max(1, tot_words)
    return oracle_wer
```
