# Practical Lab: Executing Bootstrap Significance Testing

---

## Lab Objectives
1. Implement non-parametric bootstrap resampling in NumPy.
2. Test whether an observed 0.40% WER improvement is statistically significant across 500 test sentences.
3. Compute 95% confidence intervals and plot the bootstrap delta histogram.

---

## Lab Script: `lab_significance_test.py`

```python
import numpy as np

def run_significance_lab():
    np.random.seed(42)
    M = 500 # 500 utterances
    
    # Simulate errors per utterance
    words_per_utt = np.random.randint(5, 25, size=M)
    errors_base = np.random.binomial(words_per_utt, 0.25)
    # Model candidate improves slightly (prob 0.245)
    errors_cand = np.random.binomial(words_per_utt, 0.245)
    
    B = 2000
    deltas = []
    
    for _ in range(B):
        idx = np.random.choice(M, size=M, replace=True)
        tot_w = np.sum(words_per_utt[idx])
        wer_base = np.sum(errors_base[idx]) / tot_w
        wer_cand = np.sum(errors_cand[idx]) / tot_w
        deltas.append(wer_base - wer_cand)
        
    deltas = np.array(deltas)
    p_val = 2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))
    ci = (np.percentile(deltas, 2.5), np.percentile(deltas, 97.5))
    
    print(f"Observed Mean Delta WER: {np.mean(deltas)*100:.3f}%")
    print(f"95% Confidence Interval: [{ci[0]*100:.3f}%, {ci[1]*100:.3f}%]")
    print(f"Calculated p-value:      {p_val:.4f}")
    print(f"Significant at p < 0.05: {p_val < 0.05}")
    print("SUCCESS: Bootstrap test framework executed cleanly!")

if __name__ == "__main__":
    run_significance_lab()
```
