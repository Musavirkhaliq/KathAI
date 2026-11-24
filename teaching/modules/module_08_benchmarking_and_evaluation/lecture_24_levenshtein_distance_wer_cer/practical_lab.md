# Practical Lab: Dynamic Programming Levenshtein Alignment

---

## Lab Objectives
1. Implement the Levenshtein DP matrix with full backtracking in Python.
2. Align two Kashmiri sentences and visualize the alignment tokens: `Match`, `Sub`, `Del`, `Ins`.
3. Compute micro-averaged and macro-averaged WER to observe skew on synthetic outlier sentences.

---

## Lab Script: `lab_levenshtein.py`

```python
from src.evaluation.metrics import compute_levenshtein_ops, compute_corpus_wer

def run_metric_lab():
    ref = "کٲشُر چھُ اَکھ قٔدیٖم زَبان"
    hyp = "کٲشُر چھُ زَبان" # Deleted 'اَکھ' and 'قٔدیٖم'
    
    S, D, I, C = compute_levenshtein_ops(ref.split(), hyp.split())
    print(f"Substitutions: {S} | Deletions: {D} | Insertions: {I} | Matches: {C}")
    assert D == 2 and S == 0 and I == 0 and C == 3, "Alignment calculation error!"
    
    wer = (S + D + I) / (S + D + C)
    print(f"Calculated Sentence WER: {wer*100:.2f}% (Expected 40.00%)")
    print("SUCCESS: Levenshtein alignment engine validated!")

if __name__ == "__main__":
    run_metric_lab()
```
