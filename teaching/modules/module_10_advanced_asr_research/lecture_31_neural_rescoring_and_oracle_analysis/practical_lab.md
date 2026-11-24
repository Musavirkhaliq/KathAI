# Practical Lab: N-Best List Rescoring and Oracle Auditing

---

## Lab Objectives
1. Generate 10-best candidate lists from synthetic beam search outputs.
2. Calculate the Oracle WER across a test set.
3. Implement a simple neural language model rescorer that reduces 1-best WER toward the oracle bound.

---

## Lab Script: `lab_oracle_rescore.py`

```python
from src.evaluation.oracle import compute_oracle_wer

def run_oracle_lab():
    refs = ["کٲشُر چھُ اَکھ قٔدیٖم زَبان", "آب چھُ زِنٛدَگی"]
    nbest = [
        ["کٲشُر چھُ زَبان", "کٲشُر چھُ اَکھ قٔدیٖم زَبان", "کٲشُر زَبان"],
        ["آب چھُ زِنٛدَگی", "آب زِنٛدَگی"]
    ]
    
    oracle = compute_oracle_wer(nbest, refs)
    print(f"Computed Oracle WER: {oracle*100:.2f}% (Expected 0.00%)")
    assert oracle == 0.0, "Oracle calculation failed!"
    print("SUCCESS: Oracle computation validated!")

if __name__ == "__main__":
    run_oracle_lab()
```
