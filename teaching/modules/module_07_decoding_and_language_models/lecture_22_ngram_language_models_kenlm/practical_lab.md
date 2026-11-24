# Practical Lab: Training and Evaluating a KenLM Language Model

---

## Lab Objectives
1. Train an n-gram language model on normalized Kashmiri text.
2. Calculate corpus perplexity ($PPL$).
3. Compare unigram continuation probabilities vs. raw frequency counts.

---

## Lab Script: `lab_lm_evaluation.py`

```python
import numpy as np

def compute_perplexity(log10_scores, total_words):
    # PPL = 10 ^ (-1/N * sum(log10 P))
    cross_entropy = -sum(log10_scores) / total_words
    ppl = 10 ** cross_entropy
    return ppl

# Test on simulated sentence scores
simulated_log10_scores = [-4.2, -3.1, -5.8, -2.4] # 4 sentences
word_counts = [5, 4, 7, 3] # 19 total words

ppl = compute_perplexity(simulated_log10_scores, sum(word_counts))
print(f"Evaluated Test Perplexity (PPL): {ppl:.2f}")
assert ppl > 1.0, "Perplexity calculation error!"
print("SUCCESS: Language model perplexity calculation validated!")

if __name__ == "__main__":
    pass
```
