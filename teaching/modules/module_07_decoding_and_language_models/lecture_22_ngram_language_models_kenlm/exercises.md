# Exercises: Lecture 22 (N-Gram Language Models and KenLM)

---

## Exercise 1: Continuation Probability Calculation
A toy training corpus has the following 2-gram counts:
- `("سان", "فرانسسکو")`: count = 500
- `("بٔڑ", "فرانسسکو")`: count = 0
- Total distinct contexts where `"فرانسسکو"` appears: `|\{ w' : C(w', "فرانسسکو") > 0 \}| = 1`.
- Total distinct 2-gram pairs in the corpus: 20,000.
1. Calculate the Kneser-Ney continuation probability `P_{\text{continuation}}("فرانسسکو")`.
2. If total words in the corpus = 100,000, calculate the standard unigram MLE probability `P_{\text{MLE}}("فرانسسکو")`.
3. Explain why continuation probability prevents spurious backoff hallucinations.

---

## Exercise 2: Perplexity Calculation
A 5-gram language model evaluates a test corpus of 500 words and assigns a cumulative log-likelihood:

$$\sum_{i=1}^{500} \log_{10} P(w_i \mid w_{\lt i}) = -1,250.0$$

1. Calculate the cross-entropy per word.
2. Calculate the test perplexity (PPL).
