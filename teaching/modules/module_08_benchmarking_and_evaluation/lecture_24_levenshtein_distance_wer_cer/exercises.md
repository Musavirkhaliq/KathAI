# Exercises: Lecture 24 (Levenshtein Distance, WER, and CER)

---

## Exercise 1: DP Matrix Evaluation
Given:
- Reference: `R = ("ک", "ٲ", "ش", "ر")`
- Hypothesis: `H = ("ک", "ا", "ش", "ر", "ی")`
1. Construct the complete $5 \times 6$ Levenshtein dynamic programming table.
2. Backtrack from cell $(4, 5)$ to find the optimal sequence of edit operations.
3. Calculate the Character Error Rate (CER).

---

## Exercise 2: Micro vs. Macro Inversion
An evaluation corpus contains two test sentences:
- Utterance A: Reference has 2 words. Hypothesis has 2 substitutions.
- Utterance B: Reference has 48 words. Hypothesis has 48 correct words.
1. Calculate the sentence-level WER for Utterance A and Utterance B.
2. Calculate the Macro-averaged WER across the two utterances.
3. Calculate the Micro-averaged WER.
4. Explain why Macro-averaged WER gives an absurdly misleading impression of model accuracy.
