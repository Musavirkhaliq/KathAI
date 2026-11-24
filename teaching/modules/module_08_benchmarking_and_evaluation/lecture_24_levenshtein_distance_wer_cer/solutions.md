# Solutions: Lecture 24 (Levenshtein Distance, WER, and CER)

---

## Solution to Exercise 1
1. DP Table:
```
       eps   ک   ا   ش   ر   ی
eps     0    1   2   3   4   5
 ک      1    0   1   2   3   4
 ٲ      2    1   1   2   3   4
 ش      3    2   2   1   2   3
 ر      4    3   3   2   1   2
```
2. Backtracking path:
   - $(4, 5) \to (4, 4)$ with Insertion of `'ی'`
   - $(4, 4) \to (3, 3)$ with Match of `'ر'`
   - $(3, 3) \to (2, 2)$ with Match of `'ش'`
   - $(2, 2) \to (1, 1)$ with Substitution of `'ٲ'` with `'ا'`
   - $(1, 1) \to (0, 0)$ with Match of `'ک'`
   - Total operations: 1 Substitution, 0 Deletions, 1 Insertion, 3 Matches.
3. Character Error Rate:
   $$\text{CER} = \frac{S + D + I}{N} = \frac{1 + 0 + 1}{4} = \frac{2}{4} = \mathbf{50.0\%}$$

---

## Solution to Exercise 2
1. Sentence WERs:
   $$\text{WER}_A = \frac{2}{2} = 100.0\%, \quad \text{WER}_B = \frac{0}{48} = 0.0\%$$
2. Macro-averaged WER:
   $$\text{WER}_{\text{macro}} = \frac{100.0\% + 0.0\%}{2} = \mathbf{50.0\%}$$
3. Micro-averaged WER:
   $$\text{WER}_{\text{micro}} = \frac{2 + 0}{2 + 48} = \frac{2}{50} = \mathbf{4.0\%}$$
4. Explanation: Macro-averaging treats an utterance with 2 words with equal weight to an utterance with 48 words. The model transcribed 48 out of 50 total spoken words correctly ($96\%$ word accuracy), yet macro-averaging reports an erroneous $50\%$ error rate.
