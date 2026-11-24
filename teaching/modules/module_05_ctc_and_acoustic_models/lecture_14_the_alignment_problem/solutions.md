# Solutions: Lecture 14 (The Alignment Problem)

---

## Solution to Exercise 1
1. Total cells: $N \times M = 600 \times 50 = \mathbf{30,000 \text{ cells}}$.
2. Execution time: $30,000 \times 10 \times 10^{-9} \text{ s} = 3 \times 10^{-4} \text{ s} = \mathbf{0.30 \text{ ms}}$.

---

## Solution to Exercise 2
1. **Diacritic Omission**: Written texts omit vowel diacritics (*aerab*), creating ambiguity about whether a vowel is /i/, /ɨ/, or /u/.
2. **Phonemic Central Vowels**: Acoustic GMM models lack trained central vowel states.
3. **Spontaneous Elision**: In fast speech, word-final vowels are routinely elided, causing HMM forced aligners to reject valid utterances due to zero-likelihood transitions.
