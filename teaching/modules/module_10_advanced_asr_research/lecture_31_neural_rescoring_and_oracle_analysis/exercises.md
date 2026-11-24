# Exercises: Lecture 31 (Neural Rescoring and Oracle Analysis)

---

## Exercise 1: Oracle WER Calculation
A model evaluates two sentences:
- Sentence 1 (Reference: 4 words):
  - Candidate 1: 2 errors
  - Candidate 2: 1 error
  - Candidate 3: 3 errors
- Sentence 2 (Reference: 6 words):
  - Candidate 1: 1 error
  - Candidate 2: 0 errors
1. What is the standard 1-Best WER?
2. What is the Oracle WER?
3. Calculate the potential accuracy gain achievable via second-pass rescoring.

---

## Exercise 2: Lattice vs. N-Best Rescoring
Explain why rescoring an entire lattice or WFST compact graph is computationally more powerful than rescoring an N-best list of size $N = 100$.
