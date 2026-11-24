# Instructor Delivery Script: Lecture 22 (N-Gram Language Models and KenLM)

## Timetable (90 Minutes)
```
00:00 - 00:20 | The Prior: Why Acoustics Alone Cannot Disambiguate Homophones
00:20 - 00:50 | The San Francisco Example: Deriving Kneser-Ney Continuation Probabilities
00:50 - 01:15 | Modified Kneser-Ney Three-Tier Discounting Mathematics
01:15 - 01:25 | KenLM C++ Trie Architecture and Binary Probing
01:25 - 01:30 | Synthesis & Practical Commands
```

---

## Instructor Notes & Dialogue

### 00:20 - 00:50 | The Continuation Probability Story
- Write on the board:
  - `"I want to read a book about ________"`
  - Candidate 1: `"Kashmir"`
  - Candidate 2: `"Francisco"`
- Look at the students:
  > "Which word is more likely in English?
  > In a raw text corpus, the word 'Francisco' might appear 100,000 times!
  > Does that mean 'book about Francisco' is likely?
  > No! Because 'Francisco' almost exclusively appears after one single word: 'San'!
  > It has high frequency, but almost zero continuation versatility!
  > Standard MLE smoothing says: 'Francisco is frequent, so give it high backoff probability.'
  > Kneser-Ney says: 'Nonsense! Francisco only has one predecessor. It is not a versatile continuation word.'
  > In Kashmiri, this prevents proper noun fragments from polluting acoustic backoff paths."
