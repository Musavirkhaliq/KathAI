# Instructor Delivery Script: Lecture 24 (Levenshtein Distance, WER, and CER)

## Timetable (90 Minutes)
```
00:00 - 00:20 | The Core Metric: Substitutions, Deletions, and Insertions
00:20 - 00:45 | Step-by-Step Walkthrough of the DP Edit Matrix
00:45 - 01:10 | The Mathematical Danger: Micro vs. Macro Averaging
01:10 - 01:25 | Line-by-Line Code Inspection: src/evaluation/metrics.py
01:25 - 01:30 | Summary & Homework Assignment
```

---

## Instructor Notes & Dialogue

### 00:45 - 01:10 | Micro vs. Macro Averaging on the Board
- Write two test sentences on the board:
  - Sentence 1: Reference = 1 word (`"سلام"`). Hypothesis = `"سلام سلام سلام"` (2 insertions).

$$\text{WER}_1 = \frac{2}{1} = 200\%$$

  - Sentence 2: Reference = 99 words. Hypothesis = 99 correct words.

$$\text{WER}_2 = \frac{0}{99} = 0\%$$

- Ask the students:
  > "What is the macro-averaged WER?
  > Average of 200% and 0% is 100%!
  > A paper reports: 'Our model achieved 100% error rate.'
  > But look at the total words: out of 100 words spoken, the model got 99 words completely right and hallucinated two words!
  > The true micro-averaged WER is:

$$\text{WER} = \frac{2}{1 + 99} = \frac{2}{100} = 2.0\%$$

  > In this course, if an engineer reports macro-averaged WER, their submission is rejected immediately.
  > Always micro-average across the entire corpus!"
