# Instructor Delivery Script: Lecture 31 (Neural Rescoring and Oracle Analysis)

## Timetable (90 Minutes)
```
00:00 - 00:20 | The Ceiling of First-Pass Search: Why 5-Grams Max Out
00:20 - 00:45 | Computing the Oracle WER: Finding the Hidden Jewels in the Beam
00:45 - 01:10 | The 13.40% Oracle Discovery in Stage 24
01:10 - 01:25 | Code Inspection: src/decoding/rescorer.py
01:25 - 01:30 | Synthesis & Homework Assignment
```

---

## Instructor Notes & Dialogue

### 00:20 - 00:45 | The Oracle Demonstration
- Write on the board:
  - `1-Best SOTA WER: 24.12%`
  - `100-Best Oracle WER: 13.40%`
  - `Potential Unlocked: 10.72%`
- Look at the students:
  > "Look at that gap. 10.72 percentage points!
  > That means if you had an omniscient language model that could pick the best candidate out of your top 100 beams, your Word Error Rate would plummet from 24% to 13%!
  > The acoustic model didn't fail. It actually heard the right words.
  > The words are sitting right there in candidate #4 or candidate #7.
  > But your 5-gram KenLM, which can only look 4 words into the past, ranked candidate #1 with a common word instead.
  > Second-pass neural rescoring is how we bring deep language intelligence into speech recognition."
