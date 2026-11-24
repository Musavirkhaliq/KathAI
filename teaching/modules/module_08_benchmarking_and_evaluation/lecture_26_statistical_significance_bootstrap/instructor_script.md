# Instructor Delivery Script: Lecture 26 (Statistical Significance and Bootstrap Resampling)

## Timetable (90 Minutes)
```
00:00 - 00:20 | The 0.28% Question: Scientific Breakthrough or Statistical Mirage?
00:20 - 00:45 | Why Student's t-test Fails on Non-Normal Speech Errors
00:45 - 01:10 | Non-Parametric Paired Bootstrap Resampling (B = 10,000)
01:10 - 01:25 | Line-by-Line Code Inspection: src/evaluation/significance.py
01:25 - 01:30 | Synthesis & Transition to Module 09 (Research Archaeology)
```

---

## Instructor Notes & Dialogue

### 00:00 - 00:20 | The 0.28% Demonstration
- Write on the board:
  - Model 1: `WER = 24.12%`
  - Model 2: `WER = 24.40%`
  - Difference: `0.28%`
- Look at the students:
  > "If you publish a paper saying Model 1 is better than Model 2 because of 0.28%, a competent reviewer will ask:
  > 'What is your p-value?'
  > If your test set has 500 sentences, a 0.28% difference is literally **three words**!
  > If one speaker coughs during recording, your 0.28% advantage disappears!
  > Today, we learn how to prove whether a result is statistically solid or an artifact of chance.
  > In science, if you cannot prove statistical significance, you have not discovered anything."
