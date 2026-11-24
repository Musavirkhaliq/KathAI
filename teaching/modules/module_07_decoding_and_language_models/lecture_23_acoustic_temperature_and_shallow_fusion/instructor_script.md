# Instructor Delivery Script: Lecture 23 (Acoustic Temperature and Shallow Fusion)

## Timetable (90 Minutes)
```
00:00 - 00:20 | The Tug-of-War: Acoustic Likelihood vs. Language Prior
00:20 - 00:45 | Overconfidence and Temperature Scaling Calibration (T = 1.08)
00:45 - 01:10 | The Brevity Bias and the Mathematical Role of Beta
01:10 - 01:25 | 2D Grid Search Execution on Validation Sets
01:25 - 01:30 | Module 07 Synthesis & Wrap-up
```

---

## Instructor Notes & Dialogue

### 00:20 - 00:45 | The Overconfidence Demonstration
- Plot on the board: Softmax output without temperature vs. with temperature ($T=1.08$).
- Address the students:
  > "Why does an acoustic model act like an arrogant know-it-all?
  > Because modern deep neural networks have millions of parameters and zero incentive to admit uncertainty.
  > If a model is 99.9% confident that it heard 'cat' when the true word was 'cot', beam search prunes 'cot' immediately.
  > The language model never even gets to see 'cot'!
  > Dividing by $T = 1.08$ is a dose of humility for your neural network.
  > It softens 0.999 down to 0.92, keeping alternate phonetic paths alive in the beam so the language model can save the transcription."
