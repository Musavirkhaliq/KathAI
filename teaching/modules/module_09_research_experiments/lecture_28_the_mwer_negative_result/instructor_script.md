# Instructor Delivery Script: Lecture 28 (The MWER Sequence Loss Negative Result)

## Timetable (90 Minutes)
```
00:00 - 00:20 | The Theoretical Promise: Why Not Optimize WER Directly?
00:20 - 00:45 | Mathematical Derivation of Minimum Word Error Rate (MWER)
00:45 - 01:10 | Autopsy of the Stage 19 Disaster: The 13% WAAZ Collapse
01:10 - 01:25 | Code Inspection: stages/stage_19/mwer_loss.py
01:25 - 01:30 | Why Negative Results Are Essential to Scientific Progress
```

---

## Instructor Notes & Dialogue

### 00:45 - 01:10 | The False Breakthrough
- Address the class:
  > "In Stage 19, the intern ran MWER on Bolbosh studio speech.
  > On Friday, the intern posted in the team chat: 'WER dropped from 14.2% to 12.1%! We have a breakthrough!'
  > Everyone celebrated.
  > Then on Monday, we ran the model on the full 8-domain benchmark.
  > On WAAZ spontaneous sermons, the error rate didn't improve—it spiked from 55% to 68%!
  > Overall multi-domain WER became 32.4%, far worse than the baseline!
  > What happened?
  > The model over-specialized to the clean studio distribution. It discarded its general phonetic acoustic robustness in order to shave off a few edit distance points on clean sentences.
  > In science, a **negative result** that teaches you why a method fails is just as valuable as a breakthrough.
  > Stage 19 taught us that in low-resource speech, Maximum Likelihood CTC is a superior regularizer."
