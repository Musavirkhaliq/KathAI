# Instructor Delivery Script: Lecture 29 (E22-01 Scaling and the Breakthrough)

## Timetable (90 Minutes)
```
00:00 - 00:20 | The Convergence: Bringing All 22 Stages Together
00:20 - 00:45 | Deconstructing the 106,487 Utterance Training Run
00:45 - 01:10 | Forensic Autopsy: Resolving the 4 Conflicting WER Numbers
01:10 - 01:25 | Code Walkthrough: configs/experiments/e22_01.yaml
01:25 - 01:30 | Synthesis & Practical Certification
```

---

## Instructor Notes & Dialogue

### 00:45 - 01:10 | Resolving the 4 Numbers
- Write on the board:
  - `24.74%`
  - `24.40%`
  - `24.12%`
  - `24.94%`
- Look at the students:
  > "If you look at the research reports from November 2023, you find four different WER numbers for E22-01.
  > Did the model weights change? Did someone re-train it on a secret cluster?
  > No!
  > The checkpoint hash was identical down to the last byte.
  > What changed was the **decoder**.
  > At 24.94%, temperature was 1.0.
  > When we calibrated temperature to 1.08, it dropped to 24.74%.
  > When we expanded the beam to 100 and tuned $\alpha$ to 0.50 and $\beta$ to 1.50, it dropped to 24.12%!
  > This is the ultimate lesson in speech systems engineering: **inference parameters matter just as much as training loss**."
