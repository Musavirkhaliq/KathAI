# Instructor Delivery Script: Lecture 19 (Empirical Scaling Analysis: From B-01 to B-02C)

## Timetable (90 Minutes)
```
00:00 - 00:20 | The Core Question: How Much Data Do We Really Need?
00:20 - 00:45 | Historical Evolution: B-01 (20h) -> B-02 (60h) -> B-02C (112h)
00:45 - 01:10 | Mathematical Derivation of the Power-Law Scaling Curve
01:10 - 01:25 | Domain Resilience Analysis: Why WAAZ Lagged Studio Clean
01:25 - 01:30 | Synthesis & Practical Takeaways
```

---

## Instructor Notes & Dialogue

### 00:20 - 00:45 | The Scaling Graph
- Plot a curve on the board:
  - X-axis: Hours (20, 60, 112, 174).
  - Y-axis: WER % (50%, 40%, 30%, 20%).
- Plot the points:
  - B-01: (20h, 42.1%)
  - B-02: (60h, 32.8%)
  - B-02C: (112h, 25.82%)
  - E22-01: (174h, 24.12%)
- Turn to the students:
  > "Look at the slope of this curve.
  > Going from 20 hours to 60 hours is like night and day. You drop 10 percentage points of error!
  > But look at the tail. Going from 112 hours to 174 hours only buys you 1.7 percentage points.
  > Why?
  > Because at 112 hours, the acoustic model already knows how to pronounce every phoneme.
  > The remaining errors aren't acoustic—they are syntactic, homophonic, and dialectal.
  > To break through 20% WER, you don't need 500 more hours of audio; you need a better language model, neural rescoring, and domain adaptation."
