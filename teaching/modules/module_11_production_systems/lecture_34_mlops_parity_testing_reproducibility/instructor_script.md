# Instructor Delivery Script: Lecture 34 (MLOps and Golden Logit Parity Testing)

## Timetable (90 Minutes)
```
00:00 - 00:20 | The Fragility of ML Pipelines: Why Silent Regressions Happen
00:20 - 00:45 | Golden Logit Parity: Numerical Floating-Point Verification
00:45 - 01:10 | Automated CI/CD Testing in Speech Repositories
01:10 - 01:25 | Code Inspection: tests/test_inference_parity.py
01:25 - 01:30 | Synthesis & Review
```

---

## Instructor Notes & Dialogue

### 00:20 - 00:45 | The Parity Demonstration
- Look at the students:
  > "You refactor a 500-line messy script into clean, beautiful object-oriented classes.
  > You run it. It outputs text. You feel proud.
  > But how do you know your refactored code didn't change a single number?
  > Did you accidentally shift the audio by 1 sample?
  > Did your resampler round 249 frames down to 248?
  > In Stage 27, we didn't just 'hope' it worked.
  > We compared the raw floating-point logits output by the GPU.
  > If the numbers differed by more than $0.00001$, the commit was rejected!
  > That is the standard of scientific MLOps."
