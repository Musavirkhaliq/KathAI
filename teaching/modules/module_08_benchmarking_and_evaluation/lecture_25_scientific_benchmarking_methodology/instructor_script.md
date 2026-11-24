# Instructor Delivery Script: Lecture 25 (Scientific Benchmarking Methodology)

## Timetable (90 Minutes)
```
00:00 - 00:20 | The Reproducibility Crisis in Low-Resource Speech
00:20 - 00:45 | Deconstructing the 8 Frozen Domains of EduCasheer
00:45 - 01:10 | The Five Mandatory Invariants of Certified Benchmarking
01:10 - 01:25 | Code Inspection: benchmarks/run_certified_eval.py
01:25 - 01:30 | Synthesis & Discussion
```

---

## Instructor Notes & Dialogue

### 00:00 - 00:20 | The Benchmark Illusion
- Step to the board:
  > "If a research paper claims 12% WER on Kashmiri, what does that number mean?
  > Did they test on studio recordings read by professional radio announcers?
  > Did the speaker in the test set appear 50 times in their training data?
  > Did they tune their language model weight $\alpha$ separately on the test set until the number looked pretty?
  > In real life, 12% on a clean read test set is 60% when you deploy it in an open-air marketplace in Srinagar.
  > Today, we formalize the 8-Domain Protocol.
  > A benchmark is not a vanity metric—it is a scientific stress test."
