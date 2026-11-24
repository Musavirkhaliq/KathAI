# Instructor Delivery Script: Lecture 09 (Dataset Manifests and Zero-Overlap Speaker Splits)

## Timetable (90 Minutes)
```
00:00 - 00:20 | The Illusion of Success: How Speaker Leakage Fakes State-of-the-Art
00:20 - 00:45 | Mathematical Formulation: Bipartite Graph Partitioning
00:45 - 01:10 | The 8 Frozen Evaluation Domains of EduCasheer
01:10 - 01:25 | Code Inspection: src/data/manifest.py and Split Verification
01:25 - 01:30 | Module 03 Synthesis & Wrap-up
```

---

## Instructor Notes & Dialogue

### 00:00 - 00:20 | The Speaker Leakage Illusion
- Write on the board:
  - `Random Split WER: 16.8%`
  - `Disjoint Speaker Split WER: 34.2%`
- Look at the students:
  > "These two numbers came from the exact same neural network checkpoint.
  > If I wanted to publish a fraudulent paper or fool an investor, I would show them 16.8%.
  > Why did the model score 16.8%?
  > Because the model didn't learn Kashmiri phonetics—it learned the voice of Speaker #14. It learned that whenever an audio file has an F0 of 142 Hz and a room resonance at 600 Hz, Speaker #14 usually talks about the weather.
  > When we evaluated that same model on a new speaker whose voice it had never heard, the model fell apart.
  > In this course, random splitting in speech recognition is considered scientific malpractice.
  > Today, we learn how to mathematically guarantee zero speaker overlap."
