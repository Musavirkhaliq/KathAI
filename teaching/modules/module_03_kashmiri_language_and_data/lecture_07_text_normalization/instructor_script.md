# Instructor Delivery Script: Lecture 07 (Text Normalization and Certified Normalizer v3)

## Timetable (90 Minutes)
```
00:00 - 00:20 | The Silent Killer: How Unicode Inconsistencies Destroy ASR
00:20 - 00:45 | Homoglyph Autopsy: Kaf, Yeh, Heh, and Central Vowel Codepoints
00:45 - 01:10 | NFKC Traps: Why Standard Unicode Tools Fail on Kashmiri
01:10 - 01:25 | Code Inspection: src/normalization/kashmiri_normalizer.py
01:25 - 01:30 | Lab Preview and Normalization Invariant Testing
```

---

## Instructor Notes & Dialogue

### 00:20 - 00:45 | The Homoglyph Demonstration
- Open a terminal and run `python3`.
- Type:
  ```python
  a = "ک" # U+06A9 (Keheh)
  b = "ك" # U+0643 (Arabic Kaf)
  print(a == b) # Outputs: False
  ```
- Turn to the students:
  > "Look at the screen. To human eyes, both letters look like Kaf. But Python tells you they are NOT equal.
  > If a transcriber uses an Arabic keyboard for one sentence and an Urdu keyboard for another, your neural network thinks they are two completely different letters.
  > If you don't normalize this before training, your model wastes capacity learning two separate logits for the same acoustic sound.
  > Worse: during evaluation, if the model predicts `ک` (U+06A9) and the reference has `ك` (U+0643), your evaluation script marks it as an error!
  > You can lose 8% Word Error Rate without a single acoustic mistake."
