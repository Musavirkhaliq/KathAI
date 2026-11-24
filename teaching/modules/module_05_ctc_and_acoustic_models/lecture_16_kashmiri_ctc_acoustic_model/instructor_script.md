# Instructor Delivery Script: Lecture 16 (Building the Kashmiri CTC Acoustic Model)

## Timetable (90 Minutes)
```
00:00 - 00:20 | Model Anatomy: Coupling XLS-R to the 47-Class Linear Head
00:20 - 00:45 | Initializing the CTC Head: Why Blank Bias Prevents Gradient Shocks
00:45 - 01:10 | Staged Fine-Tuning: Freezing the Feature Extractor to Prevent Forgetting
01:10 - 01:25 | Code Inspection: src/models/wav2vec2.py
01:25 - 01:30 | Synthesis & Transition to the PyTorch Training Loop
```

---

## Instructor Notes & Dialogue

### 00:20 - 00:45 | Blank Bias Demonstration
- Step to the board:
  > "If you have 47 classes and initialize your bias to zero, what is the probability of the blank token on step zero?
  > One divided by 47. About 2%.
  > But in CTC, what fraction of audio frames are actually blanks?
  > Over 90%!
  > If your model starts by predicting that blanks have only 2% probability, your very first backward pass generates a massive gradient explosion that punches your 24 Transformer layers in the face.
  > By setting the blank bias to +3.0, the model begins with 90% blank probability. The gradients are calm, smooth, and gentle from step one."
