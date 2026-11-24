# Instructor Delivery Script: Lecture 18 (Fine-Tuning Dynamics and Schedules)

## Timetable (90 Minutes)
```
00:00 - 00:20 | The Dilemma: Fine-Tuning vs. Scratch Training
00:20 - 00:45 | Mathematical Formulation: Layer-wise Learning Rate Decay (LLRD)
00:45 - 01:10 | Deriving the Warmup + Cosine Annealing Trajectory
01:10 - 01:25 | Code Inspection: Implementing LLRD in PyTorch Optimizer Groups
01:25 - 01:30 | Synthesis & Homework Assignment
```

---

## Instructor Notes & Dialogue

### 00:20 - 00:45 | LLRD on the Board
- Draw a 24-story skyscraper on the board representing the 24 Transformer layers.
- At the top floor (Layer 24), write: `LR = 3e-4`.
- At floor 12, write: `LR = 9.5e-6`.
- At floor 1, write: `LR = 3e-7`.
- Turn to the students:
  > "Why do we treat floor 1 differently from floor 24?
  > Because floor 1 knows how sound travels through air. It took 436,000 hours of multi-million dollar computing for Meta AI to teach floor 1 that physics!
  > If you hit floor 1 with a learning rate of $3 \times 10^{-4}$ on your tiny Kashmiri dataset, you erase that universal physics in five minutes.
  > But floor 24 doesn't know Kashmiri. Floor 24 needs to learn that the sound /əː/ maps to the grapheme `ٲ`.
  > LLRD gives high learning rates to the students who need to learn (top layers), and tiny learning rates to the masters who already know (bottom layers)."
