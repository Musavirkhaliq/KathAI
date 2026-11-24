# Instructor Delivery Script: Lecture 11 (Temporal Convolutions and Receptive Fields)

## Timetable (90 Minutes)
```
00:00 - 00:20 | The Memory Wall: Why 160,000 Audio Samples Break Self-Attention
00:20 - 00:50 | Step-by-Step Derivation of the Receptive Field (R = 400 Samples = 25 ms)
00:50 - 01:15 | Inspecting Learned Convolutions: The Emergence of Biological Filterbanks
01:15 - 01:25 | Code Inspection: XLS-R ConvFeatureExtractionModel
01:25 - 01:30 | Exercises & Takeaways
```

---

## Instructor Notes & Dialogue

### 00:20 - 00:50 | Receptive Field Derivation on the Board
- Walk through the recursive calculation: $R_l = R_{l-1} + (K_l - 1) \cdot J_{l-1}$.
- Calculate each layer on the board with student participation:
  - $R_1 = 10$
  - $R_2 = 20$
  - $R_3 = 40$
  - $R_4 = 80$
  - $R_5 = 160$
  - $R_6 = 240$
  - $R_7 = 400$
- Highlight the number 400:
  > "Divide 400 by 16,000. What do you get? Exactly 25 milliseconds.
  > The Facebook AI Research engineers didn't pick random numbers for kernel sizes and strides.
  > They designed the convolutional stack so that every single latent frame captures exactly one 25 ms quasi-stationary phonetic window!"
