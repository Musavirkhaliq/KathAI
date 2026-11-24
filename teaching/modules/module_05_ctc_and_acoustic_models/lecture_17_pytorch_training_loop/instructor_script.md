# Instructor Delivery Script: Lecture 17 (The Complete PyTorch CTC Training Loop)

## Timetable (90 Minutes)
```
00:00 - 00:20 | Systems Architecture: The 5 Pillars of Stable Acoustic Training
00:20 - 00:45 | Mixed Precision Mechanics: FP16 Underflows and Dynamic GradScaler
00:45 - 01:10 | Gradient Accumulation and Gradient Clipping (Norm <= 1.0)
01:10 - 01:25 | Line-by-Line Code Inspection: src/training/trainer.py
01:25 - 01:30 | Module 05 Synthesis & Course Midpoint Review
```

---

## Instructor Notes & Dialogue

### 00:20 - 00:45 | GradScaler Demonstration
- Write on the board:
  - FP16 minimum positive normal: $2^{-14} \approx 6.1 \times 10^{-5}$
  - FP16 minimum subnormal: $2^{-24} \approx 5.96 \times 10^{-8}$
- Look at the students:
  > "If your backward pass computes a gradient of $10^{-9}$, what does an FP16 GPU register do?
  > It flushes it to absolute zero!
  > If that happens across 10 layers, your model stops updating.
  > GradScaler says: 'Multiply the loss by 32,768.'
  > Now $10^{-9}$ becomes $3.27 \times 10^{-5}$, which fits comfortably in FP16!
  > Right before the weights update, you divide by 32,768.
  > That is how we get $3\times$ faster training without losing a single bit of precision."
