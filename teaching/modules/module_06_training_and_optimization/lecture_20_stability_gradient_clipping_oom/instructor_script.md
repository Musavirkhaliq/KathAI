# Instructor Delivery Script: Lecture 20 (Numerical Stability, Gradient Explosions, and OOM)

## Timetable (90 Minutes)
```
00:00 - 00:20 | The 3:00 AM Crash: Why Variable-Length Speech Breaks GPUs
00:20 - 00:45 | Memory Fragmentation and the PyTorch Caching Allocator
00:45 - 01:10 | Graceful OOM Exception Handling in Production Training
01:10 - 01:25 | NaN Containment and CTC zero_infinity Mechanics
01:25 - 01:30 | Module 06 Synthesis & Review
```

---

## Instructor Notes & Dialogue

### 00:00 - 00:20 | The OOM Demonstration
- Look at the students:
  > "You start a training run on Friday afternoon. It's epoch 14 out of 20. Loss is looking great.
  > You go to sleep. You wake up Saturday morning, check your terminal, and see:
  > `torch.cuda.OutOfMemoryError: Tried to allocate 1.42 GiB`.
  > Your entire weekend of computing is destroyed.
  > Why did it crash on step 14,210 when it worked fine on step 14,209?
  > Because an 18.5-second audio clip showed up in the same batch as three other long clips!
  > Today, we learn how to make our training loops bulletproof.
  > A professional ML systems engineer never lets an OOM crash their training run."
