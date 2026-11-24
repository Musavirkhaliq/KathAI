# Instructor Delivery Script: Lecture 30 (Checkpoint Forensics: S20R, S22C, S22E)

## Timetable (90 Minutes)
```
00:00 - 00:20 | The Checkpoint Nightmare: Which .pt File is the Real SOTA?
00:20 - 00:45 | Cryptographic Auditing: Tensor Hashing vs. File Hashing
00:45 - 01:10 | Reconciling S20R, S22C, and S22E
01:10 - 01:25 | Code Walkthrough: src/evaluation/audit_checkpoint.py
01:25 - 01:30 | Module 09 Synthesis & Exam Preparation
```

---

## Instructor Notes & Dialogue

### 00:00 - 00:20 | The Opening Challenge
- Pull up a terminal on the projector:
  ```bash
  ls -lh checkpoints/
  # Shows:
  # -rw-r--r-- 1.26G model_s20r.pt
  # -rw-r--r-- 1.26G model_s22c.pt
  # -rw-r--r-- 1.26G model_s22e.pt
  ```
- Look at the students:
  > "Three files. All three are 1.26 Gigabytes.
  > The filenames say 's20r', 's22c', 's22e'.
  > Which one produced the 24.12% WER published in the paper?
  > If you pick the wrong file, your production system runs with a 2% error penalty.
  > You cannot guess. You cannot assume based on modification time.
  > Today, we learn the discipline of **cryptographic checkpoint forensics**.
  > We will inspect the state dict, calculate tensor hashes, and establish the golden hash that locks in reproducibility forever."
