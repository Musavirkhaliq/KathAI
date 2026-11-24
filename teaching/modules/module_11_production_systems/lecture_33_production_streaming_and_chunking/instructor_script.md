# Instructor Delivery Script: Lecture 33 (Production Streaming and Chunking)

## Timetable (90 Minutes)
```
00:00 - 00:20 | The Production Wall: Why You Cannot Feed a 30-Minute File to a Transformer
00:20 - 00:45 | Deconstructing Method C2: 8-Second Windows and 1-Second Overlap
00:45 - 01:10 | Boundary Slicing and Logit Cross-Fading
01:10 - 01:25 | Code Inspection: scripts/asr_service.py and Real-Time Factor (RTF)
01:25 - 01:30 | Synthesis & Review
```

---

## Instructor Notes & Dialogue

### 00:00 - 00:20 | The Production Demo
- Open a terminal and check the running service:
  ```bash
  ps aux | grep asr_service.py
  ```
- Look at the students:
  > "In academia, we test on 5-second clean clips.
  > In production, a user uploads a 45-minute recording of a religious lecture.
  > If you pass that 45-minute file into PyTorch, your GPU will crash with an OOM before the first attention layer finishes.
  > How does our production service handle long audio?
  > It uses **Method C2**: a sliding 8-second window with a 1-second overlap.
  > Today, we study how to stitch those chunks together so that a word split right at the 8-second boundary is transcribed with 100% accuracy."
