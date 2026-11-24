# Instructor Delivery Script: Lecture 08 (Speech Corpora Collection and Curation)

## Timetable (90 Minutes)
```
00:00 - 00:20 | Low-Resource Realities: The 12 Sources of EduCasheer
00:20 - 00:45 | Acoustic Heterogeneity: Studio vs. WAAZ Spontaneous Sermons
00:45 - 01:10 | The 4-Stage Curation Pipeline: WADA-SNR and Clipping Detection
01:10 - 01:25 | Code Walkthrough: src/data/curation.py
01:25 - 01:30 | Key Takeaways & Lab Introduction
```

---

## Instructor Notes & Dialogue

### 00:20 - 00:45 | Acoustic Heterogeneity
- Play two 5-second audio snippets:
  - Snippet 1: Clean Bolbosh studio recording.
  - Snippet 2: WAAZ sermon (loud reverberant echo, distant crowd coughs, loudspeaker distortion).
- Address the class:
  > "If you train your model only on Snippet 1, your model gets 12% WER in your lab and 85% WER when a real Kashmiri farmer tries to use it in Anantnag.
  > But if you throw Snippet 2 directly into your training without filtering, the CTC loss explodes because the crowd noise causes the model to hallucinate blank tokens.
  > Data curation is the science of knowing what to keep, what to clean, and what to quarantine."
