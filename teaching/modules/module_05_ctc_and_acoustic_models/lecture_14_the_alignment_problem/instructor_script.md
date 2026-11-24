# Instructor Delivery Script: Lecture 14 (The Alignment Problem in Speech Recognition)

## Timetable (90 Minutes)
```
00:00 - 00:20 | The Dimension Mismatch: 500 Acoustic Frames vs. 45 Text Characters
00:20 - 00:45 | Historical Solutions: DTW and HMM Forced Alignment Failures
00:45 - 01:10 | Why Manual Alignment Fails on Low-Resource Kashmiri
01:10 - 01:25 | The Revolutionary Concept of Latent Marginalization
01:25 - 01:30 | Transition to Lecture 15: CTC Forward-Backward Mathematics
```

---

## Instructor Notes & Dialogue

### 00:00 - 00:20 | The Opening Demonstration
- Write on the board:
  - Audio: $T = 400 \text{ frames}$ (8 seconds)
  - Text: `"کٲشُر"` ($U = 4 \text{ letters}$)
- Turn to the students:
  > "Look at the numbers on the board. 400 frames on the left. 4 characters on the right.
  > If I ask you to train a standard neural network with Cross-Entropy loss, what is the label for frame 142?
  > Is frame 142 the letter /k/? Is it the central vowel /əː/? Or is it silence?
  > Nobody knows! Even expert phonetic linguists looking at spectrograms argue over the exact millisecond a vowel transitions into a sibilant.
  > For 30 years, speech recognition was trapped in this bottleneck: you couldn't train an acoustic model without alignments, and you couldn't get alignments without an acoustic model!
  > Today, we study how CTC solved this chicken-and-egg dilemma forever."
