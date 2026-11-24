# Instructor Delivery Script: Lecture 13 (Self-Supervised Learning and XLS-R)

## Timetable (90 Minutes)
```
00:00 - 00:20 | The Low-Resource Dilemma: Why Supervised Learning is Impossible for Kashmiri
00:20 - 00:45 | The wav2vec 2.0 Triad: Latents, Gumbel Quantization, and Context Vectors
00:45 - 01:10 | Mathematical Derivation of the InfoNCE Masked Contrastive Loss
01:10 - 01:25 | The Power of XLS-R: Cross-Lingual Transfer from 436,000 Hours
01:25 - 01:30 | Module 04 Synthesis & Midterm Preparation
```

---

## Instructor Notes & Dialogue

### 00:20 - 00:45 | The Triad on the Board
- Draw three horizontal tracks on the board:
  - Top track: Raw audio $\mathbf{x} \to$ Conv $\mathbf{z}_t$.
  - Middle track: Gumbel-Softmax Codebook $\mathbf{q}_t$.
  - Bottom track: Masked Transformer Context $\mathbf{c}_t$.
- Step into the center:
  > "How does a human baby learn language?
  > Do parents give babies a CSV file of transcribed audio with timestamps?
  > No! The baby listens to thousands of hours of unlabelled acoustic reality.
  > The baby's brain learns that certain sounds belong together and certain sounds contrast.
  > Self-supervised learning is the mathematical realization of that infant learning process.
  > The network masks out 200 milliseconds of audio and asks: 'Predict what sound belonged here.'
  > To answer that question, the 24 layers of the Transformer must learn grammar, phonetics, and vocal physics."
