# Solutions: Lecture 09 (Dataset Manifest Engineering)

---

## Solution to Exercise 1
1. Contamination rate:
   $$C = \frac{32}{200} = 16.0\% \text{ of utterances (or speakers)}$$
2. The reported 18.2% WER is untrustworthy because the acoustic model has memorized the vocal tract characteristics of the leaked speakers. On a strictly disjoint split, WER will systematically increase (typically by 8–18% absolute) as the model encounters genuinely unseen vocal tracts.

---

## Solution to Exercise 2
In PyTorch, all sequences in a batch are padded to the longest sequence in that batch.
If an 18.0s clip (288,000 samples) and a 1.5s clip (24,000 samples) share a batch, the 1.5s clip requires 264,000 padding zeros—wasting **91.7% of its computation**.
Bucket sorting groups clips of similar length (e.g., 1.0–2.5s, 2.5–5.0s), boosting batch padding efficiency to $> 92\%$.
