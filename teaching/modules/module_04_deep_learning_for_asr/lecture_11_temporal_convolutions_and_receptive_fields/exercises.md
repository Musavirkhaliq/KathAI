# Exercises: Lecture 11 (Temporal Convolutions and Receptive Fields)

---

## Exercise 1: Custom Strided Architecture
Suppose you design an alternative ASR convolutional front-end with 5 layers:
- Layer 1: $K=7, S=4$
- Layer 2: $K=5, S=2$
- Layer 3: $K=5, S=2$
- Layer 4: $K=3, S=2$
- Layer 5: $K=3, S=1$
1. Calculate the total temporal downsampling factor $D$.
2. Derive the total receptive field $R_5$ in samples.
3. At 16 kHz, what is the frame shift (ms) and receptive field duration (ms)?

---

## Exercise 2: Frame Count Calculation
For the 7-layer XLS-R feature extractor:
1. Write the explicit formula for $T_\text{frames}$ as a function of $T_\text{samples}$.
2. Calculate the exact output frame count for an input audio clip of 4.35 seconds at 16,000 Hz.
