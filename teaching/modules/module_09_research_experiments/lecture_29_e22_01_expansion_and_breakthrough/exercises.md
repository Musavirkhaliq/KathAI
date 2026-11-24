# Exercises: Lecture 29 (E22-01 Scaling and Breakthrough)

---

## Exercise 1: Decoding Optimization Attribution
Review the progression from 24.94% to 24.12%:
1. How much absolute WER reduction was contributed purely by acoustic temperature calibration ($T=1.00 \to 1.08$)?
2. How much additional reduction was contributed by tuning $\alpha$, $\beta$, and beam width ($0.30 \to 0.50, 1.00 \to 1.50, 64 \to 100$)?
3. What percentage of the total 0.82% gain came from acoustic calibration vs. language model shallow fusion?

---

## Exercise 2: FLOP Calculation for 106k Training
Given:
- 106,487 utterances with average duration 5.9 seconds.
- 20 epochs.
- XLS-R 300M parameter model requires $\approx 6 \times 10^8 \text{ FLOPs}$ per parameter per forward-backward sample.
Calculate the total theoretical floating-point operations (FLOPs) required to train model E22-01.
