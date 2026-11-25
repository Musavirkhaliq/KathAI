# Solutions: Lecture 29 (E22-01 Scaling and Breakthrough)

---

## Solution to Exercise 1
1. Temperature calibration gain:

$$24.94\% - 24.74\% = \mathbf{0.20\% \text{ absolute}}$$

2. Language model and beam tuning gain:

$$24.74\% - 24.12\% = \mathbf{0.62\% \text{ absolute}}$$

3. Gain breakdown:
   - Temperature: $\frac{0.20}{0.82} \times 100 = \mathbf{24.4\%}$
   - LM & Beam: $\frac{0.62}{0.82} \times 100 = \mathbf{75.6\%}$

---

## Solution to Exercise 2
1. Total audio seconds per epoch:

$$106,487 \times 5.9 \text{ s} \approx 628,273 \text{ seconds} \approx 174.52 \text{ hours}$$

2. Total audio frames processed over 20 epochs ($50 \text{ frames/sec}$):

$$N_{\text{frames}} = 628,273 \times 50 \times 20 \approx 6.28 \times 10^8 \text{ frames}$$

3. Training FLOPs:
   Using the standard heuristic $\text{FLOPs} \approx 6 \times P \times N_{\text{tokens}}$:

$$\text{FLOPs} \approx 6 \times (3.17 \times 10^8) \times (6.28 \times 10^8) \approx \mathbf{1.19 \times 10^{18} \text{ FLOPs}} \approx 1.19 \text{ ExaFLOPs}$$
