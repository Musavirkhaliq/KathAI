# Solutions: Lecture 25 (Scientific Benchmarking Methodology)

---

## Solution to Exercise 1
1. Errors per domain:
   $$E_1 = 4000 \times 0.125 = \mathbf{500 \text{ errors}}$$
   $$E_2 = 6000 \times 0.220 = \mathbf{1320 \text{ errors}}$$
   $$E_3 = 10000 \times 0.380 = \mathbf{3800 \text{ errors}}$$
2. Composite micro-averaged WER:
   $$\text{WER}_{\text{composite}} = \frac{500 + 1320 + 3800}{4000 + 6000 + 10000} = \frac{5620}{20000} = \mathbf{28.10\%}$$
3. Macro-averaged WER:
   $$\text{WER}_{\text{macro}} = \frac{12.5\% + 22.0\% + 38.0\%}{3} = \frac{72.5\%}{3} = \mathbf{24.17\%}$$

---

## Solution to Exercise 2
Two critical invariants were violated:
1. **Invariant 3 (Certified Resampling)**: Bilinear interpolation introduces severe high-frequency roll-off and passband distortion compared to certified Kaiser sinc filtering.
2. **Invariant 4 (Frozen Decoder Hyperparameters)**: Tuning $\alpha$ separately for each test domain constitutes **test-set hyperparameter contamination (leakage)**. In real production deployments, an ASR system cannot tune hyperparameters on unseen user speech before transcribing it.
