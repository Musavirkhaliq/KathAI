# Solutions: Lecture 18 (Fine-Tuning Dynamics and Schedules)

---

## Solution to Exercise 1
1. Formula: $\eta_l = \eta_{\text{base}} \times \xi^{24 - l}$.
   - Layer 20: $\eta_{20} = 5 \times 10^{-4} \times (0.80)^4 = 5 \times 10^{-4} \times 0.4096 = \mathbf{2.048 \times 10^{-4}}$
   - Layer 10: $\eta_{10} = 5 \times 10^{-4} \times (0.80)^{14} = 5 \times 10^{-4} \times 0.04398 = \mathbf{2.199 \times 10^{-5}}$
   - Layer 1: $\eta_1 = 5 \times 10^{-4} \times (0.80)^{23} = 5 \times 10^{-4} \times 0.0059 = \mathbf{2.95 \times 10^{-6}}$
2. Ratio:
   $$\frac{\eta_{24}}{\eta_1} = \frac{1}{(0.80)^{23}} \approx \mathbf{169.5\times}$$

---

## Solution to Exercise 2
1. During warmup ($s = 500 \le 1,000$):
   $$\eta(500) = 3 \times 10^{-4} \times \frac{500}{1000} = \mathbf{1.5 \times 10^{-4}}$$
2. During cosine decay ($s = 5,500$):
   $$\theta = \pi \frac{5500 - 1000}{10000 - 1000} = \pi \frac{4500}{9000} = \frac{\pi}{2}$$
   $$\cos(\pi / 2) = 0$$
   $$\eta(5500) = 10^{-6} + \frac{1}{2}(3 \times 10^{-4} - 10^{-6})(1 + 0) \approx 10^{-6} + 1.5 \times 10^{-4} = \mathbf{1.51 \times 10^{-4}}$$
