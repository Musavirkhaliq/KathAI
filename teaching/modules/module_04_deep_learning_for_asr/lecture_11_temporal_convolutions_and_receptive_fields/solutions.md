# Solutions: Lecture 11 (Temporal Convolutions and Receptive Fields)

---

## Solution to Exercise 1
1. Cumulative downsampling factor $D$:
   $$D = \prod_{i=1}^5 S_i = 4 \times 2 \times 2 \times 2 \times 1 = \mathbf{32}$$
2. Receptive Field $R_l = R_{l-1} + (K_l - 1) J_{l-1}$:
   - $R_1 = 1 + (7 - 1) \times 1 = 7$
   - $R_2 = 7 + (5 - 1) \times 4 = 7 + 16 = 23$
   - $R_3 = 23 + (5 - 1) \times 8 = 23 + 32 = 55$
   - $R_4 = 55 + (3 - 1) \times 16 = 55 + 32 = 87$
   - $R_5 = 87 + (3 - 1) \times 32 = 87 + 64 = \mathbf{151 \text{ samples}}$
3. In milliseconds at 16 kHz:
   - Frame shift: $\frac{32}{16000} \text{ s} = \mathbf{2.0 \text{ ms}}$
   - Receptive field: $\frac{151}{16000} \text{ s} = \mathbf{9.44 \text{ ms}}$

---

## Solution to Exercise 2
1. Using layer-by-layer truncation without padding:
   $$T_\text{samples} = 4.35 \times 16000 = 69,600 \text{ samples}$$
2. Approximate:
   $$T_\text{frames} \approx \left\lfloor \frac{69600 - 400}{320} \right\rfloor + 1 = \left\lfloor \frac{69200}{320} \right\rfloor + 1 = 216 + 1 = \mathbf{217 \text{ frames}}$$
