# Solutions: Lecture 21 (Greedy vs. Beam Search)

---

## Solution to Exercise 1
Total prior probability: $P_{\text{total}} = P_b + P_{nb} = 0.30 + 0.20 = 0.50$.
1. For prefix `"ک"`:
   - Blank extension: `P_b("ک") = P_{\text{total}} \times y_ε = 0.50 \times 0.40 = \mathbf{0.20}`.
   - Non-blank identical character extension:
> P_{nb}(**"ک"**) = P_{nb}(**"ک"**) \times y_{**"ک"**} = 0.20 \times 0.30 = \mathbf{0.06}
2. For new prefix `"کک"` (identical character following blank):
> P_{nb}(**"کک"**) = P_b(**"ک"**) \times y_{**"ک"**} = 0.30 \times 0.30 = \mathbf{0.09}
3. For new prefix `"ک"ٲ""` (distinct character):
> P_{nb}(**"کٲ"**) = P_{\text{total}} \times y_{**"ٲ"**} = 0.50 \times 0.20 = \mathbf{0.10}

---

## Solution to Exercise 2
- Going from $W=1$ to $W=10$ drops WER by **3.4% absolute**.
- Going from $W=10$ to $W=64$ drops WER by **1.6% absolute**.
- Going from $W=64$ to $W=200$ drops WER by only **0.1% absolute**, while increasing CPU decoding time by over $300\%$.
- **Optimal Sweet Spot**: **$W = 64$ to $100$**, capturing 99.6% of maximum accuracy within strict real-time constraints.
