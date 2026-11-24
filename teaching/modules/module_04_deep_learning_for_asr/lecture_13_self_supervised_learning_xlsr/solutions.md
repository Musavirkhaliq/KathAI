# Solutions: Lecture 13 (Self-Supervised Learning and XLS-R)

---

## Solution to Exercise 1
1. With all similarities equal to 0:
   $$\mathcal{L}_m = -\log \left( \frac{e^0}{e^0 + \sum_{k=1}^{100} e^0} \right) = -\log \left( \frac{1}{1 + 100} \right) = \log(101) \approx \mathbf{4.615}$$
2. With positive similarity 0.85 and 100 negatives at 0.10:
   $$z_{\text{pos}} = \frac{0.85}{0.1} = 8.5 \implies e^{8.5} \approx 4914.77$$
   $$z_{\text{neg}} = \frac{0.10}{0.1} = 1.0 \implies 100 \times e^{1.0} \approx 100 \times 2.718 = 271.83$$
   $$\mathcal{L}_m = -\log \left( \frac{4914.77}{4914.77 + 271.83} \right) = -\log \left( \frac{4914.77}{5186.6} \right) = -\log(0.9476) \approx \mathbf{0.0538}$$

---

## Solution to Exercise 2
1. Maximum entropy occurs under a uniform distribution:
   $$H_{\max} = \log(V) = \log(320) \approx \mathbf{5.768 \text{ nats}}$$
2. Under complete collapse:
   $$H = -1 \log(1) - 0 = 0$$
   $$\mathcal{L}_d = \log(V) - 0 = \mathbf{5.768}$$
   The diversity loss applies maximum penalty when codebook entries collapse.
