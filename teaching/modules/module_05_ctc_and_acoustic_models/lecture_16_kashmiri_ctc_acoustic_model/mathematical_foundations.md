# Mathematical Foundations: Parameter Counting and Capacity Allocation

---

## 1. Parameter Breakdown of EduCasheer Kashmiri Acoustic Model

### A. 7-Layer Convolutional Feature Extractor:
- Total Conv parameters:
  $$P_{\text{conv}} = \sum_{i=1}^7 K_i \cdot C_{i-1} \cdot C_i \approx 4.1 \times 10^6 \text{ parameters}$$

### B. 24-Layer Transformer Encoder:
For each layer with $d_{\text{model}} = 1024, d_{\text{ff}} = 4096$:
- Multi-Head Attention: $4 \times d_{\text{model}}^2 = 4 \times 1024^2 \approx 4.19 \times 10^6$
- Feedforward Network: $2 \times d_{\text{model}} \times d_{\text{ff}} = 2 \times 1024 \times 4096 \approx 8.39 \times 10^6$
- LayerNorms: $4 \times 1024 \approx 4,096$
- Per-layer total: $\approx 12.59 \times 10^6$
- 24 layers total: $24 \times 12.59 \times 10^6 \approx 302.2 \times 10^6 \text{ parameters}$

### C. CTC Projection Head:
$$P_{\text{head}} = (1024 \times 47) + 47 = 48,175 \text{ parameters}$$

**Grand Total**: $\approx \mathbf{317 \times 10^6 \text{ parameters}}$ (317M).
