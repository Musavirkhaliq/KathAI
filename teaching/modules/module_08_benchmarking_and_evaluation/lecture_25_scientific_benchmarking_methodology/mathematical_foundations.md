# Mathematical Foundations: Domain-Stratified Composite Metrics

---

## 1. Multi-Domain Micro-Averaged WER Formulation

Let there be $D = 8$ evaluation domains.
Each domain $d$ contains $U_d$ utterances with reference words $N_d$ and total edit operations $E_d = S_d + D_d + I_d$.

The **Domain-Specific WER** is:

$$\text{WER}_d = \frac{E_d}{N_d}$$

The **Composite Multi-Domain Micro-Averaged WER** is:

$$\text{WER}_{\text{composite}} = \frac{\sum_{d=1}^D E_d}{\sum_{d=1}^D N_d} = \sum_{d=1}^D \left( \frac{N_d}{\sum_{k=1}^D N_k} \right) \text{WER}_d$$

Notice that composite micro-averaged WER is a **strictly weighted convex combination** of domain WERs, weighted proportional to domain word volume.
