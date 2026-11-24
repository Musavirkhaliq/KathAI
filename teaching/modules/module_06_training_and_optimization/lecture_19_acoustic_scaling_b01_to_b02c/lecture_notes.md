# Lecture 19: Empirical Scaling Analysis: From B-01 to B-02C

---

## 1. The Science of Empirical Scaling in Speech Recognition

A central question in computational linguistics and AI systems engineering is:
*How does speech recognition performance scale as a function of curated audio hours?*

In low-resource languages, acquiring additional transcribed speech requires substantial financial and logistical expenditure. Understanding the **empirical scaling curve** allows research leaders to predict error rates, quantify return on investment, and identify points of diminishing returns.

The EduCasheer project systematically tracked acoustic scaling across three canonical model generations: **B-01**, **B-02**, and **B-02C**.

---

## 2. The Historical Scaling Progression

| Model ID | Research Stage | Training Hours | Utterance Count | Clean Studio WER | Disjoint Benchmark WER | WAAZ Reverberant WER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **B-01** | Stage 06 | 20.4 hours | 12,410 | 28.4% | 42.10% | 74.20% |
| **B-02** | Stage 12 | 61.8 hours | 38,150 | 19.8% | 32.80% | 61.40% |
| **B-02C** | Stage 18 | 112.5 hours | 71,200 | 14.2% | **25.82%** | **55.15%** |
| **E22-01** | Stage 22 | 174.5 hours | 106,487 | 11.8% | **24.12%** | **51.80%** |

---

## 3. Power-Law Scaling Formulation

Following Kaplan et al. (2020) and Baevski et al. (2021), Word Error Rate follows a **power-law relationship** with respect to dataset size (in hours $H$):

$$\text{WER}(H) = \text{WER}_\infty + \frac{A}{H^\alpha}$$

where:
- $\text{WER}_\infty$ is the irreducible Bayes error rate (due to acoustic ambiguity, speaker mumbling, and transcription noise).
- $A$ is an empirical scaling coefficient.
- $\alpha$ is the **power-law scaling exponent**.

### Fitting the Kashmiri Scaling Curve:
Fitting empirical metrics from B-01, B-02, and B-02C yields:
- Irreducible error floor: $\text{WER}_\infty \approx 18.5\%$ (on disjoint multi-domain speech without neural rescoring).
- Scaling exponent: $\alpha \approx 0.42$.

### Key Empirical Findings:
1. **Tripling data from 20h to 60h** reduced disjoint WER by **9.3% absolute** (42.1% to 32.8%).
2. **Doubling data from 60h to 112h** reduced disjoint WER by **7.0% absolute** (32.8% to 25.82%).
3. **Diminishing Returns**: Expanding from 112h to 174h (E22-01) yielded a 1.70% improvement on disjoint test sets, proving that beyond 120 hours, **acoustic domain diversity and language model quality** contribute more than brute-force volume scaling.
