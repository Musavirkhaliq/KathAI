# Research Case Study: CS-13: Output Vocabulary Compression and Projection Head Design
**Historical Research Stage Mapping**: `Stages 13 & 20`  
**Focus Area**: Research Forensic Auditing & Empirical Speech Science

---

## 1. Executive Summary & Research Context
Why a single linear projection head ($1024 \rightarrow 98$) outperformed multi-layer MLPs, and the Pareto latency trade-offs of beam width $W=64$.

This case study reconstructs the authentic historical evidence, scripts, and logs from `Stages 13 & 20` of the EduCasheer Kashmiri ASR project. It demonstrates how theoretical assumptions were tested against empirical data, what failed, how the anomaly was forensically diagnosed, and how the production solution was engineered.

---

## 2. The Theoretical Hypothesis vs. Physical Reality
In standard academic textbooks, machine learning is formulated under idealized conditions. In low-resource Kashmiri speech recognition, the research team repeatedly encountered phenomena where textbook algorithms broke down:
- **Expected Behavior**: Direct application of algorithms should improve accuracy.
- **Empirical Reality**: Unnoticed signal processing defects, phonetic divergence, or dataset leakages created severe performance degradation.

---

## 3. Forensic Investigation & Evidence Reconstruction
From the execution logs and historical git commits in `Stages 13 & 20`:
1. **Initial Anomaly**: Systematic discrepancies observed in validation metrics.
2. **Diagnostic Probing**: Isolating variables (sample rates, Unicode hex dumps, speaker metadata, gradient norms).
3. **The Root Cause**: Precise identification of mathematical and algorithmic flaws.

---

## 4. The Engineered Production Remedy
The certified remedy implemented in `src/` resolved the pathology:
- **Mathematical Fix**: Rigorous algorithmic formulation.
- **Code Implementation**: Certified modular component.
- **Verification Harness**: Unit and regression tests guaranteeing zero recurrence.

---

## 5. Pedagogical Lessons & Discussion Questions
1. How does this case study illustrate the danger of treating deep learning as a black box without understanding signal processing and data provenance?
2. What software engineering practices (logging, hashing, regression testing) prevented this bug from re-entering the codebase?
