# N-Best Oracle Headroom and Search Space Analysis

This document provides a systematic analysis of the Oracle Word Error Rate across varying beam widths and candidate pool sizes on `dev_assessment.jsonl`.

---

## 1. Oracle Error Rate Formulation

For an utterance with reference transcript $y^*$ and an extracted set of $N$ decoded hypotheses $\mathcal{H}_N = \{y^{(1)}, y^{(2)}, \dots, y^{(N)}\}$, the Oracle Word Error Rate represents the minimum edit distance achievable if an omniscient reranker always selected the best hypothesis in the beam:

$$\text{WER}_{\text{Oracle}}(N) = \frac{\sum_{i=1}^M \min_{y \in \mathcal{H}_N(i)} \text{EditDistance}(y, y^*_i)}{\sum_{i=1}^M \text{Length}(y^*_i)} \times 100\%$$

---

## 2. Oracle Progression on Canonical 16-Beam Decoding

Using baseline acoustic checkpoint E-03 with KenLM ($\alpha=0.35, \beta=-0.50$) across 1,001 utterances (16,845 reference words):

| Candidate Size ($N$) | Top-1 WER | Oracle WER | Total Errors Remaining | Recoverable Word Errors | Exact Sentence Matches |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **$N = 1$ (Baseline)** | 24.46% | **24.46%** | 4,121 | **0 (0.00%)** | 167 / 1,001 (16.68%) |
| **$N = 4$** | 24.46% | **21.54%** | 3,629 | **492 (2.92%)** | 248 / 1,001 (24.78%) |
| **$N = 8$** | 24.46% | **20.26%** | 3,412 | **709 (4.21%)** | 272 / 1,001 (27.17%) |
| **$N = 16$ (Champion)** | 24.46% | **19.23%** | 3,240 | **881 (5.23%)** | 302 / 1,001 (30.17%) |
| **$N = 32$** | 24.46% | **18.51%** | 3,118 | **1,003 (5.95%)** | 312 / 1,001 (31.17%) |
| **$N = 64$** | 24.46% | **18.32%** | 3,086 | **1,035 (6.14%)** | 315 / 1,001 (31.47%) |

---

## 3. Key Scientific Insights

### 3.1 The Search Ceiling vs Acoustic Ceiling
- **Search / Scoring Failure (Recoverable Headroom)**: Exactly **881 word errors** (a 5.23% absolute WER reduction, or 21.38% of total errors) are already present inside the 16-best candidate beam. These errors represent instances where the acoustic model produced the correct word tokens, but the language model or linear score combination failed to rank the hypothesis at position 1.
- **Acoustic Deficiency / Out-of-Beam**: Exactly **3,240 word errors** (78.62% of total errors) never enter the 16-best beam. These represent genuine acoustic failure modes—background noise, severe dialectal divergence, or missing acoustic mass—that cannot be recovered by any downstream rescorer without acoustic retraining or vocabulary expansion.

### 3.2 Diminishing Returns Beyond $N=16$
Expanding from $N=1$ to $N=16$ recovers 881 errors (+5.23% absolute headroom). Expanding further from $N=16$ to $N=64$ recovers only 154 additional errors (+0.91% headroom) while quadrupling memory footprint and increasing search latency by $3.4\times$. Therefore, **$N=16$ represents the optimal operational frontier** for sequence rescoring.
