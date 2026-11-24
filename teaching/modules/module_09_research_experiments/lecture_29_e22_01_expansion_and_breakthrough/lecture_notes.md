# Lecture 29: The Breakthrough: E22-01 Scaling, 106k Utterances, and SOTA

---

## 1. The Stage 22 Expansion Milestone

In Stage 22 of the EduCasheer research project, all foundational engineering streams converged:
1. The resolution of the 22.05 kHz Bolbosh defect.
2. The consolidation of 12 multi-source speech archives into a verified 174.55-hour training corpus.
3. The standardization of Certified Normalizer v3.
4. The deployment of Layer-wise Learning Rate Decay (LLRD) and mixed-precision gradient accumulation.

This unified effort culminated in the training of **`E22-01`**, the certified state-of-the-art Kashmiri ASR checkpoint.

---

## 2. Quantitative Training Profile of E22-01

- **Dataset Scale**: **106,487 audio utterances** (174.55 hours of curated clean speech).
- **Acoustic Backbone**: XLS-R 300M (24 Transformer layers, 16 attention heads, 1024 hidden dimension).
- **Output Vocabulary**: 47 certified Kashmiri graphemes (`configs/vocab.json`).
- **Optimization Parameters**:
  - Base Learning Rate: $\eta_{\text{base}} = 3 \times 10^{-4}$ with LLRD decay $\xi = 0.75$.
  - Effective Batch Size: 64 utterances (8 physical $\times$ 8 accumulation steps).
  - Mixed Precision: PyTorch AMP FP16 with dynamic `GradScaler`.
  - Gradient Clipping: $\|\mathbf{g}\| \le 1.0$.
  - Epochs: 20 epochs ($\approx 33,280$ optimizer steps).

---

## 3. Dissecting the Conflicting WER Reports in Stage 23

A critical finding uncovered during our forensic audit was that historical project reports cited four conflicting benchmark figures for model `E22-01`:
1. **24.74% WER**
2. **24.40% WER**
3. **24.12% WER**
4. **24.94% WER**

### The Forensic Resolution:
Our Stage 27 investigation proved that these numbers were not training discrepancies or bugs; they represented **four distinct decoder configurations evaluated on the exact same checkpoint**:

| Reported WER | Acoustic Temp ($T$) | LM Weight ($\alpha$) | Insertion Penalty ($\beta$) | Beam Width ($W$) | Notes |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **24.94%** | $1.00$ (Uncalibrated) | $0.30$ | $1.00$ | 64 | Uncalibrated temperature baseline |
| **24.74%** | $1.08$ | $0.30$ | $1.00$ | 64 | Configuration A (Standard balanced decoding) |
| **24.40%** | $1.08$ | $0.40$ | $1.20$ | 80 | Intermediate tuning run |
| **24.12%** | **$1.08$** | **$0.50$** | **$1.50$** | **100** | **Certified Configuration B (Official SOTA)** |

The official certified state-of-the-art result of **24.12% WER** is achieved with calibrated temperature $T = 1.08$, aggressive language model weighting $\alpha = 0.50$, word insertion bonus $\beta = 1.50$, and beam width $W = 100$.
