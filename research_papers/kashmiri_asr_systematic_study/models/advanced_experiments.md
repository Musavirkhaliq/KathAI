# Advanced Acoustic Experiments and Negative Results

This document audits the advanced experimental techniques investigated throughout the research program, highlighting negative results, stabilization challenges, and empirical ceilings.

---

## 1. Summary of Advanced Paradigms

```
┌─────────────────────────────────┬──────────┬───────────────┬───────────────────────────────┐
│ Paradigm Investigated           │ Stage    │ Target Metric │ Empirical Outcome             │
├─────────────────────────────────┼──────────┼───────────────┼───────────────────────────────┤
│ Multi-Condition Augmentation    │ Stage 12 │ Reverb / Opus │ Severe clean regression (+3.7%│
│ Layer-Wise Progressive Unfreeze │ Stage 13 │ Encoder Depth │ Underperformed full freeze    │
│ Intermediate CTC Supervision    │ Stage 13 │ Mid-Layers    │ Neutral / slight regression   │
│ Consistency Regularized CTC     │ Stage 13 │ Invariance    │ Neutral (+0.76% error gap)    │
│ Temporal Weight Averaging (WA)  │ Stage 14 │ Stoch. Jitter │ Minor smoothing (-0.05% WER)  │
│ Scaled MWER Sequence Training   │ Stage 15 │ Edit Distance │ Statistically identical to base│
│ Model Parameter Merging (SLERP) │ Stage 16 │ Checkpoints   │ Matched parent ceiling        │
│ Pseudo-Labeling Noisy Audio     │ Stage 17 │ Semi-Superv.  │ Severe acoustic drift (+0.78%)│
│ Targeted Phonetic Confusion Loss│ Stage 19 │ Vowel Pairs   │ Step 0 selected; no gain      │
│ Representation Repair (CARE)    │ Stage 20 │ Adapter Head  │ Serialization failure / bug   │
└─────────────────────────────────┴──────────┴───────────────┴───────────────────────────────┘
```

---

## 2. In-Depth Failure Analyses

### 2.1 Multi-Condition Data Augmentation (Stage 12, R-02)
- **Hypothesis**: Injecting room impulse responses (RIR), additive non-stationary environmental noise (cafeteria, street chatter), and lossy Opus bit-rate compression (16 kbps) during fine-tuning will yield an acoustically bulletproof model.
- **Empirical Reality**: On clean and near-field benchmark speech, KenLM WER degraded from **25.24%** (B-02C) to **28.94%** (R-02), an absolute penalty of **+3.70% WER (+754 word errors)**.
- **Root Cause**: Low-resource acoustic representations rely on subtle spectral-temporal transitions (e.g., palatalized off-glides [ʲ] and central vowel formants $F_1/F_2$). Synthetic noise and psychoacoustic Opus masking obliterate these high-frequency cues, forcing the acoustic encoder to collapse subtle phonetic distinctions into generic vowel states.

### 2.2 Layer-Wise Unfreezing & Intermediate CTC (Stage 13, U-01..03, I-01)
- **Hypothesis**: Freezing early transformer layers while fine-tuning only the top 4, 8, or 12 layers, or adding intermediate CTC supervision at layer 12, will accelerate training and prevent catastrophic forgetting of multilingual priors.
- **Empirical Reality**:
  - `U-01` (top 4): 25.19% WER (4,244 errors)
  - `U-02` (top 8): 24.91% WER (4,196 errors)
  - `U-03` (top 12): 25.09% WER (4,226 errors)
  - `I-01` (Inter-CTC): 25.02% WER (4,215 errors)
  - Foundation B-02C: **24.47% WER (4,122 errors)**
- **Root Cause**: When training data spans over 165 hours, updating all 24 layers end-to-end with low learning rate ($3 \times 10^{-4}$) allows deep cross-attention layers to align with the Kashmiri phonological space. Restricting updates to top layers creates representational bottlenecks.

### 2.3 Pseudo-Labeling on Uncurated Audio (Stage 17, P17-PL)
- **Hypothesis**: Generating greedy pseudo-labels on 31,617 uncurated Tier B audio clips using B-02C and retraining on the combined pool will yield semi-supervised gains.
- **Empirical Reality**: KenLM WER regressed to **25.25%** (+131 word errors over B-02C).
- **Root Cause**: Confirmation bias. When the acoustic model generated incorrect pseudo-labels on ambiguous audio, subsequent fine-tuning reinforced acoustic hallucinations, particularly on vowel diacritics and rare vocabulary.
