# Stage 29 Formal Research Plan & Pre-Registered Protocol

**Project**: EduCasheer Kashmiri Automatic Speech Recognition Research  
**Experiment Series**: Stage 29 — Meta Omnilingual ASR CTC Evaluation & Adaptation  
**Primary Candidate**: `omniASR_CTC_1B_v2`  
**Secondary Control**: `omniASR_CTC_300M_v2`  
**Certified Production Baseline**: EduCasheer `E22-01`  
**Author**: Principal ASR Research Scientist & Deep Learning Architect  
**Date**: November 17, 2026  

---

## 1. Formal Hypotheses & Experimental Design

| Experiment ID | Model Name | Architecture | Training Corpus | Purpose | Success Metric |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **O29-00A** | Omnilingual 300M v2 Zero-Shot | Wav2Vec2 300M + 10,288 CTC | Pretrained weights | Zero-shot baseline for 300M scale | Character & Word Error Rate |
| **O29-00B** | Omnilingual 1B v2 Zero-Shot | Wav2Vec2 1B + 10,288 CTC | Pretrained weights | Zero-shot baseline for 1B scale | Baseline WER before Kashmiri tuning |
| **O29-01** | Omnilingual 300M v2 Fine-Tuned | Wav2Vec2 300M + 10,288 CTC | Dataset A (166.39h) | Controlled 300M comparison vs E22-01 | Greedy Dev WER $< 26.60\%$ |
| **O29-02** | Omnilingual 1B v2 Fine-Tuned | Wav2Vec2 1B + 10,288 CTC | Dataset A (166.39h) | Primary candidate: 1B capacity scaling | Greedy Dev WER $< 24.74\%$ |
| **O29-03** | Best Omnilingual + Expanded Data | Best Architecture | Dataset B (207.24h) | Data scaling test with recovered speech | Statistically significant $\Delta\text{WER} < 0$ |
| **O29-04** | Conversational Domain Adaptation | Best Architecture | Verified WAAZ + Balanced | Spontaneous speech domain robustness | WAAZ Dev WER $< 46.12\%$ |

---

## 2. Evaluation Protocol & Statistical Rules

### 2.1 Frozen Normalization Standard
All evaluations must apply universal Kashmiri text normalization (`normalize_for_wer_evaluation`):
- Unicode NFC normalization.
- Perso-Arabic variant unification (Keheh `ک`, Heh Goal `ہ`, Kashmiri Yeh `ی`).
- Stripping of diacritics (`[\u064b-\u065f\u0670\u06d6-\u06ed]`).
- Numeral standardization (Eastern Arabic to ASCII).
- Punctuation removal and whitespace collapse.
- Reference transcripts are frozen and never modified to suit a model's preferences.

### 2.2 Benchmarking Protocols
- **Protocol A (Greedy Acoustic Output)**:
  - Raw argmax CTC output collapsed via standard CTC collapse rule ($y_t \ne y_{t-1}$, ignore blank index 0).
  - Evaluates acoustic modeling capability independent of external language models.
- **Protocol B (Calibrated Shallow Fusion / KenLM)**:
  - Evaluates models coupled with the domain-adapted 4-gram KenLM language model (`lm_word_4gram_domain_d.arpa`).
  - Decoding hyperparameters ($\alpha, \beta, \text{beam}$) calibrated strictly on `dev_calibration` and frozen before held-out test evaluation.
- **Protocol C (Sealed Independent Test)**:
  - Final evaluation executed on `test_manifest.jsonl` (5,923 utterances, 120,816 words).
  - Paired 10,000-sample bootstrap resampling and Wilcoxon signed-rank significance tests.

---

## 3. Preflight & Execution Gates

1. **Gate 1: Tokenizer Verification** (PASSED): Verified $>99.85\%$ native token coverage across canonical Kashmiri corpora.
2. **Gate 2: Hardware Feasibility** (PASSED): Unified memory profiling confirms GB10 accommodates 1B model training with $>66\text{ GB}$ safe buffer.
3. **Gate 3: One-Batch Overfitting Test**: Train on a 4-sample verified mini-batch to confirm CTC loss decreases to near-zero ($< 0.10$).
4. **Gate 4: Short Pilot Run**: 500-step pilot on Dataset A to verify stability, gradient norms, absence of NaNs, and successful checkpoint serialization.
5. **Gate 5: Full Fine-Tuning Execution**: Launch and monitor primary run to convergence, validating periodically on `dev_a_disjoint` and `dev_b_waaz_val`.
