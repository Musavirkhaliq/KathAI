# Quiz: Lecture 19 (Empirical Scaling Analysis)

---

### Question 1:
What disjoint benchmark WER was achieved by the B-02C model trained on 112.5 hours?
- [A] 42.10%
- [B] 32.80%
- [C] 25.82%
- [D] 11.80%

**Correct Answer**: **[C]**  
*Explanation*: B-02C achieved 25.82% disjoint WER, a massive milestone in Stage 18.

---

### Question 2:
What mathematical relationship best describes ASR error rate reduction as training hours increase?
- [A] Linear decay
- [B] Power-law decay ($\text{WER}_\infty + A H^{-\alpha}$)
- [C] Exponential increase
- [D] Constant flat line

**Correct Answer**: **[B]**  
*Explanation*: Neural scaling laws across speech and language universally follow power-law distributions.

---

### Question 3:
What was the primary finding when scaling from 112.5 hours (B-02C) to 174.5 hours (E22-01)?
- [A] Performance degraded due to overfitting.
- [B] Disjoint WER improved by 1.70% (25.82% to 24.12%), demonstrating diminishing returns from acoustic volume alone.
- [C] GPU memory doubled.
- [D] CTC loss became zero.

**Correct Answer**: **[B]**  
*Explanation*: Acoustic scaling showed clear diminishing returns, highlighting the need for LM fusion.

---

### Question 4:
Why did reverberant WAAZ sermons exhibit higher WER than clean studio speech across all model generations?
- [A] WAAZ was recorded in English.
- [B] Open-air reverberation, spontaneous speech elision, and crowd noise distort spectral features and smear phoneme boundaries.
- [C] The sample rate was too high.
- [D] WAAZ was transcribed incorrectly.

**Correct Answer**: **[B]**  
*Explanation*: Acoustic reverberation and spontaneous speech pose severe non-stationary challenges.

---

### Question 5:
What does the irreducible error floor $\text{WER}_\infty$ represent?
- [A] A software bug in PyTorch.
- [B] The asymptotic lower bound on error caused by inherent acoustic ambiguity, homophones, and human transcription disagreement.
- [C] The number of layers in the model.
- [D] The size of the vocabulary.

**Correct Answer**: **[B]**  
*Explanation*: Irreducible Bayes error represents the limit of performance under ambiguous acoustics.
