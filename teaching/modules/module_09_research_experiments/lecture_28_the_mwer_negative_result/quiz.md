# Quiz: Lecture 28 (The MWER Negative Result)

---

### Question 1:
What does the Minimum Word Error Rate (MWER) loss objective minimize?
- [A] The frame-level cross-entropy of phonemes.
- [B] The expected Levenshtein word edit distance over the model's N-best beam search hypotheses.
- [C] The duration of audio files.
- [D] The size of the vocabulary.

**Correct Answer**: **[B]**  
*Explanation*: MWER computes the expected value of edit distance across candidate hypotheses.

---

### Question 2:
What happened when MWER was evaluated on the 8-domain benchmark in Stage 19?
- [A] It set a new world record on all domains.
- [B] While clean studio speech improved slightly (14.2% to 12.1%), spontaneous WAAZ sermons degraded catastrophically (55.1% to 68.4%), and overall multi-domain WER worsened.
- [C] The model crashed due to GPU driver failure.
- [D] The vocabulary size exploded to 10,000 tokens.

**Correct Answer**: **[B]**  
*Explanation*: MWER caused severe domain collapse and degraded out-of-distribution performance.

---

### Question 3:
What mathematical vulnerability causes high gradient variance in MWER training?
- [A] High learning rates.
- [B] Sampling small N-best lists in low-resource regimes where candidate edit distances fluctuate wildly.
- [C] Sinc resampling.
- [D] 16 kHz sampling.

**Correct Answer**: **[B]**  
*Explanation*: Small beam candidate lists yield noisy, high-variance Monte Carlo gradient estimates.

---

### Question 4:
What research decision was made in Stage 20 regarding MWER?
- [A] To retrain it for 100 epochs.
- [B] To officially abandon MWER and return to standard CTC loss with LLRD and temperature calibration.
- [C] To publish the clean result and hide the WAAZ result.
- [D] To rewrite PyTorch in C.

**Correct Answer**: **[B]**  
*Explanation*: The project abandoned MWER to preserve robust cross-domain generalization.

---

### Question 5:
Why are negative research results valuable in machine learning science?
- [A] They allow researchers to delete code.
- [B] They prevent future researchers from wasting resources on flawed approaches and reveal critical theoretical limitations of algorithms.
- [C] They reduce paper review time.
- [D] They reduce training costs to zero.

**Correct Answer**: **[B]**  
*Explanation*: Negative results document empirical boundaries and prevent circular research dead-ends.
