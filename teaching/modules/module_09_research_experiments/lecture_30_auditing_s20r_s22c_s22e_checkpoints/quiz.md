# Quiz: Lecture 30 (Checkpoint Forensics)

---

### Question 1:
Which historical checkpoint was certified as the genuine `E22-01` production model?
- [A] `S20R`
- [B] `S22C`
- [C] `S22E`
- [D] `Kaldi_v1`

**Correct Answer**: **[C]**  
*Explanation*: `S22E` was certified as the genuine E22-01 checkpoint that achieved 24.12% disjoint WER.

---

### Question 2:
Why is tensor-level hashing preferred over file-level hashing when establishing reproducibility?
- [A] Tensor hashing is faster.
- [B] File-level hashing is vulnerable to non-deterministic zip container metadata and timestamps, whereas tensor hashing evaluates raw numerical array bytes.
- [C] Tensor hashing converts weights to 8-bit.
- [D] File hashing only works on Windows.

**Correct Answer**: **[B]**  
*Explanation*: Tensor hashing operates directly on array bytes, eliminating serialization container noise.

---

### Question 3:
What structural feature in the linear classification head distinguishes E22-01 (`S22E`) from the earlier `S20R` checkpoint?
- [A] E22-01 has 1,000 layers.
- [B] E22-01 has a positive blank bias ($b_0 = +3.0$) and 47 output classes, whereas S20R had zero bias ($b_0 = 0.0$).
- [C] S20R had no Transformer layers.
- [D] E22-01 runs only on TPU.

**Correct Answer**: **[B]**  
*Explanation*: The positive blank bias distinguishes Stage 22 initialization from Stage 20.

---

### Question 4:
What was the role of the `S20R` checkpoint in the repository history?
- [A] It was a corrupted file that crashed Python.
- [B] It was the baseline recovery checkpoint from Stage 20, preserving the B-02C model weights after the MWER experiment was rolled back.
- [C] It was an English audio file.
- [D] It was an automated test script.

**Correct Answer**: **[B]**  
*Explanation*: S20R restored the clean B-02C model baseline following the Stage 19 MWER rollback.

---

### Question 5:
What action must an engineer take before deploying an ASR checkpoint in production?
- [A] Change the file extension to `.txt`.
- [B] Verify the cryptographic SHA-256 hash of the weights and run deterministic logit parity against reference test audio.
- [C] Retrain the model for 1 step.
- [D] Delete the vocabulary file.

**Correct Answer**: **[B]**  
*Explanation*: Cryptographic hash verification and logit parity testing guarantee deployment fidelity.
