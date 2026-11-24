# Quiz: Lecture 27 (Research Archaeology: Stages 01 to 09)

---

### Question 1:
What Word Error Rate was produced by the Stage 01 Urdu Kaldi baseline on Kashmiri speech?
- [A] 12.4%
- [B] 42.1%
- [C] 88.40%
- [D] 9.8%

**Correct Answer**: **[C]**  
*Explanation*: The Urdu Kaldi model failed catastrophically at 88.40% WER due to phonological and lexical mismatch.

---

### Question 2:
Why was OpenAI Whisper abandoned in favor of XLS-R CTC in Stage 04?
- [A] Whisper cannot run on Linux.
- [B] Whisper suffered from severe autoregressive hallucination loops and dropped unstressed Kashmiri central vowels in low-resource regimes.
- [C] Whisper requires Python 2.
- [D] Whisper audio files must be 8 kHz.

**Correct Answer**: **[B]**  
*Explanation*: Attention Encoder-Decoder models hallucinate under extreme low-resource conditions; CTC guarantees acoustic monotonicity.

---

### Question 3:
What was the primary defect introduced in the Stage 03 Bolbosh preprocessing?
- [A] Audio was deleted.
- [B] Defective Fourier resampling wiped out high-frequency acoustic energy between 5.8 kHz and 8.0 kHz.
- [C] Files were converted to MP3.
- [D] Audio was slowed down by 50%.

**Correct Answer**: **[B]**  
*Explanation*: Over-filtering eliminated critical high-frequency fricative formants.

---

### Question 4:
What benchmark WER was achieved by the B-01 baseline model in Stage 06?
- [A] 88.4%
- [B] 42.10%
- [C] 24.12%
- [D] 15.0%

**Correct Answer**: **[B]**  
*Explanation*: B-01 trained on 20.4 hours achieved 42.10% disjoint WER.

---

### Question 5:
In scientific auditing, what constitutes physical evidence of an experiment's execution?
- [A] A verbal conversation between engineers.
- [B] Git commit history, executable source code, execution log timestamps, and verified checkpoint SHA-256 hashes.
- [C] A PowerPoint presentation.
- [D] A todo list.

**Correct Answer**: **[B]**  
*Explanation*: Physical code, logs, and cryptographic hashes are the sole basis of reproducible verification.
