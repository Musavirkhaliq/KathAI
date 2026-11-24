# Quiz: Lecture 29 (E22-01 Scaling and Breakthrough)

---

### Question 1:
What is the certified state-of-the-art Word Error Rate achieved by model `E22-01` on the disjoint Kashmiri benchmark?
- [A] 42.10%
- [B] 25.82%
- [C] 24.12%
- [D] 11.80%

**Correct Answer**: **[C]**  
*Explanation*: 24.12% is the official certified result using Configuration B.

---

### Question 2:
How many training utterances were curated for the Stage 22 expansion?
- [A] 12,410
- [B] 71,200
- [C] 106,487
- [D] 1,000,000

**Correct Answer**: **[C]**  
*Explanation*: The final training corpus contained 106,487 audio segments (174.55 hours).

---

### Question 3:
What caused the four conflicting WER reports (24.74%, 24.40%, 24.12%, 24.94%) in historical documents?
- [A] Different training checkpoints.
- [B] Four distinct decoder parameter configurations evaluated on the exact same model checkpoint.
- [C] Transcription spelling changes.
- [D] Audio resampling errors.

**Correct Answer**: **[B]**  
*Explanation*: Forensic reconciliation proved that all four numbers came from one checkpoint evaluated under different decoding parameters.

---

### Question 4:
What decoder configuration produced the certified 24.12% WER?
- [A] $T=1.00, \alpha=0.10, \beta=0.50$
- [B] $T=1.08, \alpha=0.50, \beta=1.50, \text{beam}=100$
- [C] Greedy decoding
- [D] Whisper beam search

**Correct Answer**: **[B]**  
*Explanation*: Configuration B with $T=1.08, \alpha=0.50, \beta=1.50$ is the certified standard.

---

### Question 5:
How many Transformer encoder layers are in the E22-01 architecture?
- [A] 12
- [B] 24
- [C] 48
- [D] 96

**Correct Answer**: **[B]**  
*Explanation*: E22-01 is based on the 24-layer XLS-R 300M architecture.
