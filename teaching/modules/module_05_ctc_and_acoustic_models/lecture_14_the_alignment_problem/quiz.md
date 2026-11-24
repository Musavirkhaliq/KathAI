# Quiz: Lecture 14 (The Alignment Problem)

---

### Question 1:
What is the primary cause of the alignment problem in speech recognition?
- [A] Audio files have stereo channels.
- [B] Acoustic feature sequences are continuous and much longer ($T$) than discrete target text sequences ($U$), with no explicit frame-level timestamp annotations.
- [C] Audio cannot be loaded into GPU memory.
- [D] Speech has non-zero noise.

**Correct Answer**: **[B]**  
*Explanation*: The variable length ratio ($T \gg U$) without frame timestamps creates the alignment problem.

---

### Question 2:
How does CTC loss resolve the alignment problem?
- [A] By hiring human annotators to label every frame.
- [B] By marginalizing (summing) the probability over all permissible valid alignments that collapse to the target sequence.
- [C] By truncating audio to match text length.
- [D] By padding text with random characters.

**Correct Answer**: **[B]**  
*Explanation*: CTC integrates over all possible alignments in $\mathcal{B}^{-1}(\mathbf{Y})$.

---

### Question 3:
What is the time complexity of Dynamic Time Warping for sequences of lengths $N$ and $M$?
- [A] $\mathcal{O}(N + M)$
- [B] $\mathcal{O}(NM)$
- [C] $\mathcal{O}(N^2 M^2)$
- [D] $\mathcal{O}(\log(NM))$

**Correct Answer**: **[B]**  
*Explanation*: DTW evaluates an $N \times M$ dynamic programming cost matrix.

---

### Question 4:
Why did HMM forced alignment discard up to 40% of Kashmiri audio in early repository experiments?
- [A] Hard drive failures.
- [B] Pronunciation dictionary mismatches and non-standardized orthography made phonetic graph matching fail.
- [C] Kashmiri speakers spoke too quietly.
- [D] Kaldi was incompatible with Linux.

**Correct Answer**: **[B]**  
*Explanation*: Strict phonetic lexicons fail in low-resource languages with unstandardized spelling.

---

### Question 5:
What monotonic constraint is enforced by both DTW and CTC?
- [A] Time cannot flow backward; alignments must advance or stay at the same symbol.
- [B] Pitch must strictly increase.
- [C] Volume must decrease.
- [D] Sequence length must be even.

**Correct Answer**: **[A]**  
*Explanation*: Monotonicity ensures that the temporal order of spoken words is strictly preserved.
