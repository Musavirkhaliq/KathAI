# Quiz: Lecture 31 (Neural Rescoring and Oracle Analysis)

---

### Question 1:
What did the Stage 24 Oracle WER analysis reveal for the EduCasheer 100-best beam candidate list?
- [A] Oracle WER was 24.12%, showing no potential gain.
- [B] Oracle WER was 13.40%, revealing a massive 10.72% potential accuracy headroom achievable with advanced language models.
- [C] Oracle WER was 0.00%.
- [D] Beam search produced zero valid candidates.

**Correct Answer**: **[B]**  
*Explanation*: The 13.40% oracle bound demonstrated that correct words were present in the beam.

---

### Question 2:
What is the primary architectural limitation of n-gram language models compared to Neural LMs?
- [A] $n$-grams only work on Windows.
- [B] $n$-grams have a strict $n-1$ word context window and treat words as discrete tokens without semantic parameter sharing.
- [C] $n$-grams are too slow for real-time decoding.
- [D] $n$-grams cannot process Kashmiri.

**Correct Answer**: **[B]**  
*Explanation*: Discrete Markov $n$-grams cannot model long-range discourse semantics.

---

### Question 3:
What is Oracle Word Error Rate?
- [A] The WER achieved by a human transcriber.
- [B] The lowest possible WER obtained by picking the best candidate hypothesis from the N-best list for every utterance against the reference.
- [C] The WER predicted by a linear regression formula.
- [D] The WER of greedy search.

**Correct Answer**: **[B]**  
*Explanation*: Oracle WER quantifies the upper bound of accuracy present within the candidate beam.

---

### Question 4:
In a two-pass decoding architecture, what is the role of the first-pass decoder?
- [A] To compute the final transcription.
- [B] To rapidly prune the exponential search space and generate a diverse N-best candidate list in real-time using a lightweight n-gram model.
- [C] To resample audio from 16 kHz to 48 kHz.
- [D] To normalize Unicode.

**Correct Answer**: **[B]**  
*Explanation*: First-pass decoding provides fast candidate generation.

---

### Question 5:
Why is N-best rescoring computationally practical for real-time deployment?
- [A] Because the expensive neural language model evaluates only $N$ candidate sentences (e.g., $N=100$), rather than evaluating every possible frame transition in the lattice.
- [B] Because it deletes the audio.
- [C] Because it runs without GPUs.
- [D] Because it uses integer arithmetic.

**Correct Answer**: **[A]**  
*Explanation*: Evaluating $N=100$ string sequences with a neural model takes only a few milliseconds.
