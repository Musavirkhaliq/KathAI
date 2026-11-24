# Quiz: Lecture 21 (Greedy vs. Beam Search)

---

### Question 1:
What is the 'greedy fallacy' in CTC decoding?
- [A] The model uses too much RAM.
- [B] The sequence of locally argmax frame emissions does not necessarily equal the sequence with the highest total marginalized probability mass across all collapsing paths.
- [C] The model emits words in reverse order.
- [D] Greedy search is non-deterministic.

**Correct Answer**: **[B]**  
*Explanation*: Multi-path probability accumulation often favors candidate sequences whose individual paths are not the argmax.

---

### Question 2:
Why does CTC prefix beam search track two separate probabilities ($P_b$ and $P_{nb}$) for each prefix?
- [A] To compute standard deviation.
- [B] To correctly differentiate between sustaining a character vs. emitting a consecutive identical character separated by a blank.
- [C] For mixed precision compatibility.
- [D] To store audio sampling rate.

**Correct Answer**: **[B]**  
*Explanation*: Tracking $P_b$ and $P_{nb}$ prevents identical adjacent characters from erroneously collapsing.

---

### Question 3:
What is the standard beam width ($W$) used in EduCasheer production pipelines?
- [A] 1
- [B] 4
- [C] 64 to 100
- [D] 10,000

**Correct Answer**: **[C]**  
*Explanation*: $W=64-100$ provides optimal balance between error rate reduction and decoding throughput.

---

### Question 4:
What computational complexity does prefix beam search have with respect to beam width $W$ and vocabulary size $|\Omega|$ per frame?
- [A] $\mathcal{O}(1)$
- [B] $\mathcal{O}(W \cdot |\Omega|)$
- [C] $\mathcal{O}(W^2 \cdot |\Omega|^2)$
- [D] $\mathcal{O}(2^W)$

**Correct Answer**: **[B]**  
*Explanation*: Each of the $W$ prefixes generates $|\Omega|$ candidate symbol extensions.

---

### Question 5:
Approximately how much absolute WER improvement does beam search typically yield over greedy decoding in Kashmiri ASR?
- [A] 0.1%
- [B] 4% to 6% absolute
- [C] 25% absolute
- [D] Greedy is better than beam search.

**Correct Answer**: **[B]**  
*Explanation*: Beam search consistently recovers 4–6% WER by exploring multi-path marginalization and language model candidates.
