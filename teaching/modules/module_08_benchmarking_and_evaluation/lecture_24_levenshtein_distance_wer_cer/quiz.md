# Quiz: Lecture 24 (Levenshtein Distance, WER, and CER)

---

### Question 1:
What are the three atomic editing operations measured by the Levenshtein distance?
- [A] Fourier, Wavelet, Mel
- [B] Substitution, Deletion, Insertion
- [C] Convolution, Attention, Pooling
- [D] Normalization, Quantization, Scaling

**Correct Answer**: **[B]**  
*Explanation*: The edit distance counts substitutions, deletions, and insertions.

---

### Question 2:
Can Word Error Rate (WER) exceed 100%?
- [A] No, it is mathematically bounded between 0% and 100%.
- [B] Yes, if the hypothesis contains more insertion errors than reference words ($I > N$).
- [C] Only on GPU hardware.
- [D] Only when using subwords.

**Correct Answer**: **[B]**  
*Explanation*: Unconstrained insertions can make the numerator arbitrarily larger than $N$.

---

### Question 3:
Why is micro-averaged WER universally required in scientific speech benchmarks?
- [A] It runs faster in Python.
- [B] It aggregates total errors across total words, preventing short outlier sentences from skewing the benchmark metric.
- [C] It requires fewer test files.
- [D] It eliminates punctuation automatically.

**Correct Answer**: **[B]**  
*Explanation*: Micro-averaging weights every spoken word equally.

---

### Question 4:
What is the computational complexity of the dynamic programming Levenshtein algorithm for sequences of length $N$ and $M$?
- [A] $\mathcal{O}(N + M)$
- [B] $\mathcal{O}(NM)$
- [C] $\mathcal{O}(2^{N+M})$
- [D] $\mathcal{O}(1)$

**Correct Answer**: **[B]**  
*Explanation*: Filling the $N \times M$ matrix takes $\mathcal{O}(NM)$ operations.

---

### Question 5:
In the formula $\text{WER} = \frac{S + D + I}{N}$, what does $N$ represent?
- [A] The number of words in the model hypothesis.
- [B] The total number of words in the ground-truth reference transcription ($C + S + D$).
- [C] The duration of the audio in seconds.
- [D] The sample rate.

**Correct Answer**: **[B]**  
*Explanation*: $N$ is the total reference word count.
