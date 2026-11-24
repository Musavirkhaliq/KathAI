# Quiz: Lecture 22 (N-Gram Language Models and KenLM)

---

### Question 1:
What fundamental linguistic question is addressed by Kneser-Ney continuation probability?
- [A] How loud the speaker's voice was.
- [B] In how many distinct linguistic contexts a word appears as a novel completion, rather than simply how frequent it is overall.
- [C] How many letters are in the alphabet.
- [D] The duration of the audio clip.

**Correct Answer**: **[B]**  
*Explanation*: Continuation probability measures contextual versatility, avoiding high backoff weights for words that only follow single specific contexts.

---

### Question 2:
Why does Modified Kneser-Ney use three separate discount parameters ($D_1, D_2, D_3$)?
- [A] To run on 3 GPUs.
- [B] Because $n$-grams with low frequency (count 1 or 2) require different discount rates than high-frequency $n$-grams to maximize likelihood on held-out text.
- [C] To support Arabic, Persian, and Urdu.
- [D] To reduce vocabulary size.

**Correct Answer**: **[B]**  
*Explanation*: Three-tier discounting provides optimal smoothing across the frequency spectrum.

---

### Question 3:
What data structure allows KenLM to evaluate language model scores in under 200 nanoseconds?
- [A] Python dictionaries
- [B] Sorted binary tries and probing hash tables optimized for CPU L3 cache lines
- [C] SQL database tables
- [D] Excel spreadsheets

**Correct Answer**: **[B]**  
*Explanation*: KenLM's binary trie minimizes memory overhead and CPU cache misses.

---

### Question 4:
What metric evaluates language model performance on unseen text independently of ASR?
- [A] Word Error Rate (WER)
- [B] Perplexity (PPL)
- [C] Signal-to-Noise Ratio (SNR)
- [D] Receptive Field

**Correct Answer**: **[B]**  
*Explanation*: Perplexity ($10^H$) measures the effective branching factor of the language model.

---

### Question 5:
What text preprocessing must be applied to the language model training corpus?
- [A] It must be converted to uppercase English.
- [B] It must undergo the exact same Certified Normalizer v3 transformation applied to the acoustic transcripts.
- [C] All vowels must be deleted.
- [D] Text must be stored in binary float32 format.

**Correct Answer**: **[B]**  
*Explanation*: Text normalization must be identical across AM and LM to prevent vocabulary mismatches during shallow fusion.
