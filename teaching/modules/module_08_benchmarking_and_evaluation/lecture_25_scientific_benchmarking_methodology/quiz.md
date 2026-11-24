# Quiz: Lecture 25 (Scientific Benchmarking Methodology)

---

### Question 1:
How many frozen evaluation domains are defined in the EduCasheer benchmarking protocol?
- [A] 2
- [B] 4
- [C] 8
- [D] 16

**Correct Answer**: **[C]**  
*Explanation*: 8 frozen domains cover studio, conversational, broadcast, religious, and reverberant acoustic profiles.

---

### Question 2:
What total duration of frozen disjoint audio is held in the 8 evaluation domains?
- [A] 1.5 hours
- [B] 5.0 hours
- [C] 22.70 hours
- [D] 100 hours

**Correct Answer**: **[C]**  
*Explanation*: The 8 domains comprise 22.70 hours across 12,840 utterances.

---

### Question 3:
Why is tuning language model weight $\alpha$ independently for each test domain considered a protocol violation?
- [A] It consumes too much CPU.
- [B] It constitutes test-set contamination (data leakage), producing artificially optimistic results that cannot be replicated in real-world deployment.
- [C] It crashes the KenLM library.
- [D] It changes the text font.

**Correct Answer**: **[B]**  
*Explanation*: Hyperparameters must be fixed prior to evaluating test sets.

---

### Question 4:
What text normalization must be executed prior to calculating benchmark Levenshtein metrics?
- [A] No normalization
- [B] Certified Normalizer v3 applied identically to both references and hypotheses
- [C] English spell-checker
- [D] Lowercase ASCII conversion

**Correct Answer**: **[B]**  
*Explanation*: Applying Normalizer v3 guarantees that typographical homoglyphs do not distort error rates.

---

### Question 5:
Which evaluation domain represents the most acoustically challenging condition in the benchmark?
- [A] DOM-01: Studio Clean
- [B] DOM-03: Broadcast News
- [C] DOM-05: Spontaneous Mosque Sermons (WAAZ)
- [D] DOM-04: Quranic Commentary

**Correct Answer**: **[C]**  
*Explanation*: WAAZ combines open-air acoustic reverberation, loudspeaker distortion, and spontaneous crowd noise.
