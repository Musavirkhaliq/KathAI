# Quiz: Lecture 09 (Dataset Manifests and Zero-Overlap Speaker Splits)

---

### Question 1:
What is the primary danger of speaker leakage between training and testing sets?
- [A] It causes CUDA out-of-memory errors.
- [B] The model memorizes speaker biometric traits rather than general phonetics, creating an artificially low WER that fails on unseen speakers.
- [C] It causes the audio files to delete themselves.
- [D] It corrupts the vocabulary file.

**Correct Answer**: **[B]**  
*Explanation*: Neural models easily overfit to individual speaker acoustic signatures, yielding illusory performance gains.

---

### Question 2:
In EduCasheer, how many frozen disjoint evaluation domains were established?
- [A] 1
- [B] 4
- [C] 8
- [D] 25

**Correct Answer**: **[C]**  
*Explanation*: 8 frozen disjoint domains evaluate robustness across clean, conversational, broadcast, religious, and reverberant audio.

---

### Question 3:
What format is used for EduCasheer dataset manifests?
- [A] Binary Pickle files
- [B] JSON Lines (`.jsonl`)
- [C] Microsoft Excel `.xlsx`
- [D] Raw text strings separated by commas

**Correct Answer**: **[B]**  
*Explanation*: JSON Lines allows streaming inspection, simple schema evolution, and fast parsing.

---

### Question 4:
Why is duration sorting (bucketing) used during batch construction?
- [A] To ensure the loudest audio is processed first.
- [B] To minimize the number of `<pad>` tokens injected into batches, maximizing GPU compute efficiency.
- [C] To sort speakers alphabetically.
- [D] To satisfy Unicode constraints.

**Correct Answer**: **[B]**  
*Explanation*: Grouping similar-length sequences prevents short utterances from being padded with massive blocks of useless zeros.

---

### Question 5:
What was the observed WER gap between random splitting and disjoint splitting in the Stage 14 audit?
- [A] 0.5%
- [B] 2.1%
- [C] ~17.4% absolute (16.8% random vs. 34.2% disjoint)
- [D] Random splitting had higher WER.

**Correct Answer**: **[C]**  
*Explanation*: The 17.4% gap demonstrated the immense distortion caused by speaker leakage.
