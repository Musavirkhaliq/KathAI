# Solutions: Lecture 06 (Kashmiri Phonology and Orthography)

---

## Solution to Exercise 1
- **Frame A ($F_1 = 310, F_2 = 2250 \text{ Hz}$)**: Corresponds to **/i/**. High $F_2$ indicates a front unrounded vowel where the tongue body is advanced toward the alveolar ridge.
- **Frame B ($F_1 = 345, F_2 = 1460 \text{ Hz}$)**: Corresponds to **/ɨ/**. Intermediate $F_2$ (~1460 Hz) is the acoustic hallmark of high central vowel production.
- **Frame C ($F_1 = 320, F_2 = 810 \text{ Hz}$)**: Corresponds to **/u/**. Low $F_2$ reflects vocal tract elongation from lip rounding and tongue retraction.

---

## Solution to Exercise 2
If `U+064A` (Arabic Yeh) appears in the reference text but is absent from `configs/vocab.json`:
1. The tokenizer maps `U+064A` to `<unk>` (index 4) or crashes with an unmapped token error.
2. The acoustic model is penalized during backpropagation for emitting the visually identical character `U+06CC`.
3. The model's loss fails to converge, and CER artificially increases because identical linguistic tokens are treated as distinct orthogonal class indices.
