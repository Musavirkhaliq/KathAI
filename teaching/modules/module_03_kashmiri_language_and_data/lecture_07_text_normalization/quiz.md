# Quiz: Lecture 07 (Text Normalization)

---

### Question 1:
What is the primary danger of applying standard Unicode NFKC normalization to Kashmiri text?
- [A] It converts all text to uppercase English letters.
- [B] It decomposes precomposed Kashmiri central vowels (like `ٲ` U+0672) into multi-token sequences with separate combining diacritics, disrupting character-level acoustic alignment.
- [C] It deletes all spaces between words.
- [D] It reverses the right-to-left reading order.

**Correct Answer**: **[B]**  
*Explanation*: NFKC splits `ٲ` into `U+0627` + `U+0654`, destroying the single-token representation required for stable CTC training.

---

### Question 2:
What mathematical property ensures that normalizing an already normalized text string produces no further changes?
- [A] Associativity
- [B] Idempotence ($\mathcal{N}(\mathcal{N}(s)) = \mathcal{N}(s)$)
- [C] Linearity
- [D] Orthogonality

**Correct Answer**: **[B]**  
*Explanation*: Idempotence is the defining property of a stable normalizer.

---

### Question 3:
Why must Arabic Kaf (`U+0643`) be mapped to Perso-Arabic Keheh (`U+06A9`)?
- [A] Because `U+0643` is an invalid Unicode codepoint.
- [B] Because they are visually identical homoglyphs, and failing to unify them fragments the vocabulary and causes false evaluation errors.
- [C] Because PyTorch does not support Arabic characters.
- [D] To reduce file size on disk.

**Correct Answer**: **[B]**  
*Explanation*: Unifying homoglyphs prevents vocabulary explosion and metric distortion.

---

### Question 4:
Which diacritics are selectively preserved by Certified Normalizer v3?
- [A] Arabic Fathah and Dammah
- [B] Kashmiri central vowel markers (`ٲ`, `ٳ`, `ۄ`, `ۆ`)
- [C] Latin accents
- [D] Zero-width non-joiners

**Correct Answer**: **[B]**  
*Explanation*: The Kashmiri central vowel markers are phonemic and essential for semantic meaning.

---

### Question 5:
What impact did introducing Certified Normalizer v3 have on historical Kashmiri ASR models?
- [A] It increased WER by 5%.
- [B] It eliminated an estimated 4–8% of false-positive Character and Word Error Rate penalties caused by homoglyph mismatches.
- [C] It tripled the training time.
- [D] It changed the sampling rate of the audio.

**Correct Answer**: **[B]**  
*Explanation*: Consistent normalization ensures that predictions matching reference phonology are not penalized by arbitrary typographical variations.
