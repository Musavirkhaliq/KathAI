# Exercises: Lecture 07 (Text Normalization)

---

## Exercise 1: Unicode Codepoint Dissection
Given the following raw text string:
`s = "كٲشُر زَبان"`
1. List each Unicode character and its 4-digit hexadecimal codepoint.
2. Identify which characters are Arabic homoglyphs and which are decorative diacritics.
3. Write the expected output string after transformation by Certified Normalizer v3.

---

## Exercise 2: NFKC Destructive Decomposition Proof
Standard Unicode NFKC (Compatibility Decomposition) replaces `U+0672` (`ٲ`) with `U+0627` + `U+0654`.
1. Explain why this decomposition is catastrophic for a character-level CTC acoustic model.
2. If a model predicts `U+0672` as a single emission at frame $t$, how many frames must it emit to represent the NFKC decomposed form?
