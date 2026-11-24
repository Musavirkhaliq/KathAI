# Solutions: Lecture 07 (Text Normalization)

---

## Solution to Exercise 1
1. Codepoint analysis of `"كٲشُر زَبان"`:
   - `ك`: `U+0643` (Arabic Kaf) -> Homoglyph
   - `ٲ`: `U+0672` (Kashmiri Alef with wavy hamza) -> Keep
   - `ش`: `U+0634` (Shin) -> Keep
   - `ُ`: `U+064F` (Dammah diacritic) -> Strip
   - `ر`: `U+0631` (Reh) -> Keep
   - ` `: `U+0020` (Space) -> Keep
   - `ز`: `U+0632` (Zain) -> Keep
   - `َ`: `U+064E` (Fathah diacritic) -> Strip
   - `ب`: `U+0628` (Beh) -> Keep
   - `ا`: `U+0627` (Alef) -> Keep
   - `ن`: `U+0646` (Noon) -> Keep

2. Expected Normalized String:
   `"کٲشر زبان"` (`U+06A9 U+0672 U+0634 U+0631 U+0020 U+0632 U+0628 U+0627 U+0646`).

---

## Solution to Exercise 2
1. NFKC splits a single conceptual vowel grapheme into two independent characters: Alef (`U+0627`) and combining Hamza (`U+0654`). In a character-level CTC model, this forces the model to allocate acoustic frames to an abstract combining diacritic that has no independent acoustic existence.
2. Emitting the decomposed form requires at least 2 distinct acoustic frames (or 3 if blank separation is required), inflating the sequence length and increasing the probability of deletion errors during decoding.
