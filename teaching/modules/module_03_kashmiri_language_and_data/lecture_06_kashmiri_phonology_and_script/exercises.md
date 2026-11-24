# Exercises: Lecture 06 (Kashmiri Phonology and Orthography)

---

## Exercise 1: Vowel Formant Identification
Given three speech frames with the following extracted formant pairs:
- Frame A: $F_1 = 310 \text{ Hz}, F_2 = 2250 \text{ Hz}$
- Frame B: $F_1 = 345 \text{ Hz}, F_2 = 1460 \text{ Hz}$
- Frame C: $F_1 = 320 \text{ Hz}, F_2 = 810 \text{ Hz}$

Identify which frame corresponds to /i/ (front), /ɨ/ (central), and /u/ (back). Explain your reasoning based on the acoustic positioning of $F_2$.

---

## Exercise 2: Unicode Homoglyph Collision
In Perso-Arabic typography, the letter *Yeh* can be written using multiple distinct Unicode codepoints:
1. `U+06CC` (Farsi Yeh: `ی`)
2. `U+064A` (Arabic Yeh: `ي`, with two dots below)
3. `U+06D2` (Urdu Yeh Barree: `ے`)

Explain what happens during CTC training if the training transcripts contain random mixtures of `U+06CC` and `U+064A`, but the vocabulary file only includes `U+06CC`.
