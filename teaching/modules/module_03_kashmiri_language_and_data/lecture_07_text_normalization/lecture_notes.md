# Lecture 07: Text Normalization, Unicode Standardization, and Certified Normalizer v3

---

## 1. The Normalization Imperative in Low-Resource Speech Recognition

In Automatic Speech Recognition, the acoustic model emits probabilities over a discrete vocabulary $\Omega$. If the ground truth text contains orthographic inconsistencies, typographical variants, or Unicode homoglyphs, two fatal failures occur:
1. **Target Fragmentation**: The same spoken word is represented by multiple disparate character sequences, dividing gradient updates across multiple paths.
2. **Artificial Metric Inflation**: A model that outputs a linguistically valid transcription is penalized with high Word and Character Error Rates simply because it used a different Unicode codepoint than the reference transcriber.

In Kashmiri, where digital text originates from heterogeneous software environments (InPage, MS Word, Android keyboards, iOS keyboards, web scraping), **rigorous text normalization is not a peripheral utility—it is the foundational prerequisite for acoustic model convergence**.

---

## 2. The Anatomy of Unicode Homoglyphs in Perso-Arabic Scripts

Perso-Arabic Unicode blocks contain numerous characters that appear visually indistinguishable on screen but have entirely different hexadecimal codepoints:

### Major Homoglyph Collision Classes:

| Visual Glyph | Name | Arabic Standard | Farsi / Urdu Standard | Kashmiri Certified |
| :---: | :--- | :--- | :--- | :--- |
| **ک / ك** | Kaf | `U+0643` (Arabic Kaf) | `U+06A9` (Keheh) | **`U+06A9`** (`ک`) |
| **ی / ي / ے** | Yeh | `U+064A` (Arabic Yeh) | `U+06CC` (Farsi Yeh) | **`U+06CC`** (`ی`) |
| **ہ / ه / ۃ** | Heh | `U+0647` (Arabic Heh) | `U+06C1` (Heh Goal) | **`U+06C1`** (`ہ`) |
| **گ / گ** | Gaf | `U+06AF` (Gaf) | `U+06AF` | **`U+06AF`** (`گ`) |
| **ٲ / اٴ** | Alef Madda/Hamza | `U+0672` (Single Char) | `U+0627 + U+0654` (Decomposed) | **`U+0672`** (`ٲ`) |

If a normalizer does not canonicalize `U+0643` to `U+06A9`, the model treats them as orthogonal classes, corrupting the confusion matrix.

---

## 3. The Evolutionary History: From v1 to Certified Normalizer v3

The EduCasheer normalization pipeline underwent three major engineering iterations:

1. **Normalizer v1 (Stage 03)**:
   - Naive regex string replacement.
   - Flaw: Stripped all diacritics indiscriminately, inadvertently destroying the central vowel markers `ٲ` and `ٳ`.
2. **Normalizer v2 (Stage 11)**:
   - Introduced Unicode NFC/NFKC canonical decomposition.
   - Flaw: NFKC decomposition splits `ٲ` (`U+0672`) into `U+0627` (Alef) + `U+0654` (Hamza Above), producing multi-token sequences that bloated vocabulary size and misaligned acoustic frames.
3. **Certified Normalizer v3 (Stage 26 / `src/normalization/kashmiri_normalizer.py`)**:
   - Deterministic 5-stage transformation pipeline:
     1. **Punctuation & Control Character Stripping**: Removes non-speech symbols, zero-width non-joiners (`ZWNJ`), and formatting marks while strictly preserving palatalization `ؠ`.
     2. **Homoglyph Canonicalization**: Maps Arabic and Urdu variants to the certified Kashmiri Perso-Arabic codepoints.
     3. **Selective Diacritic Stripping**: Strips decorative Arabic harakat (*fathah*, *kasrah*, *dammah*, *tanwin*) while **strictly preserving** semantic Kashmiri vowel markers (`ٲ`, `ٳ`, `ۄ`, `ۆ`).
     4. **Whitespace Regularization**: Collapses multiple spaces, tabs, and newlines into single spaces (`|` token).
     5. **Pre-composed Normalization**: Enforces single-codepoint representation for pre-composed Kashmiri vowels.
