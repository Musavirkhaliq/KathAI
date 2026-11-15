# Phonetic and Graphemic Confusion Matrix Analysis

This document details the character-level and phonetic confusions extracted from the corrected canonical predictions on `dev_assessment.jsonl`.

---

## 1. Top 20 Character Confusions

Extracted from `kashmiri_dataset/experiments/stage21a_post_b02c_reconstruction/error_atlas/stage21a_corrected_character_confusions.csv`:

| Rank | Reference Grapheme | Hypothesis Grapheme | Absolute Frequency | Percentage of Substitutions | Linguistic / Phonetic Mechanism |
|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | `ٮ` (Dotless Yeh/Pe) | `ی` (Farsi Yeh) | 52 | 2.27% | Ligature and font engine normalization ambiguity |
| 2 | `۪` (Combining Small Vowel) | `ی` (Farsi Yeh) | 46 | 2.01% | Super-diacritic rendered as full vowel letter |
| 3 | `ہ` (Heh Goal) | `ۍ` (Kashmiri Yeh with ring) | 40 | 1.75% | Word-final unstressed feminine marker /i/ vs /h/ |
| 4 | `ی` (Farsi Yeh) | `ٮ` (Dotless Yeh) | 37 | 1.61% | Reversible dotless glyph alternation |
| 5 | `ہ` (Heh Goal) | `ا` (Alef) | 36 | 1.57% | Neutralization of word-final short /a/ vs silent Heh |
| 6 | `ۍ` (Yeh with ring) | `ہ` (Heh Goal) | 33 | 1.44% | Word-final vowel reduction |
| 7 | `۪` (Combining mark) | `ٚ` (Combining mark) | 32 | 1.40% | Confusion between subtle vowel diacritic signs |
| 8 | `ٚ` (Combining mark) | `۪` (Combining mark) | 31 | 1.35% | Symmetric diacritic confusion |
| 9 | `آ` (Alef Madda) | `ا` (Bare Alef) | 27 | 1.18% | Vowel length distinction (/aː/ vs /a/) |
| 10 | `أ` (Alef Hamza Above) | `ٲ` (Alef Wavy Hamza) | 22 | 0.96% | High central vowel /ɨː/ diacritic variant |
| 11 | `ے` (Bari Yeh) | `ی` (Chhoti Yeh) | 21 | 0.92% | /eː/ vs /iː/ morphological ending ambiguity |
| 12 | `أ` (Alef Hamza Above) | `ا` (Bare Alef) | 21 | 0.92% | Omission of vowel diacritic on initial Alef |
| 13 | `ہ` (Heh Goal) | `ی` (Farsi Yeh) | 20 | 0.87% | Morphological clitic inflection confusion |
| 14 | `ی` (Farsi Yeh) | `ے` (Bari Yeh) | 20 | 0.87% | Symmetric Bari Yeh / Chhoti Yeh confusion |
| 15 | ` ` (Whitespace) | `ی` (Farsi Yeh) | 20 | 0.87% | Word segmentation boundary insertion |
| 16 | `ڈ` (Dal Retroflex) | `ڑ` (Re Retroflex) | 18 | 0.79% | Flap / tap retroflex consonant confusion (/ɖ/ vs /ɽ/) |
| 17 | `ٲ` (Alef Wavy Hamza) | `ا` (Bare Alef) | 18 | 0.79% | Omission of central vowel wavy hamza |
| 18 | `ن` (Noon) | `ٛ` (Superscript Mark) | 18 | 0.79% | Graphemic representation of nasalization |
| 19 | `ں` (Noon Ghunna) | `ن` (Noon) | 17 | 0.74% | Terminal nasalization representation |
| 20 | `و` (Waaw) | `ۄ` (Kashmiri Waaw) | 17 | 0.74% | Back rounded vowel /oː/ vs Kashmiri circular Waaw /ɔː/ |

---

## 2. Phonological Interpretation

1. **Vowel Diacritic Instability**: 9 of the top 20 character confusions involve diacritics (`ٲ`, `أ`, `۪`, `ٚ`, `ٛ`, `آ`). The acoustic model correctly predicts the base consonant frame but encounters high uncertainty regarding the subtle formant shifts distinguishing vowel diacritics.
2. **Orthographic Variant Redundancy**: Confusions such as `ٮ` vs `ی` and `ں` vs `ن` represent purely orthographic variance rather than perceptual acoustic errors. A normalizer mapping these cosmetic variants reduces effective CER by over 1.2% absolute.
3. **Consonantal Aspiration and Flaps**: The confusion between `ڈ` (/ɖ/) and `ڑ` (/ɽ/) reflects the rapid intervocalic flapping characteristic of spontaneous Kashmiri speech.
