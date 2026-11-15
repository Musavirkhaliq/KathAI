# Word Error Taxonomy and Error Atlas

This document classifies and quantifies the structural distribution of word errors identified in the corrected canonical evaluation of Kashmiri ASR.

---

## 1. Global Error Distribution

Across the 1,001 diagnostic assessment utterances (16,845 reference words) under the baseline model (E-03 / B-02C + KenLM 4-gram, $\alpha=0.35, \beta=-0.50$):

$$\text{Total Word Errors} = \mathbf{4,121} \quad (\text{WER} = \mathbf{24.46\%})$$

```
┌─────────────────────────────────┬───────────────┬───────────────────────────────┐
│ Error Class                     │ Absolute Count│ Percentage of All Word Errors │
├─────────────────────────────────┼───────────────┼───────────────────────────────┤
│ Substitutions (S)               │ 3,344 words   │ 81.15%                        │
│ Deletions (D)                   │   521 words   │ 12.64%                        │
│ Insertions (I)                  │   256 words   │  6.21%                        │
└─────────────────────────────────┴───────────────┴───────────────────────────────┘
```

Substitutions dominate over 81% of all recognition failures, reflecting fine-grained phonetic and orthographic ambiguities rather than acoustic boundary failures (which manifest as deletions or insertions).

---

## 2. Structural Taxonomy of Word Substitutions

Analysis of the 3,344 word substitutions yields five major linguistic categories:

### 2.1 Category 1: Nasalization and Nun Ghunna Diacritics (22.4% of Substitutions)
In Kashmiri, vowel nasalization can be marked either as a full letter Noon Ghunna (`ں`), an inline Noon (`ن`), or a superscript combining mark (`U+065B` / Arabic Vowel Sign Inverted Small V Above: `ٛ`).
- Examples:
  - `منز` vs `منٛز` (23 occurrences, highest single confusion)
  - `کانہہ` vs `کانٛہہ` (8 occurrences)
  - `کیٚنہہ` vs `کیٚنٛہہ` (6 occurrences)
  - `ہند` vs `ہنٛد` (6 occurrences)
  - `پانژھ` vs `پانٛژھ` (5 occurrences)
- **Linguistic Insight**: These variations often represent identical spoken pronunciations transcribed with differing orthographic conventions across keyboard layouts.

### 2.2 Category 2: Auxiliary Verbs, Copulas & Clitics (18.6% of Substitutions)
Kashmiri possesses an extensive system of clitics and inflected copulas agreeing in gender and number:
- Copula Alternations:
  - `چھےٚ` vs `چھ` (16 occurrences)
  - `چھ` vs `چھو` (6 occurrences)
  - `نہٕ` vs `چھنہٕ` (9 occurrences)
  - `تہ` vs `تہٕ` (10 occurrences)
  - `چھس` vs `چھ` (5 occurrences)
- **Acoustic Insight**: These clitics are unstressed and spoken with extremely short duration (< 120 ms), leading the acoustic model to produce ambiguous posterior distributions.

### 2.3 Category 3: Central Vowel and Diacritic Alternations (26.8% of Substitutions)
Distinctions between central vowels (/ɨ/, /ə/) and front vowels (/i/, /e/):
- Examples:
  - `اصل` vs `اصٕل` (14 occurrences)
  - `اسہ` vs `أسۍ` (8 occurrences)
  - `ٲسۍ` vs `ٲس` (6 occurrences)
  - `سۄ` vs `سہ` (5 occurrences)
  - `وچھان` vs `وٕچھان` (5 occurrences)
  - `کأم` vs `کٲم` (4 occurrences)

### 2.4 Category 4: Consonant Aspiration and Retroflex Alternations (14.2% of Substitutions)
Confusions between unaspirated and aspirated consonants, or dental and retroflex stops:
- Retroflex vs Dental: `ڈ` vs `ڑ` (18 occurrences), `ت` vs `ٹ`
- Aspiration: `ت` vs `تھ`, `ک` vs `کھ`, `پ` vs `پھ`.

### 2.5 Category 5: Lexical and Out-of-Vocabulary Substitutions (18.0% of Substitutions)
Low-frequency words, Persian/Arabic religious borrowings, or English technical code-switching absent from the 4-gram language model vocabulary.
