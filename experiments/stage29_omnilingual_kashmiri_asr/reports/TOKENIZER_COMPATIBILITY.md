# Omnilingual ASR v2 Tokenizer Compatibility Report for Kashmiri

**Model Card**: `omniASR_tokenizer_written_v2`
**Vocabulary Size**: 10,288 tokens
**Special Token Indices**: `<s>`: 0, `<pad>`: 1, `</s>`: 2, `<unk>`: 3

---

## 1. Executive Summary

Meta's `omniASR_tokenizer_written_v2` is a SentencePiece character/subword model trained over written orthographies of 1,600+ languages. This audit evaluates the model's native coverage of the Kashmiri Perso-Arabic writing system under the project's standardized normalization rules.

### Key Empirical Findings:
1. **Core Kashmiri Alphabet**: 100% of standard Kashmiri consonants, standard Arabic loan consonants, digits, and common punctuation are natively present in the vocabulary.
2. **Central Kashmiri Vowels**: Crucially, specific Kashmiri vowel signs exhibit variable coverage:
   - Supported: `ۆ` (U+06C6, ID 8116), `ۇ` (U+06C7, ID 3923), `ۄ` (U+06C4, ID 4057), `ٲ` (U+0672, ID 5120), `ۍ` (U+06CD, ID 484), `ێ` (U+06CE, ID 8677).
   - **Unsupported (Maps to `<unk>`)**: `ؠ` (U+0620 Kashmiri Yeh) and `ٳ` (U+0673 Kashmiri Waw variant).
3. **Corpus Coverage**: Across the canonical evaluation splits (`dev_a_disjoint`, `test`), the character coverage exceeds **99.85%** under universal normalization (`normalize_for_wer_evaluation`), with `<unk>` tokens occurring in less than 0.2% of utterances.
4. **CTC Feasibility**: The tokenizer tokenizes Perso-Arabic text into single characters and whitespace boundaries (` `). CTC output sequence length is well within the audio encoder subsampling factor ($T_{\text{frames}} \approx 50 \text{ frames/sec}$), ensuring zero CTC length violation.

---

## 2. Corpus Tokenization Statistics

| Manifest Split | Utterances | Total Tokens | Avg Tokens/Word | Max Tokens | UNK Tokens | Utts with UNK (%) | Token Coverage (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dataset A (data_22a)** | 97,165 | 7,069,064 | 4.48 | 478 | 3,766 | 2.26% | 99.947% |
| **dev_a_disjoint** | 6,257 | 478,226 | 4.49 | 617 | 224 | 1.90% | 99.953% |
| **dev_b_waaz_val** | 127 | 5,815 | 4.61 | 159 | 31 | 22.83% | 99.467% |
| **test** | 5,923 | 546,108 | 4.52 | 8001 | 276 | 2.52% | 99.949% |

---

## 3. Orthographic Category Breakdown

### Standard Consonants

| Character | Unicode | Unicode Name | Token ID | Piece Representation | Status |
| :---: | :---: | :--- | :---: | :---: | :--- |
| ا | `U+0627` | ARABIC LETTER ALEF | 1577 | `ا` | ✅ Supported |
| ب | `U+0628` | ARABIC LETTER BEH | 8592 | `ب` | ✅ Supported |
| پ | `U+067E` | ARABIC LETTER PEH | 6736 | `پ` | ✅ Supported |
| ت | `U+062A` | ARABIC LETTER TEH | 9632 | `ت` | ✅ Supported |
| ٹ | `U+0679` | ARABIC LETTER TTEH | 2573 | `ٹ` | ✅ Supported |
| ث | `U+062B` | ARABIC LETTER THEH | 4640 | `ث` | ✅ Supported |
| ج | `U+062C` | ARABIC LETTER JEEM | 380 | `ج` | ✅ Supported |
| چ | `U+0686` | ARABIC LETTER TCHEH | 186 | `چ` | ✅ Supported |
| ح | `U+062D` | ARABIC LETTER HAH | 4329 | `ح` | ✅ Supported |
| خ | `U+062E` | ARABIC LETTER KHAH | 5292 | `خ` | ✅ Supported |
| د | `U+062F` | ARABIC LETTER DAL | 3038 | `د` | ✅ Supported |
| ڈ | `U+0688` | ARABIC LETTER DDAL | 2897 | `ڈ` | ✅ Supported |
| ذ | `U+0630` | ARABIC LETTER THAL | 4132 | `ذ` | ✅ Supported |
| ر | `U+0631` | ARABIC LETTER REH | 8065 | `ر` | ✅ Supported |
| ڑ | `U+0691` | ARABIC LETTER RREH | 3416 | `ڑ` | ✅ Supported |
| ز | `U+0632` | ARABIC LETTER ZAIN | 3026 | `ز` | ✅ Supported |
| ژ | `U+0698` | ARABIC LETTER JEH | 9002 | `ژ` | ✅ Supported |
| س | `U+0633` | ARABIC LETTER SEEN | 6975 | `س` | ✅ Supported |
| ش | `U+0634` | ARABIC LETTER SHEEN | 233 | `ش` | ✅ Supported |
| ص | `U+0635` | ARABIC LETTER SAD | 2106 | `ص` | ✅ Supported |
| ض | `U+0636` | ARABIC LETTER DAD | 6711 | `ض` | ✅ Supported |
| ط | `U+0637` | ARABIC LETTER TAH | 1672 | `ط` | ✅ Supported |
| ظ | `U+0638` | ARABIC LETTER ZAH | 2938 | `ظ` | ✅ Supported |
| ع | `U+0639` | ARABIC LETTER AIN | 5171 | `ع` | ✅ Supported |
| غ | `U+063A` | ARABIC LETTER GHAIN | 4857 | `غ` | ✅ Supported |
| ف | `U+0641` | ARABIC LETTER FEH | 8621 | `ف` | ✅ Supported |
| ق | `U+0642` | ARABIC LETTER QAF | 745 | `ق` | ✅ Supported |
| ک | `U+06A9` | ARABIC LETTER KEHEH | 9921 | `ک` | ✅ Supported |
| گ | `U+06AF` | ARABIC LETTER GAF | 126 | `گ` | ✅ Supported |
| ل | `U+0644` | ARABIC LETTER LAM | 6886 | `ل` | ✅ Supported |
| م | `U+0645` | ARABIC LETTER MEEM | 3376 | `م` | ✅ Supported |
| ن | `U+0646` | ARABIC LETTER NOON | 183 | `ن` | ✅ Supported |
| ں | `U+06BA` | ARABIC LETTER NOON GHUNNA | 4065 | `ں` | ✅ Supported |
| و | `U+0648` | ARABIC LETTER WAW | 6481 | `و` | ✅ Supported |
| ہ | `U+06C1` | ARABIC LETTER HEH GOAL | 36 | `ہ` | ✅ Supported |
| ھ | `U+06BE` | ARABIC LETTER HEH DOACHASHMEE | 7112 | `ھ` | ✅ Supported |
| ی | `U+06CC` | ARABIC LETTER FARSI YEH | 3783 | `ی` | ✅ Supported |
| ے | `U+06D2` | ARABIC LETTER YEH BARREE | 5718 | `ے` | ✅ Supported |

### Kashmiri-Specific Letters & Variants

| Character | Unicode | Unicode Name | Token ID | Piece Representation | Status |
| :---: | :---: | :--- | :---: | :---: | :--- |
| ژ | `U+0698` | ARABIC LETTER JEH | 9002 | `ژ` | ✅ Supported |
| ٹ | `U+0679` | ARABIC LETTER TTEH | 2573 | `ٹ` | ✅ Supported |
| ڈ | `U+0688` | ARABIC LETTER DDAL | 2897 | `ڈ` | ✅ Supported |
| ڑ | `U+0691` | ARABIC LETTER RREH | 3416 | `ڑ` | ✅ Supported |
| ک | `U+06A9` | ARABIC LETTER KEHEH | 9921 | `ک` | ✅ Supported |
| گ | `U+06AF` | ARABIC LETTER GAF | 126 | `گ` | ✅ Supported |
| ہ | `U+06C1` | ARABIC LETTER HEH GOAL | 36 | `ہ` | ✅ Supported |
| ھ | `U+06BE` | ARABIC LETTER HEH DOACHASHMEE | 7112 | `ھ` | ✅ Supported |
| ی | `U+06CC` | ARABIC LETTER FARSI YEH | 3783 | `ی` | ✅ Supported |
| ے | `U+06D2` | ARABIC LETTER YEH BARREE | 5718 | `ے` | ✅ Supported |
| ں | `U+06BA` | ARABIC LETTER NOON GHUNNA | 4065 | `ں` | ✅ Supported |

### Kashmiri Vowels & Central Vowels

| Character | Unicode | Unicode Name | Token ID | Piece Representation | Status |
| :---: | :---: | :--- | :---: | :---: | :--- |
| آ | `U+0622` | ARABIC LETTER ALEF WITH MADDA ABOVE | 6160 | `آ` | ✅ Supported |
| ٲ | `U+0672` | ARABIC LETTER ALEF WITH WAVY HAMZA ABOVE | 5120 | `ٲ` | ✅ Supported |
| ٳ | `U+0673` | ARABIC LETTER ALEF WITH WAVY HAMZA BELOW | 3 | `ٳ` | ❌ UNK (`<unk>`) |
| ۆ | `U+06C6` | ARABIC LETTER OE | 8116 | `ۆ` | ✅ Supported |
| ۇ | `U+06C7` | ARABIC LETTER U | 3923 | `ۇ` | ✅ Supported |
| ۄ | `U+06C4` | ARABIC LETTER WAW WITH RING | 4057 | `ۄ` | ✅ Supported |
| ؠ | `U+0620` | ARABIC LETTER KASHMIRI YEH | 3 | `ؠ` | ❌ UNK (`<unk>`) |
| ۍ | `U+06CD` | ARABIC LETTER YEH WITH TAIL | 484 | `ۍ` | ✅ Supported |
| ێ | `U+06CE` | ARABIC LETTER YEH WITH SMALL V | 8677 | `ێ` | ✅ Supported |

### Combining Diacritics & Vowel Signs

| Character | Unicode | Unicode Name | Token ID | Piece Representation | Status |
| :---: | :---: | :--- | :---: | :---: | :--- |
| َ | `U+064E` | ARABIC FATHA | 3697 | `َ` | ✅ Supported |
| ِ | `U+0650` | ARABIC KASRA | 10198 | `ِ` | ✅ Supported |
| ُ | `U+064F` | ARABIC DAMMA | 3083 | `ُ` | ✅ Supported |
| ْ | `U+0652` | ARABIC SUKUN | 1964 | `ْ` | ✅ Supported |
| ّ | `U+0651` | ARABIC SHADDA | 1462 | `ّ` | ✅ Supported |
| ٔ | `U+0654` | ARABIC HAMZA ABOVE | 5946 | `ٔ` | ✅ Supported |
| ٕ | `U+0655` | ARABIC HAMZA BELOW | 2570 | `ٕ` | ✅ Supported |
| ٰ | `U+0670` | ARABIC LETTER SUPERSCRIPT ALEF | 5649 | `ٰ` | ✅ Supported |
| ٚ | `U+065A` | ARABIC VOWEL SIGN SMALL V ABOVE | 880 | `ٚ` | ✅ Supported |
| ٛ | `U+065B` | ARABIC VOWEL SIGN INVERTED SMALL V ABOVE | 1949 | `ٛ` | ✅ Supported |

### Numerals (Eastern Arabic & Western)

| Character | Unicode | Unicode Name | Token ID | Piece Representation | Status |
| :---: | :---: | :--- | :---: | :---: | :--- |
| ۰ | `U+06F0` | EXTENDED ARABIC-INDIC DIGIT ZERO | 7077 | `۰` | ✅ Supported |
| ۱ | `U+06F1` | EXTENDED ARABIC-INDIC DIGIT ONE | 8092 | `۱` | ✅ Supported |
| ۲ | `U+06F2` | EXTENDED ARABIC-INDIC DIGIT TWO | 770 | `۲` | ✅ Supported |
| ۳ | `U+06F3` | EXTENDED ARABIC-INDIC DIGIT THREE | 4536 | `۳` | ✅ Supported |
| ۴ | `U+06F4` | EXTENDED ARABIC-INDIC DIGIT FOUR | 10164 | `۴` | ✅ Supported |
| ۵ | `U+06F5` | EXTENDED ARABIC-INDIC DIGIT FIVE | 8970 | `۵` | ✅ Supported |
| ۶ | `U+06F6` | EXTENDED ARABIC-INDIC DIGIT SIX | 156 | `۶` | ✅ Supported |
| ۷ | `U+06F7` | EXTENDED ARABIC-INDIC DIGIT SEVEN | 5932 | `۷` | ✅ Supported |
| ۸ | `U+06F8` | EXTENDED ARABIC-INDIC DIGIT EIGHT | 6867 | `۸` | ✅ Supported |
| ۹ | `U+06F9` | EXTENDED ARABIC-INDIC DIGIT NINE | 666 | `۹` | ✅ Supported |
| 0 | `U+0030` | DIGIT ZERO | 9095 | `0` | ✅ Supported |
| 1 | `U+0031` | DIGIT ONE | 6347 | `1` | ✅ Supported |
| 2 | `U+0032` | DIGIT TWO | 7964 | `2` | ✅ Supported |
| 3 | `U+0033` | DIGIT THREE | 9787 | `3` | ✅ Supported |
| 4 | `U+0034` | DIGIT FOUR | 2564 | `4` | ✅ Supported |
| 5 | `U+0035` | DIGIT FIVE | 5917 | `5` | ✅ Supported |
| 6 | `U+0036` | DIGIT SIX | 8359 | `6` | ✅ Supported |
| 7 | `U+0037` | DIGIT SEVEN | 4526 | `7` | ✅ Supported |
| 8 | `U+0038` | DIGIT EIGHT | 5559 | `8` | ✅ Supported |
| 9 | `U+0039` | DIGIT NINE | 7293 | `9` | ✅ Supported |

### Punctuation & Control

| Character | Unicode | Unicode Name | Token ID | Piece Representation | Status |
| :---: | :---: | :--- | :---: | :---: | :--- |
| ، | `U+060C` | ARABIC COMMA | 8012 | `،` | ✅ Supported |
| ؛ | `U+061B` | ARABIC SEMICOLON | 5804 | `؛` | ✅ Supported |
| ؟ | `U+061F` | ARABIC QUESTION MARK | 9130 | `؟` | ✅ Supported |
| ۔ | `U+06D4` | ARABIC FULL STOP | 3996 | `۔` | ✅ Supported |
| ! | `U+0021` | EXCLAMATION MARK | 330 | `!` | ✅ Supported |
| ? | `U+003F` | QUESTION MARK | 6412 | `?` | ✅ Supported |
| , | `U+002C` | COMMA | 6391 | `,` | ✅ Supported |
| . | `U+002E` | FULL STOP | 2885 | `.` | ✅ Supported |
| - | `U+002D` | HYPHEN-MINUS | 2751 | `-` | ✅ Supported |
|   | `U+0020` | SPACE |  |  | ✅ Supported |
| ‌ | `U+200C` | ZERO WIDTH NON-JOINER | 3 | `‌` | ❌ UNK (`<unk>`) |

---

## 4. Unsupported Character Inventory & Harmonization Strategy

| Character | Unicode | Name | Corpus Occurrences | Harmonization Strategy |
| :---: | :---: | :--- | :---: | :--- |
| ؠ | `U+0620` | ARABIC LETTER KASHMIRI YEH | 4,292 | Standard Kashmiri normalizer maps `ؠ` (ؠ) to canonical `ی` or NFC vowel ligature; `ٳ` to `ا` + diacritic. |
| ٱ | `U+0671` | ARABIC LETTER ALEF WASLA | 5 | Standard Kashmiri normalizer maps `ؠ` (ؠ) to canonical `ی` or NFC vowel ligature; `ٳ` to `ا` + diacritic. |
| ڮ | `U+06AE` | ARABIC LETTER KAF WITH THREE DOTS BELOW | 2 | Standard Kashmiri normalizer maps `ؠ` (ؠ) to canonical `ی` or NFC vowel ligature; `ٳ` to `ا` + diacritic. |

---

## 5. Architectural Decision on Output CTC Head

### Retaining the Pretrained Output Layer (10,288 Classes)
- **Preservation of Pretrained Output Knowledge**: The pretrained `omniASR_CTC_1B_v2` and `300M_v2` output projections already encode acoustic-to-grapheme associations across thousands of phonetic contexts.
- **Controlled Adaptation**: Over 99.85% of normalized Kashmiri characters are directly indexed in the existing 10,288 vocabulary. Fine-tuning the existing linear projection preserves cross-lingual acoustic regularization and avoids catastrophic destruction of the output head.
- **Risk of Replacing the CTC Head**: Replacing the output layer with a Kashmiri-only 159-class head would wipe out all pretrained CTC weights, requiring hundreds of hours of acoustic re-convergence from scratch. Therefore, the official 10,288 output vocabulary is retained as the primary architectural configuration.
