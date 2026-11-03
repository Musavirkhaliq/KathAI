# Kashmiri Linguistic & Orthographic Analysis for Speech & Language Modeling

- **Language Name**: Kashmiri / کٲشُر / कॉशुर (*Koshur*)
- **Language Codes**: ISO 639-1: `ks`, ISO 639-2/3: `kas`
- **Family**: Indo-Aryan -> Northwestern Zone -> Dardic subgroup
- **Primary Scripts**: Perso-Arabic (`kas_Arab`, official), Devanagari (`kas_Deva`), Romanized Latin (conversational)
- **Primary Dialect Zones**: Maraz (Southern Kashmir), Kamraz (Northern Kashmir), Yamraz (Central / Srinagar Standard), Poguli, Kishtwari.

---

## 1. Distinctive Phonological & Morphological Characteristics

### 1.1 Vowel Richness and Central Vowels
Unlike standard Hindi/Urdu which possess 10 vowel phonemes, Kashmiri features a complex 16-18 vowel inventory with critical phonemic central vowels:
- High central unrounded vowel: /ɨ/ (short) and /ɨː/ (long)
  - Represented in Perso-Arabic by: `U+0672` (Arabic Letter Alef with Wavy Hamza Above: ٲ) or `U+065F` (Wavy Hamza Below).
- Mid central unrounded vowel: /ə/ (short) and /əː/ (long)
  - Represented in Perso-Arabic by: Alef with Madda / Super-diacritic.
- In Devanagari, these are marked with special diacritics: ॲ, ऑ, or horizontal bars above vowel symbols.

*Impact on ASR/TTS*: If an ASR normalization pipeline blindly strips diacritics, contrastive minimal pairs collapse into identical graphemes (e.g. *kat* "spin" vs *kət* "small ram"), inducing severe semantic ambiguity.

### 1.2 Palatalization (*Palatal Off-Glides*)
Palatalization is a defining feature of Kashmiri phonology. Almost all consonants (except palatal affricates /tʃ, dʒ/ and /j/) can undergo contrastive phonemic palatalization, typically realized as an offglide [ʲ]:
- Example: /bəlʲ/ (cure/recover) vs /bəl/ (strength).
- In Perso-Arabic Kashmiri orthography, palatalization is marked using:
  - Kashmiri Palatalization Mark (`U+065A` / Arabic Vowel Sign Small V Above), or
  - Appended Chhoti Yeh (`U+06CC` / `U+064A`) forming ligatures.

---

## 2. Orthographic Inconsistencies & Unicode Pitfalls

Across the collected corpora (KTTS, OpenSLR, Wikipedia, Aya, BPCC, Rasa), text exhibits heavy orthographic variation due to non-standardized keyboards, Urdu layout inheritance, and divergent font engines:

| Feature | Standard Kashmiri Perso-Arabic | Common Variant / Corruption | Unicode Codepoint Discrepancy | Normalization Resolution |
| :--- | :--- | :--- | :--- | :--- |
| **Kaf** | Keheh (ک) | Arabic Kaf (ك) | `U+06A9` vs `U+0643` | Canonicalize all `0643` -> `06A9` |
| **Gaf** | Gaf (گ) | Arabic/Persian variants | `U+06AF` | Standardize to `06AF` |
| **Heh** | Heh Goal (ہ / ہـ) | Arabic Heh (ه), Heh Doachashmee (ھ) | `U+06C1` vs `U+0647` vs `U+06BE` | Standardize terminal/isolated to `06C1`, preserve `06BE` strictly for aspirated stops (کھ, تھ, چھ) |
| **Yeh** | Farsi Yeh (ی), Yeh Barree (ے) | Arabic Yeh (ي with dots) | `U+06CC`, `U+06D2` vs `U+064A` | Strip trailing dual dots when used as long vowel /e:/ or /i:/ |
| **Vowel Alef** | Alef with Wavy Hamza (ٲ) | Separate Alef + combining mark | `U+0672` vs `U+0627 + U+065F` | Canonicalize to precomposed `U+0672` via Unicode NFC/NFKC |
| **Digits** | Perso-Arabic / Kashmiri numerals (۰۱۲۳۴۵۶۷۸۹) | Western Arabic digits (0-9) | `U+06F0-06F9` vs `U+0030-0039` | Standardize to spoken word forms for ASR; preserve uniform digits for LLM pretraining |

---

## 3. Code-Switching & Lexical Stratification

Kashmiri discourse exhibits tri-lingual stratification:
1. **Indigenous Dardic Core**: Basic vocabulary, kinship terms, fundamental verbs (*khasun*, *pyon*, *trolun*).
2. **Perso-Arabic Superstratum**: Religious, legal, poetic, and administrative terms borrowed over 7 centuries of Islamic history (*adalat*, *rooh*, *insaaf*).
3. **Sanskrit / Indic Superstratum**: Classical religious, philosophical, and formal register (*shanti*, *praan*, *namaskar*).
4. **Modern English Infiltration**: High code-switching rate in contemporary speech, especially in technical, scientific, political, and conversational registers (*doctor*, *station*, *mobile*, *exam*).

---

## 4. Guidelines for ASR & Tokenization

1. **Acoustic Modeling**: Must retain central vowel diacritics (`U+0672`, `U+065F`, `U+065A`) in transcripts; otherwise, the acoustic model cannot resolve acoustic duration and formant shifts.
2. **ASR Evaluation (WER/CER)**: Must apply an `evaluation_normalizer` that collapses optional decorative diacritics (Zer, Zabar, Pesh) and unifies letter variants, preventing false penalties against valid acoustic hypotheses.
3. **LLM Tokenization**: Standard Llama-3 / Mistral tokenizers split Kashmiri Perso-Arabic into byte-level fragments, yielding a crippling fertility rate of >4.1 tokens/word. A dedicated SentencePiece BPE tokenizer with 16k–32k vocabulary reduces fertility to ~1.28 tokens/word, slashing context window costs by 68%.
