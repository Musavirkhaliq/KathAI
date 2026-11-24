# Lecture 06: Kashmiri Phonology, Central Vowels, and Extended Perso-Arabic Orthography

---

## 1. Introduction: The Linguistic Uniqueness of Kashmiri

Kashmiri (*Koshur*, كٲشُر) belongs to the **Dardic sub-branch** of the Indo-Aryan language family. Spoken by over 7 million people predominantly in the Kashmir Valley, it exhibits typological, phonetic, and morphosyntactic characteristics that diverge sharply from neighboring Indo-Aryan languages like Urdu, Hindi, and Punjabi.

From an ASR perspective, three linguistic phenomena distinguish Kashmiri:
1. **Unusual Central Vowel Inventory**: A rich system of phonemic central vowels (/ɨ/, /ɨː/, /ə/, /əː/) absent in standard Indo-Aryan and Perso-Arabic languages.
2. **Systemic Secondary Articulation (Palatalization)**: Consonants undergo phonemic palatalization ($C^j$), which carries inflectional morphology (e.g., gender, number, and past tense).
3. **Perso-Arabic Orthographic Adaptation**: Written right-to-left in a modified Perso-Arabic script with non-standardized diacritics (*aerab*) and multi-source Unicode representations.

---

## 2. Kashmiri Vowel Inventory and Acoustic Formants

Standard Indo-Aryan languages typically have a 3-way vertical vowel distinction (high, mid, low) and a 2-way horizontal distinction (front, back).
Kashmiri features a **3-way horizontal contrast**: **Front**, **Central**, and **Back**, each with phonemic length contrasts (short vs. long).

### The Complete Kashmiri Vowel Matrix:

| Position | Front (Unrounded) | Central (Unrounded) | Back (Rounded) |
| :--- | :--- | :--- | :--- |
| **High (Short)** | /i/ (اِ) | **/ɨ/ (إ)** | /u/ (اُ) |
| **High (Long)** | /iː/ (ای) | **/ɨː/ (ٳ)** | /uː/ (اوٗ) |
| **Mid (Short)** | /e/ (ای) | **/ə/ (اۆ)** | /o/ (او) |
| **Mid (Long)** | /eː/ (ای) | **/əː/ (ٲ)** | /oː/ (او) |
| **Low (Short)** | - | /a/ (اَ) | /ɔ/ (اۄ) |
| **Low (Long)** | - | /aː/ (آ) | /ɔː/ (اۄ) |

### Acoustic Formant Spaces ($F_1$ vs. $F_2$)
Central vowels occupy the acoustic center of the vowel quadrilateral:
- High front /i/: $F_1 \approx 300 \text{ Hz}, F_2 \approx 2200 \text{ Hz}$
- High central **/ɨ/**: $F_1 \approx 340 \text{ Hz}, F_2 \approx 1450 \text{ Hz}$
- High back /u/: $F_1 \approx 320 \text{ Hz}, F_2 \approx 800 \text{ Hz}$
- Mid central **/ə/**: $F_1 \approx 480 \text{ Hz}, F_2 \approx 1500 \text{ Hz}$
- Low-mid central **/əː/** (represented by `ٲ`): $F_1 \approx 580 \text{ Hz}, F_2 \approx 1550 \text{ Hz}$

Because $F_2$ is intermediate ($1400 - 1600 \text{ Hz}$), acoustic models trained on Urdu or Hindi invariably collapse Kashmiri central vowels into either front vowels (/i/, /e/) or back vowels (/u/, /o/), leading to catastrophic semantic error rates.

---

## 3. Palatalization and Morphological Signalling

In Kashmiri, almost any consonant can be palatalized. Palatalization is denoted orthographically by the small zero-width joined Kashmiri yae (`ؠ`, Unicode `U+0620`):
- `/b/` vs. `/bʲ/` (e.g., `دَب` /dab/, a fall vs. `دَبؠ` /dabʲ/, falls)
- `/k/` vs. `/kʲ/`
- `/n/` vs. `/nʲ/`

Acoustically, palatalization causes a rapid upward transition in the second formant ($F_2$) of the following vowel or an elongated high-frequency formant tail before closure. In CTC modeling, if the vocabulary omits `ؠ`, the grammatical inflections of the entire language are lost.

---

## 4. The Perso-Arabic Orthographic Quagmire

Kashmiri was historically written in Sharada script, but for several centuries has utilized an extended Perso-Arabic alphabet. Unlike Persian or Arabic, which have 32 and 28 letters respectively, the standard Kashmiri alphabet recognized by the Jammu & Kashmir Academy of Art, Culture and Languages contains **44 basic letters and diacritics**:

### Critical Extended Graphemes:
1. `ٲ` (`U+0672`, Alef with wavy hamza above): Represents long central vowel /əː/.
2. `ٳ` (`U+0673`, Alef with wavy hamza below): Represents long high-central vowel /ɨː/.
3. `ۄ` (`U+06C4`, Kashmiri Vav): Represents short back rounded /ɔ/.
4. `ۆ` (`U+06C6`, Vav with dagger / circle): Represents mid-central vowel /ə/.
5. `ؠ` (`U+0620`, Kashmiri Kashmiri Letter Kashmiri Yeh): Signifies palatalization.

### The Diacritic Dropping Dilemma:
In printed literature and digital communication, Kashmiri writers frequently omit short vowel diacritics (*zer*, *zabar*, *pesh*, *taltrein*). Consequently, a single written word form `سر` can represent:
- /sar/ (pond / head)
- /sɨr/ (secret)
- /sur/ (ashes)

An ASR acoustic model decoding to diacriticized text faces extreme ambiguity unless guided by an external language model or trained with standardized orthographic targets.
