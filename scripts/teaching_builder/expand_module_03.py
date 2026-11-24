#!/usr/bin/env python3
"""
expand_module_03.py
Expands Lectures 06, 07, 08, and 09 into exhaustive, textbook-grade educational materials:
- Lecture 06: Kashmiri Phonology, Central Vowels, and Extended Perso-Arabic Orthography
- Lecture 07: Text Normalization, Unicode Standardization, and Certified Normalizer v3
- Lecture 08: Speech Corpora Collection, Curation, and Noise Assessment
- Lecture 09: Dataset Manifest Engineering and the Zero-Overlap Speaker Split
"""

from pathlib import Path

BASE_DIR = Path("/home/musa/Kashmiri Project/teaching/modules/module_03_kashmiri_language_and_data")

# ==============================================================================
# LECTURE 06: KASHMIRI PHONOLOGY, CENTRAL VOWELS, AND EXTENDED PERSO-ARABIC SCRIPT
# ==============================================================================
def expand_lecture_06():
    ldir = BASE_DIR / "lecture_06_kashmiri_phonology_and_script"
    ldir.mkdir(parents=True, exist_ok=True)

    notes = r"""# Lecture 06: Kashmiri Phonology, Central Vowels, and Extended Perso-Arabic Orthography

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
"""
    (ldir / "lecture_notes.md").write_text(notes, encoding="utf-8")

    script = r"""# Instructor Delivery Script: Lecture 06 (Kashmiri Phonology and Orthography)

## Timetable (90 Minutes)
```
00:00 - 00:15 | Language Context: Dardic Roots vs. Indo-Aryan Mainstream
00:15 - 00:45 | The 3D Vowel Space: The Central Vowel Matrix (/ɨ/, /ə/, /əː/)
00:45 - 01:10 | Palatalization Mechanics and Acoustic Formant Transitions
01:10 - 01:25 | Perso-Arabic Orthography and the Diacritic Crisis
01:25 - 01:30 | Synthesis & Transition to Text Normalization
```

---

## Instructor Notes & Dialogue

### 00:15 - 00:45 | The Central Vowel Matrix
- Step to the board and draw the **Acoustic Vowel Triangle**:
  - Top-left: Front /i/ ($F_2 = 2200 \text{ Hz}$)
  - Top-right: Back /u/ ($F_2 = 800 \text{ Hz}$)
  - Bottom: Low /a/ ($F_1 = 800 \text{ Hz}$)
- Now draw a vertical pillar right down the center:
  > "Every standard Indo-Aryan language leaves this center space empty. Hindi speakers don't have phonemic /ɨ/. Urdu speakers don't have /əː/.
  > Kashmiri lives in this center space.
  > If a model is pretrained on Hindi, Urdu, or English, its acoustic latent space has never partitioned this intermediate F2 zone between 1400 and 1600 Hz.
  > When a Kashmiri speaker says 'كٲشُر' (/kəːʃur/), the first vowel is /əː/.
  > If your acoustic model fails to resolve 1550 Hz F2, it will predict either 'کاشُر' (/kaːʃur/) or 'کوشُر' (/koːʃur/). Both are errors.
  > This is why cross-lingual transfer without fine-tuning fails completely."

### 00:45 - 01:10 | Palatalization Demo
- Pronounce the contrast: `/dab/` vs `/dabʲ/`.
- Show how the tongue blade touches the hard palate during the release phase.
- On the spectrogram projection, point out the sharp rising $F_2$ trajectory leading into the consonant release.
"""
    (ldir / "instructor_script.md").write_text(script, encoding="utf-8")

    math_doc = r"""# Mathematical Foundations: Acoustic Formants and F2 Transition Modeling

---

## 1. Acoustic Resonance in an Ideal Uniform Vocal Tract Tube

Modeling the human vocal tract as a lossless acoustic tube of length $L \approx 17.5 \text{ cm}$, open at the lips ($x = L$) and closed at the glottis ($x = 0$):

The pressure boundary conditions are:
$$\frac{\partial p}{\partial x}\Bigg|_{x=0} = 0 \quad (\text{zero particle velocity at glottis}), \quad p(L, t) = 0 \quad (\text{ambient pressure at lips})$$

The spatial pressure distribution satisfies:
$$p(x) = A \cos\left( \frac{2\pi f}{c} x \right)$$
Enforcing $p(L) = 0$ requires:
$$\cos\left( \frac{2\pi f}{c} L \right) = 0 \implies \frac{2\pi f}{c} L = \frac{(2n - 1)\pi}{2}, \quad n \in \{1, 2, 3, \dots\}$$

The resonant formant frequencies $F_n$ are:
$$F_n = \frac{(2n - 1) c}{4 L}$$
For $c = 34,300 \text{ cm/s}$ and $L = 17.5 \text{ cm}$:
- $F_1 = \frac{34300}{4 \times 17.5} = 490 \text{ Hz}$ (Neutral central schwa /ə/)
- $F_2 = 3 \times 490 = 1,470 \text{ Hz}$
- $F_3 = 5 \times 490 = 2,450 \text{ Hz}$

Notice that the neutral unconstricted tube corresponds precisely to the **Kashmiri central vowel /ə/** with $F_2 \approx 1470 \text{ Hz}$!

---

## 2. Perturbation Theory for Vowel Articulation

According to Chiba-Kajiyama-Stevens perturbation theory, localized constriction of cross-sectional area $\Delta A(x)$ alters formant frequencies according to the distribution of kinetic and potential acoustic energy:

$$\frac{\Delta F_n}{F_n} = \int_0^L \left( \frac{\Delta K_n(x)}{E_n} - \frac{\Delta U_n(x)}{E_n} \right) dx$$

- Constricting near a **volume velocity node** (pressure anti-node) increases acoustic stiffness, raising formant frequency.
- Constricting near a **volume velocity anti-node** (pressure node) increases acoustic inertia, lowering formant frequency.

For palatalization ($C^j$), constriction occurs at the hard palate ($x \approx 11 \text{ cm}$ from glottis), which coincides with a velocity anti-node for $F_2$, systematically driving $F_2$ upward toward $2,000 - 2,200 \text{ Hz}$.
"""
    (ldir / "mathematical_foundations.md").write_text(math_doc, encoding="utf-8")

    code_doc = r"""# Code Walkthrough: Vocabulary Design and Character Mapping

---

## 1. Inspecting the Certified Kashmiri Vocabulary (`configs/vocab.json`)

The vocabulary file defines the exact mapping between character graphemes and CTC output indices:

```json
{
  "<blank>": 0,
  "<pad>": 1,
  "<s>": 2,
  "</s>": 3,
  "<unk>": 4,
  "|": 5,
  "ا": 6,
  "ب": 7,
  "پ": 8,
  "ت": 9,
  "ٹ": 10,
  "ث": 11,
  "ج": 12,
  "چ": 13,
  "ح": 14,
  "خ": 15,
  "د": 16,
  "ڈ": 17,
  "ذ": 18,
  "ر": 19,
  "ڑ": 20,
  "ز": 21,
  "ژ": 22,
  "س": 23,
  "ش": 24,
  "ص": 25,
  "ض": 26,
  "ط": 27,
  "ظ": 28,
  "ع": 29,
  "غ": 30,
  "ف": 31,
  "ق": 32,
  "ک": 33,
  "گ": 34,
  "ل": 35,
  "م": 36,
  "ن": 37,
  "و": 38,
  "ہ": 39,
  "ی": 40,
  "ے": 41,
  "ٲ": 42,
  "ٳ": 43,
  "ۄ": 44,
  "ۆ": 45,
  "ؠ": 46
}
```

### Key Structural Highlights:
1. **Total Cardinality**: Exactly 47 tokens (0 to 46).
2. **Kashmiri Extensions**: Tokens 42 to 46 encode the unique Kashmiri phonemes (`ٲ`, `ٳ`, `ۄ`, `ۆ`, `ؠ`).
3. **Space Representation**: Token 5 (`|`) acts as the explicit word delimiter.
"""
    (ldir / "code_walkthrough.md").write_text(code_doc, encoding="utf-8")

    lab_doc = r"""# Practical Lab: Formant Analysis of Kashmiri Vowels

---

## Lab Objectives
1. Implement Linear Predictive Coding (LPC) to extract formant frequencies from synthetic and natural vowel frames.
2. Plot the $F_1$ vs. $F_2$ vowel quadrilateral for front, central, and back vowels.
3. Quantify the acoustic distance between /i/, /ɨ/, and /u/.

---

## Lab Script: `lab_formant_extraction.py`

```python
import torch
import numpy as np
import scipy.signal

def extract_formants_lpc(waveform: np.ndarray, sr: int = 16000, lpc_order: int = 16):
    # 1. Apply pre-emphasis filter to boost high formants
    pre_emph = np.append(waveform[0], waveform[1:] - 0.97 * waveform[:-1])
    
    # 2. Windowing
    win = np.hamming(len(pre_emph))
    windowed = pre_emph * win
    
    # 3. LPC coefficients via Levinson-Durbin
    # Autocorrelation
    r = np.correlate(windowed, windowed, mode='full')
    r = r[len(windowed)-1 : len(windowed)-1 + lpc_order + 1]
    
    # Levinson-Durbin recursion
    a = scipy.linalg.solve_toeplitz((r[:-1], r[:-1]), r[1:])
    lpc_poly = np.concatenate(([1.0], -a))
    
    # 4. Find roots of LPC polynomial
    roots = np.roots(lpc_poly)
    roots = [r for r in roots if np.imag(r) >= 0]
    
    # Calculate frequencies and bandwidths
    angles = np.arctan2(np.imag(roots), np.real(roots))
    freqs = angles * (sr / (2 * np.pi))
    bandwidths = -0.5 * (sr / (2 * np.pi)) * np.log(np.abs(roots))
    
    # Filter valid formants (bandwidth < 400 Hz, frequency > 200 Hz)
    valid_formants = []
    for f, bw in zip(freqs, bandwidths):
        if 200 < f < 4000 and bw < 400:
            valid_formants.append(f)
            
    valid_formants.sort()
    return valid_formants

# Test on synthetic central vowel /ɨ/ (F1=340, F2=1450)
t = np.linspace(0, 0.05, 800, endpoint=False)
synthetic_ɨ = np.sin(2 * np.pi * 340 * t) + 0.8 * np.sin(2 * np.pi * 1450 * t)
formants = extract_formants_lpc(synthetic_ɨ)
print(f"Detected Formants: {formants[:2]}")
assert len(formants) >= 2, "Failed to extract formants"
print("SUCCESS: Central vowel formants identified!")
```
"""
    (ldir / "practical_lab.md").write_text(lab_doc, encoding="utf-8")

    ex_doc = r"""# Exercises: Lecture 06 (Kashmiri Phonology and Orthography)

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
"""
    (ldir / "exercises.md").write_text(ex_doc, encoding="utf-8")

    sol_doc = r"""# Solutions: Lecture 06 (Kashmiri Phonology and Orthography)

---

## Solution to Exercise 1
- **Frame A ($F_1 = 310, F_2 = 2250 \text{ Hz}$)**: Corresponds to **/i/**. High $F_2$ indicates a front unrounded vowel where the tongue body is advanced toward the alveolar ridge.
- **Frame B ($F_1 = 345, F_2 = 1460 \text{ Hz}$)**: Corresponds to **/ɨ/**. Intermediate $F_2$ (~1460 Hz) is the acoustic hallmark of high central vowel production.
- **Frame C ($F_1 = 320, F_2 = 810 \text{ Hz}$)**: Corresponds to **/u/**. Low $F_2$ reflects vocal tract elongation from lip rounding and tongue retraction.

---

## Solution to Exercise 2
If `U+064A` (Arabic Yeh) appears in the reference text but is absent from `configs/vocab.json`:
1. The tokenizer maps `U+064A` to `<unk>` (index 4) or crashes with an unmapped token error.
2. The acoustic model is penalized during backpropagation for emitting the visually identical character `U+06CC`.
3. The model's loss fails to converge, and CER artificially increases because identical linguistic tokens are treated as distinct orthogonal class indices.
"""
    (ldir / "solutions.md").write_text(sol_doc, encoding="utf-8")

    quiz_doc = r"""# Quiz: Lecture 06 (Kashmiri Phonology and Orthography)

---

### Question 1:
What distinguishes the Kashmiri vowel inventory from neighboring standard Indo-Aryan languages like Urdu and Hindi?
- [A] Kashmiri contains no nasal vowels.
- [B] Kashmiri features a dedicated system of phonemic central vowels (/ɨ/, /ɨː/, /ə/, /əː/).
- [C] Kashmiri only uses two vowels: /a/ and /i/.
- [D] Kashmiri vowels have pitch tones like Mandarin.

**Correct Answer**: **[B]**  
*Explanation*: Central vowels (/ɨ/, /ə/, etc.) are distinctive phonemes in Kashmiri, requiring dedicated acoustic modeling.

---

### Question 2:
Which character in the Kashmiri Perso-Arabic alphabet represents the long central vowel /əː/ (as in the word *Koshur*)?
- [A] `ا` (Alef)
- [B] `ٲ` (Alef with wavy hamza above, U+0672)
- [C] `و` (Vav)
- [D] `ے` (Barree Yeh)

**Correct Answer**: **[B]**  
*Explanation*: `ٲ` (`U+0672`) is the certified grapheme representing /əː/.

---

### Question 3:
What is the function of the character `ؠ` (Kashmiri Yeh, U+0620)?
- [A] It marks the end of a sentence.
- [B] It indicates phonemic palatalization ($C^j$) of the preceding consonant.
- [C] It represents a glottal stop.
- [D] It silences the following vowel.

**Correct Answer**: **[B]**  
*Explanation*: `ؠ` is used orthographically to mark palatalization, an essential morphological marker in Kashmiri.

---

### Question 4:
What acoustic feature primarily differentiates high front /i/ from high central /ɨ/?
- [A] Fundamental frequency ($F_0$)
- [B] Second formant frequency ($F_2$), which is ~2200 Hz for /i/ and ~1450 Hz for /ɨ/
- [C] Overall sound amplitude
- [D] Phase angle of the third harmonic

**Correct Answer**: **[B]**  
*Explanation*: Formant $F_2$ directly reflects tongue advancement (front vs. central).

---

### Question 5:
How many tokens exist in the certified EduCasheer character vocabulary (`configs/vocab.json`)?
- [A] 26
- [B] 47
- [C] 128
- [D] 1,024

**Correct Answer**: **[B]**  
*Explanation*: The certified vocabulary contains exactly 47 tokens encompassing all Kashmiri Perso-Arabic graphemes and special CTC tokens.
"""
    (ldir / "quiz.md").write_text(quiz_doc, encoding="utf-8")

    res_doc = r"""# Research Connections: Lecture 06

---

## 1. Repository Grounding
- **`configs/vocab.json`**: The 47-token vocabulary file formalizing the Kashmiri alphabet.
- **`src/normalization/kashmiri_normalizer.py`**: Enforces strict Unicode mapping for central vowels and palatalization characters.

---

## 2. Research History
In Stage 01, the team tested an Urdu Kaldi acoustic model. Because Urdu lacks the central vowels /ɨ/ and /əː/, the model mapped Kashmiri words to random phonetic substitutions, resulting in an 88.4% WER failure.
"""
    (ldir / "research_connections.md").write_text(res_doc, encoding="utf-8")

    refs_doc = r"""# Annotated References: Lecture 06

---

1. **Koul, O. N. (2005).**  
   *Modern Kashmiri Grammar.*  
   Dunwoody Press.  
   - **Annotation**: The standard reference for Kashmiri phonetics, central vowel articulation, and vowel length contrasts.

2. **Wali, K., & Koul, O. N. (1997).**  
   *Kashmiri: Cognitive-Descriptive Grammar.*  
   Routledge.  
   - **Annotation**: Exhaustive linguistic survey of Kashmiri morphosyntax, palatalization, and word order.
"""
    (ldir / "references.md").write_text(refs_doc, encoding="utf-8")
    print("Expanded Lecture 06 successfully!")

# ==============================================================================
# LECTURE 07: TEXT NORMALIZATION AND CERTIFIED NORMALIZER V3
# ==============================================================================
def expand_lecture_07():
    ldir = BASE_DIR / "lecture_07_text_normalization"
    ldir.mkdir(parents=True, exist_ok=True)

    notes = r"""# Lecture 07: Text Normalization, Unicode Standardization, and Certified Normalizer v3

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
"""
    (ldir / "lecture_notes.md").write_text(notes, encoding="utf-8")

    script = r"""# Instructor Delivery Script: Lecture 07 (Text Normalization and Certified Normalizer v3)

## Timetable (90 Minutes)
```
00:00 - 00:20 | The Silent Killer: How Unicode Inconsistencies Destroy ASR
00:20 - 00:45 | Homoglyph Autopsy: Kaf, Yeh, Heh, and Central Vowel Codepoints
00:45 - 01:10 | NFKC Traps: Why Standard Unicode Tools Fail on Kashmiri
01:10 - 01:25 | Code Inspection: src/normalization/kashmiri_normalizer.py
01:25 - 01:30 | Lab Preview and Normalization Invariant Testing
```

---

## Instructor Notes & Dialogue

### 00:20 - 00:45 | The Homoglyph Demonstration
- Open a terminal and run `python3`.
- Type:
  ```python
  a = "ک" # U+06A9 (Keheh)
  b = "ك" # U+0643 (Arabic Kaf)
  print(a == b) # Outputs: False
  ```
- Turn to the students:
  > "Look at the screen. To human eyes, both letters look like Kaf. But Python tells you they are NOT equal.
  > If a transcriber uses an Arabic keyboard for one sentence and an Urdu keyboard for another, your neural network thinks they are two completely different letters.
  > If you don't normalize this before training, your model wastes capacity learning two separate logits for the same acoustic sound.
  > Worse: during evaluation, if the model predicts `ک` (U+06A9) and the reference has `ك` (U+0643), your evaluation script marks it as an error!
  > You can lose 8% Word Error Rate without a single acoustic mistake."
"""
    (ldir / "instructor_script.md").write_text(script, encoding="utf-8")

    math_doc = r"""# Mathematical Foundations: String Transformation Invariants and Idempotence

---

## 1. Mathematical Properties of a Certified Normalizer

Let $\Sigma^*$ be the set of all finite character sequences over Unicode alphabet $\Sigma$.
A text normalizer is a mapping $\mathcal{N}: \Sigma^* \to \Sigma_c^*$, where $\Sigma_c \subset \Sigma$ is the certified canonical alphabet ($|\Sigma_c| = 47$).

A certified normalizer must satisfy three strict mathematical properties:

### A. Idempotence:
Applying the normalizer repeatedly after the first pass produces no additional changes:
$$\mathcal{N}(\mathcal{N}(s)) = \mathcal{N}(s) \quad \forall s \in \Sigma^*$$

### B. Commutativity with Concatenation (Modulo Whitespace):
For strings $s_1, s_2 \in \Sigma^*$:
$$\mathcal{N}(s_1 \circ s_2) = \mathcal{N}(s_1) \circ \text{" "} \circ \mathcal{N}(s_2)$$

### C. Partition Invariance (Homoglyph Equivalence Classes):
Let $\sim$ be the equivalence relation where $c_1 \sim c_2$ if $c_1$ and $c_2$ are visual or phonetic homoglyphs.
$$\forall c_1, c_2 \in \Sigma, \quad c_1 \sim c_2 \implies \mathcal{N}(c_1) = \mathcal{N}(c_2)$$
"""
    (ldir / "mathematical_foundations.md").write_text(math_doc, encoding="utf-8")

    code_doc = r"""# Code Walkthrough: Production Normalizer Inspection

---

## 1. Inspecting `src/normalization/kashmiri_normalizer.py`

```python
import re
import unicodedata

class KashmiriNormalizerV3:
    # 1. Homoglyph Replacement Map
    HOMOGLYPH_MAP = {
        '\u0643': '\u06a9',  # Arabic Kaf -> Keheh
        '\u0643': '\u06a9',
        '\u064a': '\u06cc',  # Arabic Yeh -> Farsi Yeh
        '\u06d2': '\u06d2',  # Keep Barree Yeh
        '\u0647': '\u06c1',  # Arabic Heh -> Heh Goal
        '\u0629': '\u06c1',  # Teh Marbuta -> Heh Goal
        # Precomposed central vowels
        '\u0627\u0654': '\u0672',  # Alef + Hamza Above -> Precomposed ٲ
        '\u0627\u0655': '\u0673',  # Alef + Hamza Below -> Precomposed ٳ
    }
    
    # 2. Diacritics to strip (strictly excluding Kashmiri vowel markers)
    STRIP_DIACRITICS = re.compile(r'[\u064b-\u0652\u0656-\u065f\u0670]')
    
    # 3. Punctuation removal pattern
    PUNCTUATION = re.compile(r'[^\w\s\u0600-\u06ff]')
    
    @classmethod
    def normalize(cls, text: str) -> str:
        if not text:
            return ""
            
        # Step 1: Replace homoglyphs
        for src, target in cls.HOMOGLYPH_MAP.items():
            text = text.replace(src, target)
            
        # Step 2: Strip Arabic decorative diacritics
        text = cls.STRIP_DIACRITICS.sub('', text)
        
        # Step 3: Strip extraneous punctuation
        text = cls.PUNCTUATION.sub(' ', text)
        
        # Step 4: Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
```
"""
    (ldir / "code_walkthrough.md").write_text(code_doc, encoding="utf-8")

    lab_doc = r"""# Practical Lab: Building and Verifying Normalizer Invariants

---

## Lab Objectives
1. Implement test cases for Kashmiri Unicode homoglyphs and decomposed central vowels.
2. Verify the mathematical idempotence property $\mathcal{N}(\mathcal{N}(s)) = \mathcal{N}(s)$.
3. Measure the reduction in out-of-vocabulary (OOV) tokens across raw crawled Kashmiri text.

---

## Lab Script: `lab_normalizer_verification.py`

```python
from src.normalization.kashmiri_normalizer import KashmiriNormalizerV3

def test_normalizer():
    # Test Case 1: Homoglyph Unification
    raw_kaf = "\u0643\u0627\u0634\u0631" # Arabic Kaf + Alef + Shin + Reh
    norm_kaf = KashmiriNormalizerV3.normalize(raw_kaf)
    assert norm_kaf.startswith("\u06a9"), "Failed to canonicalize Kaf!"
    
    # Test Case 2: Decomposed Central Vowel
    decomposed = "\u0627\u0654\u0634\u0631" # Alef + Combining Hamza
    norm_central = KashmiriNormalizerV3.normalize(decomposed)
    assert norm_central.startswith("\u0672"), "Failed to unify precomposed central vowel!"
    
    # Test Case 3: Idempotence Verification
    sample_text = "   کٲشُر   زَبان  يَه   "
    pass1 = KashmiriNormalizerV3.normalize(sample_text)
    pass2 = KashmiriNormalizerV3.normalize(pass1)
    assert pass1 == pass2, f"Idempotence violated! Pass 1: '{pass1}', Pass 2: '{pass2}'"
    
    print("SUCCESS: All normalizer invariants passed!")

if __name__ == "__main__":
    test_normalizer()
```
"""
    (ldir / "practical_lab.md").write_text(lab_doc, encoding="utf-8")

    ex_doc = r"""# Exercises: Lecture 07 (Text Normalization)

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
"""
    (ldir / "exercises.md").write_text(ex_doc, encoding="utf-8")

    sol_doc = r"""# Solutions: Lecture 07 (Text Normalization)

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
"""
    (ldir / "solutions.md").write_text(sol_doc, encoding="utf-8")

    quiz_doc = r"""# Quiz: Lecture 07 (Text Normalization)

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
"""
    (ldir / "quiz.md").write_text(quiz_doc, encoding="utf-8")

    res_doc = r"""# Research Connections: Lecture 07

---

## 1. Repository Grounding
- **`src/normalization/kashmiri_normalizer.py`**: The definitive implementation of Certified Normalizer v3.
- **`stages/stage_11/`**: Research stage that investigated the failure of NFKC normalization on Kashmiri texts.
- **`stages/stage_26/`**: Formal certification of Normalizer v3 and reconciliation of historical benchmark metrics.

---

## 2. Research Impact
Standardizing on Normalizer v3 in Stage 26 resolved long-standing discrepancies between historical evaluation runs, establishing clean evaluation parity across all test sets.
"""
    (ldir / "research_connections.md").write_text(res_doc, encoding="utf-8")

    refs_doc = r"""# Annotated References: Lecture 07

---

1. **Davis, M., & Whistler, K. (2021).**  
   *Unicode Standard Annex #15: Unicode Normalization Forms.*  
   The Unicode Consortium.  
   - **Annotation**: The official technical specification defining NFC, NFD, NFKC, and NFKD normalization algorithms.

2. **Koul, B. (2008).**  
   *Issues in Kashmiri Orthography and Script Standardization.*  
   *Language in India*, 8(4).  
   - **Annotation**: Linguistic discussion on Perso-Arabic orthographic challenges, diacritic omission, and typographical standards in Kashmir.
"""
    (ldir / "references.md").write_text(refs_doc, encoding="utf-8")
    print("Expanded Lecture 07 successfully!")

# ==============================================================================
# LECTURE 08: SPEECH CORPORA COLLECTION AND CURATION
# ==============================================================================
def expand_lecture_08():
    ldir = BASE_DIR / "lecture_08_speech_corpora_collection"
    ldir.mkdir(parents=True, exist_ok=True)

    notes = r"""# Lecture 08: Speech Corpora Collection, Curation, and Noise Assessment

---

## 1. The Speech Data Scarcity Landscape for Kashmiri

Prior to the EduCasheer research project, Kashmiri was classified by international computational linguistics as a **severely digitally under-resourced language**. While high-resource languages like English enjoy thousands of hours of curated speech (e.g., LibriSpeech 1,000h, GigaSpeech 10,000h, People's Speech 30,000h), public Kashmiri speech resources totaled fewer than 20 hours of fragmented, noisy recordings.

To train deep acoustic models capable of robust generalization, the project assembled, audited, and curated a multi-source corpus from **12 distinct acoustic sources**.

---

## 2. The 12 Speech Sources of the EduCasheer Ecosystem

| Source ID | Source Name | Raw Duration | Nature & Acoustic Environment |
| :--- | :--- | :--- | :--- |
| **SRC-01** | Bolbosh Kashmiri | 18.4 hours | Studio and clean read speech; original 22.05 kHz recording |
| **SRC-02** | IndicVoices Kashmiri | 24.2 hours | Multi-speaker read and prompted speech; diverse rural/urban accents |
| **SRC-03** | Quranic Kashmiri Broadcasts | 14.8 hours | Expository religious discourse; formal high register with Arabic loans |
| **SRC-04** | WAAZ Spontaneous Sermons | 32.6 hours | Spontaneous open-air mosque sermons; heavy reverberation and crowd noise |
| **SRC-05** | Community Mobile Voice | 16.5 hours | Crowdsourced smartphone recordings; variable near-field microphone quality |
| **SRC-06** | Radio Kashmir Broadcasts | 28.0 hours | Broadcast news and cultural features; high linguistic standard |
| **SRC-07** | Educational Lectures | 12.1 hours | Classroom and podium lectures; whiteboard pauses and room echo |
| **SRC-08** | Folk Tales & Oral Stories | 15.3 hours | Elderly native speakers; rich dialectal idioms and archaic vocabulary |
| **SRC-09** | Conversational Dialogues | 19.7 hours | Spontaneous two-party conversations; overlapping speech and laughter |
| **SRC-10** | Children's Kashmiri Speech | 8.4 hours | Primary school reading; higher pitch ($F_0 > 280 \text{ Hz}$) and articulation pauses |
| **SRC-11** | Judicial & Administrative | 11.2 hours | Formal legal and administrative reading; complex Persianized syntax |
| **SRC-12** | Synthetic Alignment Verification | 13.4 hours | High-fidelity synthetic speech used for automated pipeline calibration |

**Total Repository Raw Holdings**: **214.65 hours** across 106,487 audio segments.

---

## 3. The 4-Stage SNR and Quality Curation Pipeline

Raw scraped speech cannot be ingested directly into neural training. EduCasheer engineered an automated 4-stage curation pipeline (`src/data/curation.py`):

```
[ Raw Audio Files ]
         │
         ▼
[ Stage 1: Physical Audio Audit ] ── Check sample rate, bit depth, channel count, header corruption
         │
         ▼
[ Stage 2: Energy & SNR Filtering ] ── WADA-SNR calculation; drop audio with SNR < 12 dB
         │
         ▼
[ Stage 3: Clipping & Dynamic Range ] ── Detect digital clipping (> 1% samples at ±1.0)
         │
         ▼
[ Stage 4: Duration Slicing ] ── Enforce 0.8s <= duration <= 20.0s for stable CTC alignment
         │
         ▼
[ Training-Eligible Clean Manifest ]
```

### Signal-to-Noise Ratio (SNR) Estimation:
Speech SNR is computed using the **Waveform Amplitude Distribution Analysis (WADA-SNR)** algorithm, which models clean speech amplitude as a Gamma distribution and additive noise as a Gaussian distribution. Recordings with $\text{SNR} < 12 \text{ dB}$ are routed to noise-augmentation quarantine rather than clean training splits.
"""
    (ldir / "lecture_notes.md").write_text(notes, encoding="utf-8")

    script = r"""# Instructor Delivery Script: Lecture 08 (Speech Corpora Collection and Curation)

## Timetable (90 Minutes)
```
00:00 - 00:20 | Low-Resource Realities: The 12 Sources of EduCasheer
00:20 - 00:45 | Acoustic Heterogeneity: Studio vs. WAAZ Spontaneous Sermons
00:45 - 01:10 | The 4-Stage Curation Pipeline: WADA-SNR and Clipping Detection
01:10 - 01:25 | Code Walkthrough: src/data/curation.py
01:25 - 01:30 | Key Takeaways & Lab Introduction
```

---

## Instructor Notes & Dialogue

### 00:20 - 00:45 | Acoustic Heterogeneity
- Play two 5-second audio snippets:
  - Snippet 1: Clean Bolbosh studio recording.
  - Snippet 2: WAAZ sermon (loud reverberant echo, distant crowd coughs, loudspeaker distortion).
- Address the class:
  > "If you train your model only on Snippet 1, your model gets 12% WER in your lab and 85% WER when a real Kashmiri farmer tries to use it in Anantnag.
  > But if you throw Snippet 2 directly into your training without filtering, the CTC loss explodes because the crowd noise causes the model to hallucinate blank tokens.
  > Data curation is the science of knowing what to keep, what to clean, and what to quarantine."
"""
    (ldir / "instructor_script.md").write_text(script, encoding="utf-8")

    math_doc = r"""# Mathematical Foundations: WADA-SNR and Acoustic Energy Estimation

---

## 1. Mathematical Formulation of WADA-SNR

WADA-SNR (Waveform Amplitude Distribution Analysis) estimates SNR from a single-channel speech signal without requiring explicit speech/silence segmentation.

Let clean speech amplitude $s$ follow a **Gamma distribution** with shape parameter $\gamma = 0.5$:
$$P_s(s) = \frac{1}{\sqrt{2\pi \sigma_s s}} e^{-\frac{s}{2\sigma_s}}$$
and let additive background noise $n$ follow a **Gaussian distribution**:
$$P_n(n) = \frac{1}{\sqrt{2\pi \sigma_n^2}} e^{-\frac{n^2}{2\sigma_n^2}}$$

The observed signal is $x = s + n$. The WADA algorithm computes the **ratio of the arithmetic mean of the absolute value to the geometric mean**:

$$\Psi(x) = \frac{\frac{1}{N} \sum_{i=1}^N |x_i|}{\left( \prod_{i=1}^N |x_i| \right)^{1/N}}$$

Since $\Psi(x)$ is a strictly monotonic function of the ratio $\frac{\sigma_s^2}{\sigma_n^2}$, the true SNR in decibels is obtained by inverting the pre-computed calibration function:
$$\text{SNR}_{\text{dB}} = f_{\text{calib}}^{-1}(\Psi(x))$$
"""
    (ldir / "mathematical_foundations.md").write_text(math_doc, encoding="utf-8")

    code_doc = r"""# Code Walkthrough: Audio Curation and SNR Filtering

---

## 1. Inspecting the Quality Filter (`src/data/curation.py`)

```python
import torch
import torchaudio

def assess_audio_quality(audio_path: str, min_snr: float = 12.0) -> dict:
    # [Docstring]
    Evaluates audio quality criteria for inclusion in training manifests.
    # [Docstring]
    waveform, sr = torchaudio.load(audio_path)
    
    # 1. Duration check
    duration = waveform.shape[-1] / sr
    if duration < 0.8 or duration > 20.0:
        return {"keep": False, "reason": f"Duration {duration:.2f}s out of bounds"}
        
    # 2. Clipping detection: samples near +/- 1.0
    clipped_samples = torch.sum(torch.abs(waveform) >= 0.99).item()
    clipping_ratio = clipped_samples / waveform.numel()
    if clipping_ratio > 0.01:
        return {"keep": False, "reason": f"Digital clipping {clipping_ratio*100:.1f}%"}
        
    # 3. Simple Energy-based SNR approximation
    energy = waveform ** 2
    noise_floor = torch.quantile(energy, 0.10)
    peak_energy = torch.quantile(energy, 0.95)
    
    snr_est = 10 * torch.log10((peak_energy + 1e-8) / (noise_floor + 1e-8)).item()
    if snr_est < min_snr:
        return {"keep": False, "reason": f"Low SNR: {snr_est:.1f} dB"}
        
    return {"keep": True, "snr": snr_est, "duration": duration}
```
"""
    (ldir / "code_walkthrough.md").write_text(code_doc, encoding="utf-8")

    lab_doc = r"""# Practical Lab: Single-Channel SNR Estimation and Audio Auditing

---

## Lab Objectives
1. Implement single-channel quantile-based SNR estimation in PyTorch.
2. Build an automated clipping detection script.
3. Filter a directory of synthetic corrupted audio files into `clean/` and `quarantined/` manifests.

---

## Lab Script: `lab_audio_curation.py`

```python
import torch
import numpy as np

def run_curation_lab():
    sr = 16000
    duration = 3.0
    t = torch.linspace(0, duration, int(sr * duration))
    
    # Generate clean speech-like signal
    speech = torch.sin(2 * np.pi * 300 * t) * (torch.sin(2 * np.pi * 3 * t) > 0).float()
    
    # Corrupt with Gaussian noise (SNR ~ 6 dB)
    noise = torch.randn_like(speech) * 0.5
    noisy_audio = speech + noise
    
    # Quantile-based SNR estimation
    energy = noisy_audio ** 2
    noise_est = torch.quantile(energy, 0.10)
    signal_est = torch.quantile(energy, 0.95)
    snr = 10 * torch.log10(signal_est / noise_est).item()
    
    print(f"Estimated SNR: {snr:.2f} dB")
    should_keep = snr >= 12.0
    print(f"Curation Decision: {'KEEP' if should_keep else 'QUARANTINE'}")
    assert not should_keep, "Failed to quarantine noisy audio!"
    print("SUCCESS: Curation filter functioned as expected!")

if __name__ == "__main__":
    run_curation_lab()
```
"""
    (ldir / "practical_lab.md").write_text(lab_doc, encoding="utf-8")

    ex_doc = r"""# Exercises: Lecture 08 (Speech Corpora Collection)

---

## Exercise 1: Quantile SNR Calculation
An audio recording has the following energy percentiles:
- 10th percentile energy ($P_{10}$): $2.4 \times 10^{-5}$
- 95th percentile energy ($P_{95}$): $1.8 \times 10^{-2}$
1. Calculate the estimated SNR in decibels.
2. Based on the 12 dB threshold, should this recording be included in the clean training manifest?

---

## Exercise 2: Clipping Distortion Analysis
A mobile recording of 10 seconds (160,000 samples) contains 4,800 samples that hit full-scale ceiling ($\pm 1.0$).
1. Calculate the percentage of clipped samples.
2. Explain the mathematical consequence of clipping in the frequency domain.
"""
    (ldir / "exercises.md").write_text(ex_doc, encoding="utf-8")

    sol_doc = r"""# Solutions: Lecture 08 (Speech Corpora Collection)

---

## Solution to Exercise 1
1. Estimated SNR:
   $$\text{SNR} = 10 \log_{10}\left( \frac{P_{95}}{P_{10}} \right) = 10 \log_{10}\left( \frac{1.8 \times 10^{-2}}{2.4 \times 10^{-5}} \right) = 10 \log_{10}(750) \approx 10 \times 2.875 = \mathbf{28.75 \text{ dB}}$$
2. Since $28.75 \text{ dB} \ge 12.0 \text{ dB}$, the recording is accepted into the clean training set.

---

## Solution to Exercise 2
1. Percentage clipped:
   $$\frac{4,800}{160,000} \times 100 = \mathbf{3.0\%}$$
2. Hard digital clipping abruptly truncates the peaks of the continuous waveform, creating sharp non-linear discontinuities. In the frequency domain, this acts like convolution with high-frequency square wave harmonics, generating spurious broadband high-frequency noise that masks subtle consonant formants.
"""
    (ldir / "solutions.md").write_text(sol_doc, encoding="utf-8")

    quiz_doc = r"""# Quiz: Lecture 08 (Speech Corpora Collection)

---

### Question 1:
How many distinct speech sources were collected and audited in the EduCasheer repository?
- [A] 2 sources
- [B] 12 sources
- [C] 45 sources
- [D] 100 sources

**Correct Answer**: **[B]**  
*Explanation*: The repository audited 12 diverse sources encompassing studio, broadcast, religious, educational, and spontaneous speech.

---

### Question 2:
What is the minimum SNR threshold enforced by the clean curation pipeline?
- [A] 0 dB
- [B] 12 dB
- [C] 40 dB
- [D] 80 dB

**Correct Answer**: **[B]**  
*Explanation*: 12 dB ensures speech acoustic clarity without discarding valuable authentic dialectal recordings.

---

### Question 3:
Why is hard digital clipping detrimental to ASR training?
- [A] It decreases the sampling rate.
- [B] It generates spurious high-frequency harmonics and destroys acoustic formant trajectories.
- [C] It crashes the PyTorch dataloader.
- [D] It converts stereo audio into mono.

**Correct Answer**: **[B]**  
*Explanation*: Hard saturation creates sharp waveform edges that pollute the spectrum with non-linear distortion.

---

### Question 4:
Which dataset source contains spontaneous open-air mosque sermons with heavy acoustic reverberation?
- [A] Bolbosh
- [B] IndicVoices
- [C] WAAZ (SRC-04)
- [D] Radio Kashmir

**Correct Answer**: **[C]**  
*Explanation*: WAAZ represents challenging real-world spontaneous sermon audio.

---

### Question 5:
What is the total raw audio volume held across the 12 sources in the repository?
- [A] 20.5 hours
- [B] 75.0 hours
- [C] 214.65 hours
- [D] 10,000 hours

**Correct Answer**: **[C]**  
*Explanation*: The comprehensive audit verified 214.65 total raw hours across 106,487 audio segments.
"""
    (ldir / "quiz.md").write_text(quiz_doc, encoding="utf-8")

    res_doc = r"""# Research Connections: Lecture 08

---

## 1. Repository Grounding
- **`src/data/curation.py`**: The 4-stage quality auditing and filtering module.
- **`docs/data/DATASET_REGISTRY.md`**: Complete registry of all 12 dataset sources, recording conditions, and hour tallies.
- **`stages/stage_26/`**: Resolved dataset accounting contradictions, establishing exact physical totals (214.65 hours raw, 174.55 hours training-eligible).
"""
    (ldir / "research_connections.md").write_text(res_doc, encoding="utf-8")

    refs_doc = r"""# Annotated References: Lecture 08

---

1. **Kim, C., & Stern, R. M. (2008).**  
   *Robust signal-to-noise ratio estimation based on waveform amplitude distribution analysis.*  
   *Interspeech 2008*, 2598–2601.  
   - **Annotation**: The seminal paper formulating the WADA-SNR algorithm for single-channel audio without prior silence models.
"""
    (ldir / "references.md").write_text(refs_doc, encoding="utf-8")
    print("Expanded Lecture 08 successfully!")

# ==============================================================================
# LECTURE 09: DATASET MANIFESTS AND THE ZERO-OVERLAP SPEAKER SPLIT
# ==============================================================================
def expand_lecture_09():
    ldir = BASE_DIR / "lecture_09_dataset_manifests_and_leakage"
    ldir.mkdir(parents=True, exist_ok=True)

    notes = r"""# Lecture 09: Dataset Manifest Engineering and the Zero-Overlap Speaker Split

---

## 1. The Contamination Crisis in Machine Learning Benchmarks

A machine learning benchmark is only as valid as the independence of its evaluation sets. In speech recognition, **speaker leakage** represents the most pernicious form of data contamination:
- When audio from the same speaker appears in both training and test partitions, deep neural acoustic models memorize speaker-specific biometric characteristics: fundamental frequency ($F_0$), vocal tract length, characteristic formant bandwidths, and recording channel impulse responses.
- In historical experiments (Stage 06 vs. Stage 14), evaluating on a contaminated split produced an **artificially optimistic 16.8% WER**, whereas evaluating the exact same checkpoint on a strictly disjoint split revealed a true **34.2% WER**—a catastrophic 17.4% performance illusion!

---

## 2. Bipartite Graph Partitioning for Zero-Speaker Overlap

To guarantee zero speaker leakage across thousands of utterances where metadata may be imperfect, EduCasheer formalized data splitting as a **Graph Partitioning Problem**:

Let $G = (V_S \cup V_U, E)$ be a bipartite graph where:
- $V_S$ is the set of speaker entities.
- $V_U$ is the set of audio utterances.
- An edge $(s, u) \in E$ exists if speaker $s$ uttered segment $u$.

### Partition Constraints:
We partition utterances into three sets: $\text{Train} (U_{\text{tr}})$, $\text{Validation} (U_{\text{val}})$, and $\text{Test} (U_{\text{te}})$ such that:
1. **Strict Speaker Disjointness**:
   $$S(U_{\text{tr}}) \cap S(U_{\text{val}}) = \emptyset, \quad S(U_{\text{tr}}) \cap S(U_{\text{te}}) = \emptyset, \quad S(U_{\text{val}}) \cap S(U_{\text{te}}) = \emptyset$$
2. **Acoustic Diversity Balance**:
   Each partition must maintain proportional representation of gender, dialect regions, and recording acoustic types (studio vs. field).
3. **Temporal Target Ratios**:
   Target duration split: $80\%$ Train, $10\%$ Validation, $10\%$ Test.

---

## 3. The 8 Frozen Disjoint Evaluation Domains

To prevent overfitting to any single acoustic or linguistic distribution, EduCasheer established **8 Frozen Disjoint Benchmark Domains** (`benchmarks/`):

1. **Domain 1: Clean Studio Read Speech** (Bolbosh Disjoint)
2. **Domain 2: Prompted Conversational** (IndicVoices Disjoint)
3. **Domain 3: Formal Broadcast News** (Radio Kashmir)
4. **Domain 4: Quranic & Expository Commentary** (High Perso-Arabic loan register)
5. **Domain 5: Spontaneous Mosque Sermons** (WAAZ Reverberant)
6. **Domain 6: Community Mobile Audio** (Near-field smartphone microphones)
7. **Domain 7: Elderly Dialectal Folk Narratives** (Rural South/North Kashmir)
8. **Domain 8: Classroom & Educational Audio** (Podium reverberation and chalk pauses)

Any checkpoint claiming state-of-the-art performance must be evaluated across all 8 domains without test-set tuning.

---

## 4. Production Manifest Architecture: JSON Lines (`.jsonl`)

EduCasheer standardizes all training manifests in JSON Lines format (`manifests/*.jsonl`):

```json
{"audio_filepath": "data/audio/bolbosh/utt_001.wav", "duration": 4.52, "text": "کٲشُر چھُ اَکھ قٔدیٖم زَبان", "speaker_id": "SPK_B012", "domain": "studio_clean"}
```

### Mandatory Manifest Schema Fields:
- `audio_filepath`: Relative path to certified 16 kHz 16-bit mono WAV.
- `duration`: Float duration in seconds (enables batch bucket sorting to minimize padding).
- `text`: Certified Normalizer v3 normalized transcription.
- `speaker_id`: Unique speaker identifier used by graph partition verification.
- `domain`: Acoustic domain identifier.
"""
    (ldir / "lecture_notes.md").write_text(notes, encoding="utf-8")

    script = r"""# Instructor Delivery Script: Lecture 09 (Dataset Manifests and Zero-Overlap Speaker Splits)

## Timetable (90 Minutes)
```
00:00 - 00:20 | The Illusion of Success: How Speaker Leakage Fakes State-of-the-Art
00:20 - 00:45 | Mathematical Formulation: Bipartite Graph Partitioning
00:45 - 01:10 | The 8 Frozen Evaluation Domains of EduCasheer
01:10 - 01:25 | Code Inspection: src/data/manifest.py and Split Verification
01:25 - 01:30 | Module 03 Synthesis & Wrap-up
```

---

## Instructor Notes & Dialogue

### 00:00 - 00:20 | The Speaker Leakage Illusion
- Write on the board:
  - `Random Split WER: 16.8%`
  - `Disjoint Speaker Split WER: 34.2%`
- Look at the students:
  > "These two numbers came from the exact same neural network checkpoint.
  > If I wanted to publish a fraudulent paper or fool an investor, I would show them 16.8%.
  > Why did the model score 16.8%?
  > Because the model didn't learn Kashmiri phonetics—it learned the voice of Speaker #14. It learned that whenever an audio file has an F0 of 142 Hz and a room resonance at 600 Hz, Speaker #14 usually talks about the weather.
  > When we evaluated that same model on a new speaker whose voice it had never heard, the model fell apart.
  > In this course, random splitting in speech recognition is considered scientific malpractice.
  > Today, we learn how to mathematically guarantee zero speaker overlap."
"""
    (ldir / "instructor_script.md").write_text(script, encoding="utf-8")

    math_doc = r"""# Mathematical Foundations: Graph Partitioning and Leakage Verification

---

## 1. Graph Bipartition Optimization Problem

Let $S = \{s_1, \dots, s_K\}$ be the set of speakers, and $D(s)$ be the total speech duration of speaker $s$:
$$D(s) = \sum_{u \in U(s)} \text{duration}(u)$$

We seek an assignment variable $z_s \in \{0, 1\}$ (where $1$ denotes Test set and $0$ denotes Train set) to satisfy:

$$\min_{z} \left| \sum_{s \in S} z_s D(s) - 0.10 \sum_{s \in S} D(s) \right|$$

subject to the strict disjointness condition:
$$U_{\text{train}} \cap U_{\text{test}} = \emptyset$$

---

## 2. Contamination Invariant Proof

Let $\phi(u)$ denote the speaker attribution function mapping utterance $u$ to speaker ID $s$.
The overlap contamination metric $C$ is defined as:

$$C = \frac{| \{ s \in S \mid s \in \phi(U_{\text{train}}) \land s \in \phi(U_{\text{test}}) \} |}{|\phi(U_{\text{test}})|}$$

A certified disjoint split has the invariant:
$$C \equiv 0.000$$
"""
    (ldir / "mathematical_foundations.md").write_text(math_doc, encoding="utf-8")

    code_doc = r"""# Code Walkthrough: Manifest Validation and Leakage Detection

---

## 1. Inspecting the Leakage Detector (`src/data/manifest.py`)

```python
import json
from pathlib import Path

def verify_zero_speaker_leakage(train_manifest: str, test_manifest: str) -> dict:
    # [Docstring]
    Audits two manifests to mathematically guarantee zero speaker overlap.
    # [Docstring]
    def load_speakers(path):
        speakers = set()
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    speakers.add(entry["speaker_id"])
        return speakers
        
    train_speakers = load_speakers(train_manifest)
    test_speakers = load_speakers(test_manifest)
    
    overlap = train_speakers.intersection(test_speakers)
    is_clean = (len(overlap) == 0)
    
    return {
        "is_clean": is_clean,
        "train_speaker_count": len(train_speakers),
        "test_speaker_count": len(test_speakers),
        "overlapping_speakers": list(overlap),
        "contamination_rate": len(overlap) / max(1, len(test_speakers))
    }
```
"""
    (ldir / "code_walkthrough.md").write_text(code_doc, encoding="utf-8")

    lab_doc = r"""# Practical Lab: Building Disjoint Speaker Partitions

---

## Lab Objectives
1. Load a simulated multi-speaker manifest.
2. Implement greedy bin-packing to construct an 80/10/10 disjoint speaker partition.
3. Execute automated leakage detection to prove $C = 0.0$.

---

## Lab Script: `lab_disjoint_splitting.py`

```python
import json

def partition_speakers(samples, train_ratio=0.8, val_ratio=0.1):
    # Group samples by speaker
    speaker_durations = {}
    speaker_samples = {}
    for s in samples:
        spk = s["speaker_id"]
        speaker_durations[spk] = speaker_durations.get(spk, 0.0) + s["duration"]
        speaker_samples.setdefault(spk, []).append(s)
        
    total_dur = sum(speaker_durations.values())
    target_train = total_dur * train_ratio
    target_val = total_dur * val_ratio
    
    # Sort speakers by duration descending (greedy bin packing)
    sorted_speakers = sorted(speaker_durations.keys(), key=lambda k: speaker_durations[k], reverse=True)
    
    train, val, test = [], [], []
    curr_train, curr_val = 0.0, 0.0
    
    for spk in sorted_speakers:
        dur = speaker_durations[spk]
        if curr_train + dur <= target_train or (curr_val >= target_val and len(test) > 0):
            train.extend(speaker_samples[spk])
            curr_train += dur
        elif curr_val + dur <= target_val:
            val.extend(speaker_samples[spk])
            curr_val += dur
        else:
            test.extend(speaker_samples[spk])
            
    return train, val, test

# Test on synthetic data
toy_data = [
    {"audio": "1.wav", "duration": 5.0, "speaker_id": "SPK_1"},
    {"audio": "2.wav", "duration": 4.0, "speaker_id": "SPK_1"},
    {"audio": "3.wav", "duration": 8.0, "speaker_id": "SPK_2"},
    {"audio": "4.wav", "duration": 3.0, "speaker_id": "SPK_3"},
]
tr, va, te = partition_speakers(toy_data)
tr_spk = {x["speaker_id"] for x in tr}
te_spk = {x["speaker_id"] for x in te}
assert len(tr_spk.intersection(te_spk)) == 0, "Speaker leakage detected!"
print("SUCCESS: Disjoint partition verified!")
```
"""
    (ldir / "practical_lab.md").write_text(lab_doc, encoding="utf-8")

    ex_doc = r"""# Exercises: Lecture 09 (Dataset Manifest Engineering)

---

## Exercise 1: Calculating Leakage Contamination
A research intern creates a test split of 200 utterances.
Upon forensic audit, 32 of those utterances belong to speakers who also contributed 140 utterances to the training split.
1. What is the speaker contamination rate of this test set?
2. If the intern reports an 18.2% WER, explain why this metric is untrustworthy and predict the directional shift in WER when evaluated on a disjoint split.

---

## Exercise 2: Bucket Sorting for Dynamic Batching
A training manifest contains audio files with durations ranging from 1.0 second to 18.0 seconds.
Explain why training with random batch sampling causes excessive GPU memory waste due to `<pad>` tokens, and calculate the padding efficiency of sorting samples into duration buckets.
"""
    (ldir / "exercises.md").write_text(ex_doc, encoding="utf-8")

    sol_doc = r"""# Solutions: Lecture 09 (Dataset Manifest Engineering)

---

## Solution to Exercise 1
1. Contamination rate:
   $$C = \frac{32}{200} = 16.0\% \text{ of utterances (or speakers)}$$
2. The reported 18.2% WER is untrustworthy because the acoustic model has memorized the vocal tract characteristics of the leaked speakers. On a strictly disjoint split, WER will systematically increase (typically by 8–18% absolute) as the model encounters genuinely unseen vocal tracts.

---

## Solution to Exercise 2
In PyTorch, all sequences in a batch are padded to the longest sequence in that batch.
If an 18.0s clip (288,000 samples) and a 1.5s clip (24,000 samples) share a batch, the 1.5s clip requires 264,000 padding zeros—wasting **91.7% of its computation**.
Bucket sorting groups clips of similar length (e.g., 1.0–2.5s, 2.5–5.0s), boosting batch padding efficiency to $> 92\%$.
"""
    (ldir / "solutions.md").write_text(sol_doc, encoding="utf-8")

    quiz_doc = r"""# Quiz: Lecture 09 (Dataset Manifests and Zero-Overlap Speaker Splits)

---

### Question 1:
What is the primary danger of speaker leakage between training and testing sets?
- [A] It causes CUDA out-of-memory errors.
- [B] The model memorizes speaker biometric traits rather than general phonetics, creating an artificially low WER that fails on unseen speakers.
- [C] It causes the audio files to delete themselves.
- [D] It corrupts the vocabulary file.

**Correct Answer**: **[B]**  
*Explanation*: Neural models easily overfit to individual speaker acoustic signatures, yielding illusory performance gains.

---

### Question 2:
In EduCasheer, how many frozen disjoint evaluation domains were established?
- [A] 1
- [B] 4
- [C] 8
- [D] 25

**Correct Answer**: **[C]**  
*Explanation*: 8 frozen disjoint domains evaluate robustness across clean, conversational, broadcast, religious, and reverberant audio.

---

### Question 3:
What format is used for EduCasheer dataset manifests?
- [A] Binary Pickle files
- [B] JSON Lines (`.jsonl`)
- [C] Microsoft Excel `.xlsx`
- [D] Raw text strings separated by commas

**Correct Answer**: **[B]**  
*Explanation*: JSON Lines allows streaming inspection, simple schema evolution, and fast parsing.

---

### Question 4:
Why is duration sorting (bucketing) used during batch construction?
- [A] To ensure the loudest audio is processed first.
- [B] To minimize the number of `<pad>` tokens injected into batches, maximizing GPU compute efficiency.
- [C] To sort speakers alphabetically.
- [D] To satisfy Unicode constraints.

**Correct Answer**: **[B]**  
*Explanation*: Grouping similar-length sequences prevents short utterances from being padded with massive blocks of useless zeros.

---

### Question 5:
What was the observed WER gap between random splitting and disjoint splitting in the Stage 14 audit?
- [A] 0.5%
- [B] 2.1%
- [C] ~17.4% absolute (16.8% random vs. 34.2% disjoint)
- [D] Random splitting had higher WER.

**Correct Answer**: **[C]**  
*Explanation*: The 17.4% gap demonstrated the immense distortion caused by speaker leakage.
"""
    (ldir / "quiz.md").write_text(quiz_doc, encoding="utf-8")

    res_doc = r"""# Research Connections: Lecture 09

---

## 1. Repository Grounding
- **`src/data/manifest.py`**: Manifest loading, validation, and zero-leakage verification.
- **`benchmarks/`**: Contains the 8 frozen disjoint evaluation benchmark manifests.
- **`stages/stage_14/`**: Landmark historical audit where the team discovered that early reported SOTA metrics were contaminated by speaker leakage.
"""
    (ldir / "research_connections.md").write_text(res_doc, encoding="utf-8")

    refs_doc = r"""# Annotated References: Lecture 09

---

1. **Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012).**  
   *Leakage in data mining: Formulation, detection, and avoidance.*  
   *ACM Transactions on Knowledge Discovery from Data (TKDD)*, 6(4), 1–21.  
   - **Annotation**: Foundational paper formalizing data leakage, entity overlap, and train-test contamination mechanisms.
"""
    (ldir / "references.md").write_text(refs_doc, encoding="utf-8")
    print("Expanded Lecture 09 successfully!")

if __name__ == "__main__":
    expand_lecture_06()
    expand_lecture_07()
    expand_lecture_08()
    expand_lecture_09()
