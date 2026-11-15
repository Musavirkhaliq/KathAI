# Character Error Rate (CER) Metric Definitions and Conventions

This document formalizes the mathematical definitions of Character Error Rate (CER) to prevent future reporting divergence.

---

## 1. The Denominator Mismatch Problem

Historical research reports exhibited a confusing disparity in reported CER:
- Stage 20T reported **9.03% CER** for E-03 on `dev_assessment.jsonl`.
- Stage 21A Gate 8 reported **11.19% CER** on the identical hypothesis and reference text.

Gate 1 forensic inspection in Stage 21B revealed that this discrepancy was caused entirely by divergent whitespace and denominator conventions:
1. When whitespace is treated as an active character token (`|`), both reference and hypothesis strings contain spaces. Any deleted, inserted, or substituted space incurs a Levenshtein character penalty. The denominator is the total character length including spaces ($N_{\text{chars}} + N_{\text{spaces}}$).
2. When whitespace is stripped prior to computing edit distance, character distances are lower, and the denominator is strictly non-whitespace graphemes ($N_{\text{chars}}$).
3. The Stage 21A Gate 8 script accidentally computed edit distance on text **with spaces**, but divided by character length **without spaces**, producing an artificially inflated ratio:
   $$\text{CER}_{\text{mismatched}} = \frac{\text{Distance}(\text{text\_with\_spaces})}{\text{Length}(\text{text\_without\_spaces})} \approx \frac{10,250}{91,600} = 11.19\%$$

---

## 2. Standardized Dual CER Definitions

We establish two explicit, standardized formulas that must never be conflated:

### Definition A: Character Error Rate with Space ($\text{CER}_{\text{with\_space}}$)
Whitespace characters (`U+0020` or `|`) are treated as valid graphemes:

$$\text{CER}_{\text{with\_space}} = \frac{S_{\text{char}} + D_{\text{char}} + I_{\text{char}}}{N_{\text{chars}} + N_{\text{spaces}}} \times 100\%$$

On `dev_assessment.jsonl` (E-03 + KenLM baseline):
$$\text{CER}_{\text{with\_space}} = \mathbf{9.01\%}$$

### Definition B: Character Error Rate without Space ($\text{CER}_{\text{no\_space}}$)
Whitespace characters are removed from both reference and hypothesis strings prior to Levenshtein alignment:

$$\text{CER}_{\text{no\_space}} = \frac{S_{\text{grapheme}} + D_{\text{grapheme}} + I_{\text{grapheme}}}{N_{\text{chars}}} \times 100\%$$

On `dev_assessment.jsonl` (E-03 + KenLM baseline):
$$\text{CER}_{\text{no\_space}} = \mathbf{9.97\%}$$

---

## 3. Publication Protocol Rule
All manuscripts must explicitly state which convention is used. In this systematic study, when a single CER figure is cited without qualifier, it refers to $\text{CER}_{\text{with\_space}}$ (9.01%), and tables provide both columns to maintain complete transparency.
