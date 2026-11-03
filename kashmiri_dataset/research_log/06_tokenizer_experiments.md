# Research Log Entry 06: Kashmiri SentencePiece Tokenizer Experiments & Benchmarks

- **Timestamp**: 2026-11-02T14:35:00+05:30
- **Author/Investigator**: Antigravity AI Pair Programmer & Research Team
- **Phase**: Workstream 3 — Kashmiri Tokenizer Research & Character Coverage

---

## 1. Objective
Design, train, and empirically benchmark dedicated Kashmiri Byte-Pair Encoding (BPE) tokenizers across four vocabulary tiers (8K, 16K, 24K, 32K) against existing multilingual foundation tokenizers (`bert-base-multilingual-cased`, `xlm-roberta-base`, and Llama-style tokenizers). Quantify token fertility, characters per token, and character coverage over native Perso-Arabic orthography (`kas_Arab`).

---

## 2. Methodology & Configuration
- **Training Corpus**: Plain-text clean Kashmiri Wikipedia dump (14,061 articles, 193,479 sentences, 1,816,621 words, 11,829,094 characters).
- **Algorithm**: SentencePiece BPE with `character_coverage=0.9995`.
- **Special Kashmiri User-Defined Symbols**:
  - `U+0672` (ٲ - Alef with Wavy Hamza Above)
  - `U+0620` (ؠ - Kashmiri Kashmiri Yeh)
  - `U+06C6` (ۆ - Kashmiri Oe)
  - `U+06C4` (ۄ - Kashmiri Waw with Ring)
  - `U+06D0` (ێ - Kashmiri Yeh with Tail)
  - `U+06CD` (ۍ - Kashmiri Yeh with Tail Below)
  - `U+063D` (ؽ - Persian Farsi Yeh variant)

---

## 3. Empirical Results

| Tokenizer Model | Vocabulary Size | Token Fertility (Tokens / Word) | Characters per Token | Compression Efficiency vs mBERT |
| :--- | :---: | :---: | :---: | :---: |
| **`mBERT-Multilingual`** | 119,547 | **2.30** | 2.42 | Baseline (High subword fragmentation) |
| **`Kashmiri-BPE-8k`** | 8,000 | **2.12** | 2.63 | +7.8% efficiency |
| **`Kashmiri-BPE-16k`** | 16,000 | **1.82** | 3.06 | +20.9% efficiency |
| **`Kashmiri-BPE-24k`** | 24,000 | **1.66** | 3.36 | +27.8% efficiency |
| **`Kashmiri-BPE-32k`** | 32,000 | **1.63** | 3.42 | **+29.1% efficiency** |

---

## 4. Scientific Findings & Recommendations
1. **The Fertility Cliff**: General multilingual models (mBERT, Llama 3) split Kashmiri words excessively into individual bytes and single diacritics (fertility > 2.30 to 4.10).
2. **Optimal Trade-Off (16K vs 24K)**:
   - Doubling from 8k to 16k yields a dramatic drop in fertility (from 2.12 to 1.82).
   - Moving from 16k to 24k drops fertility further to 1.66.
   - Expanding from 24k to 32k yields marginal diminishing returns (1.66 -> 1.63, a mere 1.8% gain for 8,000 additional embedding rows).
3. **Recommendation for Kashmiri SLM/LLM**: Adopt the **16K or 24K BPE model**. It strikes the ideal balance between embedding table parameter footprint and text compression efficiency.

---

## 5. Artifacts Generated
- `kashmiri_dataset/tokenizers/spm_8k/kashmiri_spm_8k.model`
- `kashmiri_dataset/tokenizers/spm_16k/kashmiri_spm_16k.model`
- `kashmiri_dataset/tokenizers/spm_24k/kashmiri_spm_24k.model`
- `kashmiri_dataset/tokenizers/spm_32k/kashmiri_spm_32k.model`
- `kashmiri_dataset/quality_reports/kashmiri_tokenizer_benchmark_results.json`
