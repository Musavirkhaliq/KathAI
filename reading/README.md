# Reading Room: Tokenization, Multilingual Pathologies, and Kashmiri NLP Engineering

Welcome to the **KathAI (کتھ) Tokenization & Linguistic Engineering Knowledge Base**. This directory contains exhaustive, research-grade documentation and operational guides on tokenization theory, mathematical foundations, multilingual failure modes, empirical findings from the KathAI project, and a dedicated architectural blueprint for the Kashmiri language (*Koshur*).

---

## 📚 Curriculum & Document Navigation

The knowledge base is structured into four core research treatises and one master compendium:

```text
reading/
├── README.md                                         # This navigation guide & executive summary
├── 01_FOUNDATIONS_AND_MATHEMATICS_OF_TOKENIZATION.md # Theory, Information Theory, BPE, WordPiece, Unigram, BBPE
├── 02_THE_PROBLEMS_WITH_NORMAL_TOKENIZERS.md        # The Multilingual Tokenization Tax, Glitches, Context Collapse
├── 03_WHAT_WAS_DONE_IN_THIS_PROJECT.md              # Exact KathAI experiments, SP 8k-32k, CTC 159-vocab, Normalizer
├── 04_WHAT_WE_SHOULD_DO_FOR_KASHMIRI_AND_WHY.md     # Dedicated Kashmiri blueprint: Vowels, ZWNJ, Grafting, Regex
└── MASTER_TOKENIZATION_HANDBOOK.md                  # Unified, all-in-one encyclopedic reference compendium
```

---

## 📖 Executive Summary of Documents

### 1. [01. Foundations & Mathematics of Tokenization](file:///home/musa/Kashmiri%20Project/reading/01_FOUNDATIONS_AND_MATHEMATICS_OF_TOKENIZATION.md)
* **Scope**: Information theory (Shannon entropy, Zipf's law, Kolmogorov complexity), why neural networks need subwords, and the complete mathematical formulation of the major subword tokenization algorithms.
* **Key Topics**:
  - Byte-Pair Encoding (BPE) merge mechanics and frequency objective.
  - WordPiece likelihood scoring via mutual information maximization.
  - Unigram Language Model (Kudo, 2018) expectation-maximization (EM) optimization and Viterbi pruning.
  - SentencePiece whitespace-preserving meta-symbol architecture (`_` / `U+2581`).
  - Byte-Level BPE (BBPE / Tiktoken) 256-byte initialization and regex pre-tokenization rules.
  - Subword Regularization and BPE-Dropout for stochastic data augmentation.
  - Frontier byte-level / tokenization-free architectures (ByT5, MegaByte, MambaByte).

### 2. [02. The Problems with Normal Tokenizers & The Multilingual Tax](file:///home/musa/Kashmiri%20Project/reading/02_THE_PROBLEMS_WITH_NORMAL_TOKENIZERS.md)
* **Scope**: The systemic structural, economic, and representational discrimination faced by low-resource, non-Latin, and right-to-left (RTL) scripts in standard foundation tokenizers (OpenAI `cl100k_base`, LLaMA-3, Mistral, mBERT).
* **Key Topics**:
  - **The Tokenization Tax**: Mathematical definition of fertility ($\text{Tokens} / \text{Word}$) and the "double penalty."
  - **Economic Discrimination**: Why an API call in Kashmiri or Urdu costs 3x–6x more than the exact same semantic prompt in English.
  - **Context Window Collapse**: How a nominal 32k or 128k context window shrinks to a fraction of its capacity for low-resource languages.
  - **Representational & Latency Penalties**: Quadratic memory footprint, slow sequential autoregressive decoding, and attention capacity wasted on byte stitching.
  - **Morphological Blindness**: Arbitrary subword boundaries shattering inflections, case markers, and clitics.
  - **Unicode & Script Pitfalls**: Normalization discrepancies (NFC vs NFD), combining diacritics detachment, zero-width characters (ZWNJ/ZWJ), and Arabic Presentation Forms.
  - **Glitch Tokens**: "SolidGoldMagikarp" phenomena, unmerged byte strings, and adversarial token behavior.

### 3. [03. What Was Done in This Project (KathAI)](file:///home/musa/Kashmiri%20Project/reading/03_WHAT_WAS_DONE_IN_THIS_PROJECT.md)
* **Scope**: A forensic, line-by-line breakdown of every tokenizer model, training run, dataset, script, and empirical benchmark generated in the KathAI repository.
* **Key Topics**:
  - **Corpus Curation**: Clean Kashmiri Wikipedia dump (14,061 articles, 1.81M words, 11.83M chars) and the 3.4 GB / 4.13M line pretraining corpus (`kashmiri_text_corpus.jsonl`).
  - **SentencePiece BPE Suite**: Training four distinct vocabulary tiers (8k, 16k, 24k, 32k) with `character_coverage=0.9995`.
  - **User-Defined Symbol Registration**: Explicit preservation of Kashmiri-specific Unicode graphemes (`ٲ`, `ؠ`, `ۆ`, `ۄ`, `ێ`, `ۍ`, `ؽ`).
  - **Acoustic CTC Character Tokenizer**: The 159-token character-level output representation for wav2vec2 / XLS-R and Conformer speech models.
  - **Empirical Benchmarks vs Foundation Models**:
    - `mBERT-Multilingual`: Fertility **2.30**, 2.42 chars/token.
    - `Kashmiri-BPE-8k`: Fertility **2.12** (+7.8% compression).
    - `Kashmiri-BPE-16k`: Fertility **1.82** (+20.9% compression).
    - `Kashmiri-BPE-24k`: Fertility **1.66** (+27.8% compression).
    - `Kashmiri-BPE-32k`: Fertility **1.63** (+29.1% compression).
    - Foundation LLMs (Llama-3, GPT-4): Fertility **> 4.10**.
  - **Text Normalization Engine (`normalize_transcripts.py`)**: Script canonicalization, Kashida/Tatweel stripping, ZWNJ standardization, dual-mode evaluation normalizer, and trilingual code-switching detection.

### 4. [04. What Should We Do Specifically About Kashmiri & Why](file:///home/musa/Kashmiri%20Project/reading/04_WHAT_WE_SHOULD_DO_FOR_KASHMIRI_AND_WHY.md)
* **Scope**: The exhaustive linguistic, phonetic, orthographic, and machine learning blueprint tailored specifically to the Kashmiri language (*Koshur*).
* **Key Topics**:
  - **The Phonological Imperative**: 16–18 vowels, central vowels /ɨ/, /ɨː/, /ə/, /əː/, and palatalization [ʲ]; why diacritics are **phonemic** in Kashmiri and cannot be stripped.
  - **Dual-Script & Diaspora Reality**: Perso-Arabic (`kas_Arab`), Devanagari (`kas_Deva`), and Romanized conversational Kashmiri; unified transliteration matrices.
  - **The Zero-Width Non-Joiner (ZWNJ) Rule**: Grammatical affixes, compound nouns, and cursive ligature prevention without spacing.
  - **Kashmiri Pre-tokenization Regex**: Custom regex pattern preserving clitics (*چھُ*, *ہُنٛد*, *سٕنٛز*, *پؠٹھ*), combining marks, and RTL word boundaries.
  - **Vocabulary Grafting & Embedding Expansion**: Grafting 16k Kashmiri tokens into base models (Qwen-2.5-1.5B/7B, Llama-3.2-3B) with Mean Byte Interpolation $\mathbf{w}_{\text{new}} = \frac{1}{|C|} \sum_{c \in C} \mathbf{w}_c$.
  - **Subword Regularization for Dialectal Invariance**: BPE-Dropout to handle regional spelling variations between Srinagar (Yamraz), Anantnag (Maraz), and Baramulla (Kamraz).

### 5. [Master Tokenization Handbook](file:///home/musa/Kashmiri%20Project/reading/MASTER_TOKENIZATION_HANDBOOK.md)
* **Scope**: A single, self-contained, 360-degree reference volume that combines all four treatises into an unabridged textbook. Perfect for offline reading, sharing with collaborators, or publishing as a technical whitepaper.

---

## 🎯 Quick Reference: The Empirical Fertility Cliff

The fundamental finding of this project's tokenizer research is summarized in this comparative benchmark:

| Tokenizer Model | Vocabulary Size | Fertility ($\frac{\text{Tokens}}{\text{Word}}$) | Chars / Token | Context Efficiency vs Llama-3 | Primary Intended Target |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **OpenAI GPT-4 / Tiktoken** | 100,277 | ~4.15 | ~1.35 | Baseline (Severe penalty) | General English / Code |
| **Meta LLaMA-3 / 3.2** | 128,256 | ~4.10 | ~1.38 | Baseline (Severe penalty) | General Multilingual |
| **mBERT-Multilingual** | 119,547 | **2.30** | 2.42 | +43.9% compression | Legacy Multilingual NLP |
| **KathAI Kashmiri-BPE-8k** | 8,000 | **2.12** | 2.63 | +48.3% compression | Compact Speech Decoders |
| **KathAI Kashmiri-BPE-16k** | 16,000 | **1.82** | 3.06 | **+55.6% compression** | **Recommended for SLMs (Qwen/Llama)** |
| **KathAI Kashmiri-BPE-24k** | 24,000 | **1.66** | 3.36 | **+59.5% compression** | High-Capacity Foundation LLMs |
| **KathAI Kashmiri-BPE-32k** | 32,000 | **1.63** | 3.42 | +60.2% compression | Diminishing returns threshold |
| **KathAI Character-CTC** | 159 | 5.28 | 1.00 | N/A (Acoustic Frame Target) | **XLS-R & Conformer ASR** |

---

*Authored for the KathAI Project by Musavir Khaliq and the AI Research Team (November 2026).*
