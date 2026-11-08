# Chapter 3: What Was Done in This Project (KathAI)

---

## 1. Project Background and Objective

The **KathAI (کتھ)** repository, developed by Musavir Khaliq, is an end-to-end research framework and engineering pipeline dedicated to the Kashmiri language (*Koshur* / کٲشُر). The project spans speech recognition (ASR), speech synthesis (TTS), text normalization, and language model development (SLMs and LLMs).

A central pillar of the KathAI research roadmap was resolving the **Tokenization Crisis**: foundation models (mBERT, LLaMA, Whisper) treated Kashmiri text as an unrepresented foreign script, shattering native orthography into byte-level fragments.

To address this, KathAI executed **Workstream 3: Kashmiri Tokenizer Research & Character Coverage**, establishing custom, linguistically grounded tokenization models across multiple granularities.

---

## 2. The Training Corpora

Tokenizers were trained and benchmarked on verified, curated Kashmiri corpora:

### 2.1 The Wikipedia Pretraining Corpus
- **File Path**: `kashmiri_dataset/processed/text/wikipedia/kashmiri_wikipedia_plain.txt`
- **Volume**:
  - Total Articles: **14,061**
  - Total Sentences: **193,479**
  - Total Words: **1,816,621**
  - Total Characters: **11,829,094**
- **Quality**: Pre-cleaned of HTML tags, metadata headers, and non-Kashmiri foreign articles.

### 2.2 The Expanded 3.4 GB Unified Text Corpus
- **File Path**: `kashmiri_dataset/merged/KASHMIRI-TEXT/kashmiri_text_corpus.jsonl`
- **Volume**: **4,136,878 lines** (~800M+ tokens)
- **Sources**:
  - Classical Kashmiri literature & Sufi poetry (Lal Ded, Sheikh-ul-Alam, Habba Khatoon)
  - Modern web crawls, news syndications, and legal texts
  - Parallel translation corpora (SMUQamar, Omarrran KS-PRET-5M, IndicTrans2 subsets)
  - Instructional reasoning datasets (Aya Kashmiri, Cohere For AI)

---

## 3. The SentencePiece BPE Suite (8K, 16K, 24K, 32K)

KathAI trained four dedicated subword Byte-Pair Encoding (BPE) tokenizers using Google's **SentencePiece** framework to systematically determine the optimal vocabulary frontier for Kashmiri.

### 3.1 Training Configuration & Hyperparameters
The models were trained using the script:
`kashmiri_dataset/scripts/train_and_evaluate_tokenizers.py`

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="kashmiri_dataset/processed/text/wikipedia/kashmiri_wikipedia_plain.txt",
    model_prefix=f"kashmiri_dataset/tokenizers/spm_{vs//1000}k/kashmiri_spm_{vs//1000}k",
    vocab_size=vs,                   # 8000, 16000, 24000, 32000
    model_type="bpe",                # Byte-Pair Encoding backend
    character_coverage=0.9995,       # 99.95% character coverage to prevent OOV
    pad_id=0,                        # [PAD] token
    unk_id=1,                        # [UNK] token
    bos_id=2,                        # [BOS] beginning of sequence
    eos_id=3,                        # [EOS] end of sequence
    user_defined_symbols=[
        "ٲ",  # U+0672: Alef with Wavy Hamza Above (Central vowel /ɨ/)
        "ؠ",  # U+0620: Kashmiri Kashmiri Yeh (Palatal offglide marker)
        "ۆ",  # U+06C6: Kashmiri Oe (Vowel /oː/)
        "ۄ",  # U+06C4: Kashmiri Waw with Ring (Vowel /ɔ/)
        "ێ",  # U+06D0: Kashmiri Yeh with Tail (Vowel /e/)
        "ۍ",  # U+06CD: Kashmiri Yeh with Tail Below
        "ؽ"   # U+063D: Persian Farsi Yeh variant
    ]
)
```

### 3.2 The Critical Role of `user_defined_symbols`
A major breakthrough in KathAI was explicitly passing Kashmiri's unique phonemic characters as `user_defined_symbols`. 

In standard SentencePiece training:
1. Rare Unicode glyphs that occur in the long tail of the corpus risk being assigned to byte-fallbacks or split into isolated multi-byte UTF-8 fragments.
2. By declaring them as **User-Defined Symbols**, SentencePiece guarantees that the tokenizer treats them as **atomic, indivisible base units** that can never be fragmented, decomposed, or treated as unknown tokens.

---

## 4. The Speech ASR Output Representation: CTC Character Tokenizer

For Automatic Speech Recognition (ASR) acoustic modeling (specifically **Track B: Wav2Vec2 / XLS-R-300M** and **Track C: Conformer CTC**), subword tokenizers introduce severe pathologies.

### 4.1 Why Subword BPE Fails in CTC Acoustic Models
1. **Embedding Sparsity on Small Audio Manifests**: The acoustic training dataset (`train.jsonl`) contains 12.35 hours (7,789 utterances). A 16,000-token subword vocabulary contains thousands of tokens that appear only once or twice across the entire audio corpus. Their linear projection weights in the CTC classification head remain under-trained, causing acoustic hallucinations.
2. **Temporal Frame Misalignment**: CTC loss assumes that each 20ms or 40ms acoustic frame aligns with an individual sound unit. Subwords span variable time windows (from 100ms to 1200ms), creating unstable gradient paths during blank token transitions.
3. **Dialectal Variation**: Kashmiri exhibits significant phonetic variation across regions (Srinagar vs Anantnag vs Baramulla). A subword tokenizer commits to a rigid lexical spelling, whereas a character tokenizer allows the acoustic model to transcribe sounds phonetically without catastrophic OOV failure.

### 4.2 The 159-Token Character CTC Vocabulary
KathAI constructed an exact, exhaustive character-level vocabulary using:
`kashmiri_dataset/scripts/build_kashmiri_ctc_tokenizer.py`

- **Total Vocabulary Size**: **159 tokens**
- **Reserved Special Tokens**:
  - `[PAD]`: Index 0 (CTC Blank Token)
  - `[UNK]`: Index 1 (Unknown Grapheme)
  - `|`: Index 2 (Word Boundary Whitespace Delimiter)
- **Grapheme Coverage**:
  - 100% of canonical Kashmiri Perso-Arabic consonants, vowels, and diacritics.
  - Aspirated stop ligatures (*کھ, تھ, چھ, پھ, ڈھ, جھ*).
  - Common Latin loanword characters (`A-Z`, `a-z`) to support acoustic transcription of code-switched technical terms.
  - Digits and standardized punctuation marks.
- **Manifest Location**: `kashmiri_dataset/experiments/native_asr_training/training_configurations/kashmiri_ctc_vocab.json`

---

## 5. Empirical Benchmark Results & The "Fertility Cliff"

The trained tokenizers were rigorously benchmarked against the standard foundation multilingual baseline (`bert-base-multilingual-cased` / mBERT) across an evaluation suite containing native Perso-Arabic Kashmiri, code-switched dialogue, and Devanagari script.

The empirical results recorded in `kashmiri_dataset/quality_reports/kashmiri_tokenizer_benchmark_results.json`:

```
========================================================================================
TOKENIZER BENCHMARK COMPARISON ON NATIVE KASHMIRI EVALUATION CORPUS
========================================================================================
Tokenizer Model            Vocab Size  Fertility (Tokens/Word)  Chars/Token  Efficiency Gain
----------------------------------------------------------------------------------------
mBERT-Multilingual            119,547           2.30                2.42        Baseline
KathAI Kashmiri-BPE-8k          8,000           2.12                2.63        +7.8%
KathAI Kashmiri-BPE-16k        16,000           1.82                3.06        +20.9%
KathAI Kashmiri-BPE-24k        24,000           1.66                3.36        +27.8%
KathAI Kashmiri-BPE-32k        32,000           1.63                3.42        +29.1%
OpenAI GPT-4 / LLaMA-3        128,256          >4.10               ~1.35        Severe Penalty
========================================================================================
```

### 5.1 Scientific Findings

```
Fertility (Tokens / Word) vs Vocabulary Size
 4.5 |   * LLaMA-3 / GPT-4 (>4.10)
 4.0 |
 3.5 |
 3.0 |
 2.5 |                 * mBERT (2.30, 119k vocab)
 2.0 |                         * 8k (2.12)
 1.8 |                                 * 16k (1.82)  <-- Optimal Knee of the Curve
 1.6 |                                         * 24k (1.66)
 1.6 |                                                 * 32k (1.63) [Diminishing Returns]
     +--------------------------------------------------------------------------------
       0               20k            40k             60k            80k            120k
```

1. **The Foundation Model Deficit**: Standard models like LLaMA-3 and GPT-4 produce over 4.10 tokens per Kashmiri word. A 1,000-word Kashmiri text requires over 4,100 tokens, choking context windows and inflating inference costs by over 300%.
2. **The 8k to 16k Leap**: Moving from 8k to 16k vocabulary yields a massive **14.2% drop in fertility** (from 2.12 down to 1.82 tokens per word). The tokenizer successfully merges full syllables, roots, and common nouns.
3. **The 16k to 24k Gain**: Expanding from 16k to 24k further drops fertility from 1.82 to 1.66 (+8.8% gain), capturing frequent verbal conjugations and compound forms.
4. **The 32k Plateau (Diminishing Returns)**: Expanding from 24k to 32k vocabulary yields a minuscule 1.8% drop in fertility (1.66 $\to$ 1.63), while adding 8,000 embedding rows that consume millions of unnecessary parameters.
5. **Project Recommendation**: **The 16K or 24K BPE model is the empirical Sweet Spot** for Kashmiri Small Language Models (SLMs). It delivers ~95% of maximum possible text compression while preserving compact embedding memory.

---

## 6. The Text Normalization & Script Harmonization Engine

Tokenization is only as good as the input text representation. In Kashmiri, identical words are routinely typed using different keyboard layouts (Urdu vs Arabic vs Farsi vs Kashmiri), generating divergent Unicode codepoints.

KathAI engineered a production-grade normalization pipeline in `kashmiri_dataset/scripts/normalize_transcripts.py`:

### 6.1 Unicode Grapheme Canonicalization
The script implements an exact Unicode mapping dictionary (`CHAR_MAP`):

```python
CHAR_MAP = {
    '\u0643': '\u06a9',  # Arabic Kaf (ك)          -> Kashmiri Keheh (ک)
    '\u06aa': '\u06a9',  # Swash Kaf (ڪ)           -> Kashmiri Keheh (ک)
    '\u06b3': '\u06af',  # Gaf variants (ڱ)        -> Kashmiri Gaf (گ)
    '\u0647': '\u06c1',  # Arabic Heh (ه)          -> Kashmiri Heh Goal (ہ)
    '\u0629': '\u06c1',  # Teh Marbuta (ة)         -> Kashmiri Heh Goal (ہ)
    '\u06c3': '\u06c2',  # Teh Marbuta Goal (ۃ)    -> Heh Goal with Hamza (ۂ)
    '\u0649': '\u06cc',  # Alef Maksura (ى)        -> Kashmiri Yeh (ی)
    '\u064a': '\u06cc',  # Arabic Yeh with dots (ي)-> Kashmiri Yeh (ی)
    '\u0640': '',        # Tatweel / Kashida (_)   -> STRIPPED
    '\u200b': '',        # Zero-width space        -> STRIPPED
    '\ufeff': '',        # Byte Order Mark (BOM)   -> STRIPPED
}
```

### 6.2 Zero-Width Non-Joiner (ZWNJ / `\u200c`) Standardization
- **Preserved in Semantics**: ZWNJ is strictly retained where it separates morphemes.
- **Purged of Visual Corruption**: Naive keyboards insert spaces around ZWNJ (e.g., `' \u200c'` or `'\u200c '`), breaking words into disjoint chunks. The KathAI normalizer collapses consecutive ZWNJ marks and strips adjacent whitespace:
  ```python
  text = re.sub(r'[\u200c\u200d]+', '\u200c', text)
  text = text.replace(' \u200c', '\u200c').replace('\u200c ', '\u200c')
  ```

### 6.3 Dual-Mode Normalization: Training vs. Evaluation
KathAI established two distinct normalization pipelines:
1. **`normalize_kashmiri_text(text, strip_diacritics=False)` (Training Mode)**:
   - Preserves all phonemic central vowel marks (`ٲ`, `ۄ`, `ۆ`, `ێ`, `U+065F`, `U+065A`).
   - Essential for acoustic models (speech duration and formants depend on vowel quality) and LLMs (prevents homograph collision).
2. **`normalize_for_wer_evaluation(text)` (ASR Evaluation Mode)**:
   - Strips optional decorative diacritics (Zer, Zabar, Pesh).
   - Removes non-alphanumeric punctuation.
   - Prevents an ASR hypothesis from being penalized with false Word Error Rate (WER) if the model predicted the correct word but omitted an optional Zer diacritic.

### 6.4 Trilingual Code-Switching Detection
Conversational Kashmiri contains frequent code-switching with English (technical terms) and Urdu/Hindi.
The KathAI module `detect_code_switching(text)` calculates the ratio of Perso-Arabic, Latin, and Devanagari characters:
$$\text{Arabic Ratio} = \frac{N_{\text{Arabic}}}{N_{\text{Total}}}, \quad \text{Latin Ratio} = \frac{N_{\text{Latin}}}{N_{\text{Total}}}, \quad \text{Devanagari Ratio} = \frac{N_{\text{Deva}}}{N_{\text{Total}}}$$
Utterances with $>5\%$ Latin or Devanagari characters are automatically tagged as `code_switched` and routed to dedicated evaluation splits.

---

## 7. Artifacts Produced by the Project

All generated tokenizer models, vocabularies, and benchmarks are preserved in the repository:

| Artifact Path | Format | Description |
| :--- | :--- | :--- |
| `kashmiri_dataset/tokenizers/spm_8k/kashmiri_spm_8k.model` | Binary SentencePiece | 8,000-vocabulary Kashmiri BPE model |
| `kashmiri_dataset/tokenizers/spm_16k/kashmiri_spm_16k.model` | Binary SentencePiece | 16,000-vocabulary Kashmiri BPE model |
| `kashmiri_dataset/tokenizers/spm_24k/kashmiri_spm_24k.model` | Binary SentencePiece | 24,000-vocabulary Kashmiri BPE model |
| `kashmiri_dataset/tokenizers/spm_32k/kashmiri_spm_32k.model` | Binary SentencePiece | 32,000-vocabulary Kashmiri BPE model |
| `kashmiri_dataset/quality_reports/kashmiri_tokenizer_benchmark_results.json` | JSON | Full benchmark results & compression comparisons |
| `kashmiri_dataset/experiments/native_asr_training/training_configurations/kashmiri_ctc_vocab.json` | JSON | 159-token character CTC vocabulary for speech |
| `kashmiri_dataset/scripts/normalize_transcripts.py` | Python Script | Production Kashmiri normalization & CS detector |
| `kashmiri_dataset/scripts/train_and_evaluate_tokenizers.py` | Python Script | SentencePiece training and benchmarking engine |
