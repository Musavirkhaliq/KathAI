# KathAI (کتھ) — What We Have Done So Far: Comprehensive Audit

**Document**: Research Audit & Completed Milestones  
**Path**: `research/01_WHAT_WE_HAVE_DONE_SO_FAR.md`  
**Repository**: `Musavirkhaliq/KathAI`  
**Date**: November 2026  

---

## 1. Quantitative Overview of Completed Assets

Through rigorous engineering across data collection, cleaning, linguistic normalization, tokenizer training, and acoustic modeling, the repository has transitioned Kashmiri from a fragmented low-resource language into a consolidated multimodal research repository:

| Metric / Modality | Initial Ingestion State | Current Consolidated State | Net Growth / Gain | Verification Artifact |
| :--- | :--- | :--- | :--- | :--- |
| **Speech Audio Volume** | 13.12 hours | **141.61 hours** | **+979% (10.8x)** | `kashmiri_dataset/merged/KASHMIRI-SPEECH/speech_master_manifest.jsonl` (42 MB) |
| **Speech Samples / Utterances** | 9,180 clips | **75,679 verified clips** | **+724% (8.2x)** | Master manifest records with duration, SNR, speaker ID |
| **Unique Native Speakers** | 268 speakers | **639 distinct speakers** | **+138%** | Disjoint speaker partitions with zero train-test leakage |
| **Monolingual Text Records** | 0 records | **4,134,203 lines (3.4 GB)** | **Full Corpus** | `kashmiri_dataset/merged/KASHMIRI-TEXT/kashmiri_text_corpus.jsonl` |
| **Parallel Translation Pairs** | 0 pairs | **344,445 parallel pairs (90 MB)** | **Full Corpus** | `kashmiri_dataset/merged/KASHMIRI-PARALLEL/kashmiri_parallel_corpus.jsonl` |
| **Lexical Dictionary Entries** | 0 entries | **131,231 verified entries (51 MB)** | **Full Lexicon** | `kashmiri_dataset/merged/KASHMIRI-LEXICON/kashmiri_master_lexicon.jsonl` |
| **Synthetic OCR Text-Line Images**| 0 images | **>1,000,000 labeled images** | **Full OCR Set** | `kashmiri_dataset/external/omarrran_koshur_pixel` & `faizaniqbal_koshur-ocr-synth` |
| **Total Disk Storage Managed** | 91.62 GB | **105.80 GB** | **+15.5%** | Immutable raw store (`Data/`) + structured derivatives |

---

## 2. Workstream by Workstream Breakdown

### Workstream 1: Raw Data Ingestion & Provenance Architecture
- **Immutable Raw Data Policy (Decision 001)**: The raw archive directories `/home/musa/Kashmiri Project/Data` (22 multi-part archive volumes) and `kashmiri_dataset/external/` (46 distinct datasets) were designated as strictly read-only.
- **External Dataset Cataloging**: Discovered and downloaded 46 external open-source datasets spanning Hugging Face, OpenSLR, Mendeley Data, and Wikimedia.
  - *Key Audio Corpora*: Bolbosh (11.8 GB, NIT Srinagar), IndicVoices Kashmiri Split (16.9 GB, AI4Bharat), OpenSLR SLR122 (3.38h, 43 sessions), KTTS Single Speaker (3.5h, Mendeley), Programindz Kashmiri Audio (2.53h), XKaab ASR (4.0h).
  - *Key Text Corpora*: Aya Kashmiri SFT (100k+ pairs, 4.12M lines), Kashmiri Wikipedia (14,061 articles, 193k sentences, 1.8M words), KS-PRET-5M (12M tokens), 3.1M Kashmiri LLM Pre-training corpus, Kashmiri News Corpus, Kashmiri Poetry (Shruks of Lal Ded and Sheikh-ul-Alam).
  - *Key Parallel Corpora*: SMUQamar 270K Kashmiri-English, Omarrran 49K Parallel Corpus, 1M Translations corpus.
  - *Key OCR Corpora*: Koshur Pixel (613,078 images across mixed, sentence, paragraph, and page configs; arXiv:2606.23144), Koshur-OCR-Synth (285,864 line images in Nastaliq and Naskh), Koshur-SynthOCR (119k mixed samples), 31k Text Recognition dataset.
  - *Key Dictionaries*: Sir George Grierson's Kashmiri Dictionary, Multilingual Kashmiri Dictionary, Kashmiri Words Restructured.
- **Provenance Registry**: Implemented `build_provenance_registry.py` tracking license classes (Apache-2.0, MIT, CC-BY-4.0, CC-BY-NC-SA, GPL-3.0), source URLs, checksums, and permitted usages.

### Workstream 2: Perso-Arabic Linguistic Analysis & Text Normalization
- **Linguistic Investigation (`KASHMIRI_LINGUISTIC_ANALYSIS.md`)**:
  - Documented Kashmiri's 16–18 vowel inventory, focusing on phonemic central unrounded vowels: /ɨ/, /ɨː/ (`U+0672` ٲ and `U+065F`) and /ə/, /əː/.
  - Documented phonemic palatalization offglides ([ʲ]) marked by `U+065A` (Arabic Vowel Sign Small V Above) or Kashmiri Yeh ligatures.
- **Dual-Mode Normalization Engine (`normalize_transcripts.py`, Decision 003)**:
  - *Acoustic / Pronunciation Normalizer*: Preserves contrastive central vowel diacritics and palatalization marks necessary for acoustic modeling, phonetic alignments, and TTS synthesis. Canonicalizes precomposed glyphs (`U+0672` instead of broken combining sequences).
  - *Evaluation Normalizer*: Collapses optional decorative Arabic diacritics (Zer, Zabar, Pesh), unifies Unicode character variants (Arabic Kaf `U+0643` -> Keheh `U+06A9`; Arabic Heh `U+0647` -> Heh Goal `U+06C1` / Heh Doachashmee `U+06BE` for aspirated stops; Farsi Yeh `U+06CC` / Yeh Barree `U+06D2`), normalizes punctuation, and standardizes whitespace for standardized Levenshtein WER/CER scoring.
- **Code-Switching Detection**: Built regex-based lexical analyzers calculating English and Urdu code-mixing ratios per utterance.

### Workstream 3: Audio Canonicalization & Quality Auditing
- **Standardized Pipeline (`validate_audio.py`, `quality_filter.py`)**:
  - Re-encoded all heterogeneous audio (WebM Opus, MP3, 8kHz/44.1kHz/48kHz, multi-channel) into canonical **16,000 Hz, 16-bit Mono Linear PCM WAV** with clean RIFF headers (Decision 004).
  - Filtered container mismatches that previously caused 3–5x DataLoader overhead.
- **Acoustic Quality Metrics**:
  - Calculated Signal-to-Noise Ratio (SNR) via WADA-SNR and energy-based estimators.
  - Detected digital clipping, samples with duration < 0.5s or > 30s, and high ambient reverberation.
  - Enforced **Quality-Flagging Policy (Decision 005)**: Rather than silently discarding noisy or code-switched audio, samples were annotated with descriptive flags (`low_snr`, `is_clipped`, `is_code_switched`) in the master manifest, enabling task-specific dynamic filtering (e.g. strict studio quality for TTS vs noisy diversity for robust ASR).

### Workstream 4: Zero-Leakage Speaker-Independent Splitting
- **The Contamination Problem Solved (Decision 002)**: Random shuffling in prior Kashmiri datasets leaked speaker identities and acoustic environments between train and test sets, artificially deflating reported WER by 15–30%.
- **Disjoint Partitioning Algorithm (`build_speaker_independent_splits.py`)**:
  - Grouped all 75,679 utterances by speaker identity across all 639 speakers.
  - Executed greedy bin-packing to produce mutually exclusive splits:
    - **Train (~80%)**: 60,543 utterances (~113.3 hours, 511 speakers)
    - **Validation (~10%)**: 7,568 utterances (~14.2 hours, 64 speakers)
    - **Test (~10%)**: 7,568 utterances (~14.1 hours, 64 speakers)
    - Strict mathematical guarantee: $\text{Speakers}_{\text{train}} \cap \text{Speakers}_{\text{val}} = \emptyset$ and $\text{Speakers}_{\text{train}} \cap \text{Speakers}_{\text{test}} = \emptyset$.
- **Diagnostic Benchmark Evaluation Slices**:
  - `test_clean.jsonl`: High SNR (>20 dB), unclipped, zero code-switching, canonical Kashmiri pronunciation.
  - `test_noisy.jsonl`: Real-world field recordings with background noise (SNR < 15 dB).
  - `test_code_switched.jsonl`: Utterances containing English and Urdu loanwords and phrases.
  - `test_speaker_unseen.jsonl`: Dedicated subset from previously completely unseen regional dialects (Kamraz / Maraz).

### Workstream 5: Tokenizer Engineering & Fertility Optimization
- **The Tokenizer Pathology**: Standard multilingual tokenizers (mBERT, Llama-3, Mistral) lack Kashmiri subwords, fracturing words into individual bytes. On our evaluation benchmark:
  - `mBERT-Multilingual`: 168 tokens, 2.30 fertility, multiple `[UNK]` tokens.
  - `Llama-3-Tokenizer`: >4.10 tokens per word.
- **Custom Tokenizer Training (`train_and_evaluate_tokenizers.py`)**:
  - Trained 4 SentencePiece Byte-Pair Encoding (BPE) tokenizers on the cleaned Kashmiri corpus:
    - **Kashmiri-BPE-8k**: Fertility **2.12**, 2.63 chars/token.
    - **Kashmiri-BPE-16k**: Fertility **1.82**, 3.06 chars/token.
    - **Kashmiri-BPE-24k**: Fertility **1.66**, 3.36 chars/token.
    - **Kashmiri-BPE-32k**: Fertility **1.63**, 3.42 chars/token, **0 UNK tokens**.
  - Slashed token inflation by **>60%**, dramatically expanding effective context length and speeding up LLM generation.
- **ASR Character CTC Vocabulary (`build_kashmiri_ctc_tokenizer.py`, `build_hf_ctc_tokenizer.py`)**:
  - Built a 161-token character vocabulary covering all Perso-Arabic Kashmiri consonants, central vowels (`U+0672`, `U+065F`), palatalization signs (`U+065A`), English code-switched Latin characters, blank token `[PAD]`, and unknown token `[UNK]`.

### Workstream 6: Acoustic Baseline Benchmarking & Controlled Experiment B-01
- **ASR SOTA Landscape Audit (`MODEL_BENCHMARKS.md`, `09_sota_asr_landscape_and_benchmarks.md`)**:
  - Benchmarked existing external models: `ai4bharat/indicconformer_stt_ks_hybrid_ctc_rnnt_large` (21.4% clean WER), `facebook/mms-1b-all` (28.6% WER), `openai/whisper-small` community fine-tunes (26.8% WER).
- **Controlled Experiment B-01 Execution (`train_b01_xlsr.py`, `12_b01_experiment_execution_complete.md`)**:
  - Model: `facebook/wav2vec2-xls-r-300m` with linear CTC classification head (161 Kashmiri classes).
  - Hardware: NVIDIA DGX Spark GB10.
  - Dynamic CTC Blank-Posterior Optimization: Confirmed smooth training dynamics without NaN or gradient divergence.
  - Empirical Loss Monotonicity:
    - Training loss dropped smoothly from **14.21** down to **3.42**.
    - Validation loss dropped monotonically from **4.34** down to **3.64**.
    - Gradient norms stabilized between **0.87 and 1.97**.
  - Hardware Efficiency: Peak VRAM **14.53 GB**, Real-Time Factor (RTF) **0.024** (~41x faster than real-time playback).
- **Production Full-Schedule Trainer (`train_b01_full_schedule.py`)**:
  - Built production script supporting PyTorch DDP/AMP fp16, bucketing DataLoader (by audio length), zero-infinity CTC loss handling, cosine annealing scheduler with warmup, and dynamic evaluation checkpointing.

---

## 3. Summary of Core Repository Scripts & Tools

| Script | Purpose | Status |
| :--- | :--- | :--- |
| `audit_master_inventory.py` | Scans and computes disk sizes, formats, and checksums of all 22 parts and 46 external sets | Verified |
| `build_provenance_registry.py` | Tracks dataset licensing, citations, source repositories, and redistribution terms | Verified |
| `normalize_transcripts.py` | Production text normalizer with dual-mode acoustic and evaluation normalization | Verified |
| `validate_audio.py` & `quality_filter.py` | Canonicalizes audio to 16kHz mono PCM, computes SNR, VAD, and flags clipping | Verified |
| `build_speaker_independent_splits.py` | Generates 0-speaker-leakage train/val/test splits and 4 benchmark test slices | Verified |
| `train_and_evaluate_tokenizers.py` | Trains SentencePiece BPE models (8k–32k) and runs fertility evaluation | Verified |
| `build_hf_ctc_tokenizer.py` | Compiles 161-token Hugging Face Wav2Vec2CTCTokenizer JSON files | Verified |
| `train_b01_xlsr.py` | Pilot controlled training script for XLS-R-300M CTC on DGX GB10 | Executed |
| `train_b01_full_schedule.py` | Production-grade 50-epoch ASR training script with length bucketing and fp16 | Ready to Run |
| `run_whisper_benchmark.py` | Zero-shot & fine-tune evaluation pipeline for OpenAI Whisper on Kashmiri splits | Ready to Run |
| `classify_text_corpus.py` | Partitions 3.4GB text into SFT, pretraining, Wikipedia, parallel, and poetry | Verified |

---
*Proceed to [02_WHAT_WE_NEED_TO_DO_STEP_BY_STEP.md](file:///home/musa/Kashmiri%20Project/research/02_WHAT_WE_NEED_TO_DO_STEP_BY_STEP.md) for the sequential execution plan and next actions.*
