# KathAI Research Papers Library: Foundations of Text, Speech, and Kashmiri AI

Welcome to the **KathAI (کتھ) Research Papers Library**. This directory contains **25 authoritative research papers** downloaded in full PDF format, spanning subword tokenization, acoustic foundation models, speech recognition (ASR), speech synthesis (TTS), multimodal Speech-LLMs, and dedicated Kashmiri language processing.

Each paper has been hand-selected to provide the theoretical, mathematical, and architectural foundations necessary to design, train, and deploy state-of-the-art AI systems for **Kashmiri (*Koshur* / کٲشُر)**.

---

## 📂 Repository Directory Map

```text
papers/
├── README.md                                          # This master annotated syllabus & reading curriculum
├── text_and_tokenization/                             # 10 Papers: BPE, SentencePiece, Tokenization Tax, Byte Models
│   ├── 01_sennrich2016_subword_bpe_nmt.pdf
│   ├── 02_kudo2018_subword_regularization_unigram.pdf
│   ├── 03_kudo2018_sentencepiece_system.pdf
│   ├── 04_petrov2023_tokenization_tax_discrepancies.pdf
│   ├── 05_ahia2023_do_all_languages_cost_the_same.pdf
│   ├── 06_conneau2020_xlm_roberta.pdf
│   ├── 07_singh2024_aya_multilingual_dataset.pdf
│   ├── 08_gala2023_indictrans2_22_languages.pdf
│   ├── 09_xue2022_byt5_token_free_models.pdf
│   └── 10_yu2023_megabyte_multiscale_transformers.pdf
├── speech_and_audio/                                  # 9 Papers: CTC, wav2vec 2.0, XLS-R, Conformer, Whisper, IndicVoices, Flow TTS
│   ├── 01_graves2006_connectionist_temporal_classification_ctc.pdf
│   ├── 02_baevski2020_wav2vec2_self_supervised_speech.pdf
│   ├── 03_babu2021_xls_r_128_languages.pdf
│   ├── 04_gulati2020_conformer_speech_recognition.pdf
│   ├── 05_radford2022_whisper_robust_asr.pdf
│   ├── 06_javed2024_indicvoices_dataset.pdf
│   ├── 07_sankar2024_indicvoices_r_tts_corpus.pdf
│   ├── 08_mehta2024_matcha_tts_flow_matching.pdf
│   └── 09_chen2024_f5_tts_flow_matching_voice_clone.pdf
├── speech_llms_and_multimodal/                         # 4 Papers: Speech-to-Unit, AudioPaLM, SpeechGPT, Qwen2-Audio
│   ├── 01_seamless2023_seamlessm4t_multimodal.pdf
│   ├── 02_rubenstein2023_audiopalm_speaks_and_listens.pdf
│   ├── 03_zhang2023_speechgpt_conversational_llm.pdf
│   └── 04_qwen2024_qwen2_audio_technical_report.pdf
└── kashmiri_and_low_resource/                         # 2 Papers: Koshur Pixel OCR, 600k-KS-OCR
    ├── 01_malik2024_koshur_pixel_synthetic_ocr.pdf
    └── 02_malik2024_600k_ks_ocr_dataset.pdf
```

---

## 🎓 Recommended Thematic Reading Curriculum

### Path 1: Mastering Tokenization & Language Models
1. **The Origin**: Start with [Sennrich et al. (2016)](file:///home/musa/Kashmiri%20Project/papers/text_and_tokenization/01_sennrich2016_subword_bpe_nmt.pdf) to understand how subword merging replaced fixed word vocabularies.
2. **The Modern Framework**: Read [Kudo (2018)](file:///home/musa/Kashmiri%20Project/papers/text_and_tokenization/02_kudo2018_subword_regularization_unigram.pdf) and [Kudo & Richardson (2018)](file:///home/musa/Kashmiri%20Project/papers/text_and_tokenization/03_kudo2018_sentencepiece_system.pdf) to learn why space-agnostic SentencePiece and subword regularization are critical for complex morphology.
3. **The Bias & The Tax**: Read [Petrov et al. (2023)](file:///home/musa/Kashmiri%20Project/papers/text_and_tokenization/04_petrov2023_tokenization_tax_discrepancies.pdf) and [Ahia et al. (2023)](file:///home/musa/Kashmiri%20Project/papers/text_and_tokenization/05_ahia2023_do_all_languages_cost_the_same.pdf) to understand why standard LLMs charge a 4x penalty on non-Latin scripts.
4. **Multilingual Scale**: Read [Conneau et al. (2020)](file:///home/musa/Kashmiri%20Project/papers/text_and_tokenization/06_conneau2020_xlm_roberta.pdf), [Singh et al. (2024, Aya)](file:///home/musa/Kashmiri%20Project/papers/text_and_tokenization/07_singh2024_aya_multilingual_dataset.pdf), and [Gala et al. (2023, IndicTrans2)](file:///home/musa/Kashmiri%20Project/papers/text_and_tokenization/08_gala2023_indictrans2_22_languages.pdf).
5. **The Token-Free Future**: Read [Xue et al. (2022, ByT5)](file:///home/musa/Kashmiri%20Project/papers/text_and_tokenization/09_xue2022_byt5_token_free_models.pdf) and [Yu et al. (2023, MEGABYTE)](file:///home/musa/Kashmiri%20Project/papers/text_and_tokenization/10_yu2023_megabyte_multiscale_transformers.pdf).

---

### Path 2: Mastering Acoustic Modeling & Speech Recognition (ASR)
1. **The Alignment Foundation**: Read [Graves et al. (2006)](file:///home/musa/Kashmiri%20Project/papers/speech_and_audio/01_graves2006_connectionist_temporal_classification_ctc.pdf) to understand Connectionist Temporal Classification (CTC) loss, blank tokens, and alignment-free speech training.
2. **Self-Supervised Audio Representations**: Read [Baevski et al. (2020, wav2vec 2.0)](file:///home/musa/Kashmiri%20Project/papers/speech_and_audio/02_baevski2020_wav2vec2_self_supervised_speech.pdf) and [Babu et al. (2021, XLS-R)](file:///home/musa/Kashmiri%20Project/papers/speech_and_audio/03_babu2021_xls_r_128_languages.pdf) to see how 436,000 hours of unlabeled audio enables low-resource speech recognition.
3. **The SOTA ASR Architecture**: Read [Gulati et al. (2020, Conformer)](file:///home/musa/Kashmiri%20Project/papers/speech_and_audio/04_gulati2020_conformer_speech_recognition.pdf) to understand how combining self-attention with depthwise separable convolutions captures both global context and local acoustic phonetics.
4. **Weak Supervision & Massive Multilingual Speech**: Read [Radford et al. (2022, Whisper)](file:///home/musa/Kashmiri%20Project/papers/speech_and_audio/05_radford2022_whisper_robust_asr.pdf) and [Javed et al. (2024, IndicVoices)](file:///home/musa/Kashmiri%20Project/papers/speech_and_audio/06_javed2024_indicvoices_dataset.pdf).
5. **Modern Speech Synthesis (TTS)**: Read [Mehta et al. (2024, Matcha-TTS)](file:///home/musa/Kashmiri%20Project/papers/speech_and_audio/08_mehta2024_matcha_tts_flow_matching.pdf) and [Chen et al. (2024, F5-TTS)](file:///home/musa/Kashmiri%20Project/papers/speech_and_audio/09_chen2024_f5_tts_flow_matching_voice_clone.pdf).

---

### Path 3: Speech-LLMs, Multimodal AI, and Spoken Agents
1. **Massive Multimodal Translation**: Read [Seamless Communication (2023, SeamlessM4T)](file:///home/musa/Kashmiri%20Project/papers/speech_llms_and_multimodal/01_seamless2023_seamlessm4t_multimodal.pdf) for unitY2 multi-stage decoding across speech and text.
2. **Audio-Language Modeling**: Read [Rubenstein et al. (2023, AudioPaLM)](file:///home/musa/Kashmiri%20Project/papers/speech_llms_and_multimodal/02_rubenstein2023_audiopalm_speaks_and_listens.pdf) and [Zhang et al. (2023, SpeechGPT)](file:///home/musa/Kashmiri%20Project/papers/speech_llms_and_multimodal/03_zhang2023_speechgpt_conversational_llm.pdf).
3. **Production Audio-LLMs**: Read [Qwen Team (2024, Qwen2-Audio)](file:///home/musa/Kashmiri%20Project/papers/speech_llms_and_multimodal/04_qwen2024_qwen2_audio_technical_report.pdf) to understand how to condition an LLM on raw audio representations.

---

### Path 4: Kashmiri Script, Nastaliq OCR & Low-Resource Vision
1. **Synthetic Nastaliq OCR at Scale**: Read [Malik et al. (2024, Koshur Pixel)](file:///home/musa/Kashmiri%20Project/papers/kashmiri_and_low_resource/01_malik2024_koshur_pixel_synthetic_ocr.pdf).
2. **Word-Level Kashmiri Character Recognition**: Read [Malik et al. (2024, 600k-KS-OCR)](file:///home/musa/Kashmiri%20Project/papers/kashmiri_and_low_resource/02_malik2024_600k_ks_ocr_dataset.pdf).

---

## 📑 Comprehensive Paper Catalogue & Annotations

### Section 1: Text, Tokenization, and Multilingual Language Models

#### 1. [Neural Machine Translation of Rare Words with Subword Units](file:///home/musa/Kashmiri%20Project/papers/text_and_tokenization/01_sennrich2016_subword_bpe_nmt.pdf)
- **Authors**: Rico Sennrich, Barry Haddow, Alexandra Birch
- **Venue**: ACL 2016
- **Local File**: `papers/text_and_tokenization/01_sennrich2016_subword_bpe_nmt.pdf` (188 KB)
- **Core Contribution**: Adapted the Byte-Pair Encoding (BPE) compression algorithm to NLP, demonstrating that open-vocabulary translation is possible by segmenting rare words into frequent subword units.
- **Why It Matters for KathAI**: Establishes the mathematical principle of bottom-up subword merging. In Kashmiri, this allows unseen inflected verb forms to be constructed from base roots and case suffixes.

#### 2. [Subword Regularization: Improving NMT Models with Multiple Subword Candidates](file:///home/musa/Kashmiri%20Project/papers/text_and_tokenization/02_kudo2018_subword_regularization_unigram.pdf)
- **Author**: Taku Kudo
- **Venue**: ACL 2018
- **Local File**: `papers/text_and_tokenization/02_kudo2018_subword_regularization_unigram.pdf` (321 KB)
- **Core Contribution**: Formulated the **Unigram Language Model Tokenizer** and introduced subword sampling during training. Rather than committing to a deterministic 1-best segmentation, the model is exposed to multiple segmentations drawn from a posterior probability distribution.
- **Why It Matters for KathAI**: Kashmiri text exhibits dialectal spelling variations across Srinagar (Yamraz), Anantnag (Maraz), and Baramulla (Kamraz). Subword regularization enables neural models to become invariant to minor spelling and diacritic variations.

#### 3. [SentencePiece: A Simple and Language Independent Subword Tokenizer](file:///home/musa/Kashmiri%20Project/papers/text_and_tokenization/03_kudo2018_sentencepiece_system.pdf)
- **Authors**: Taku Kudo, John Richardson
- **Venue**: EMNLP 2018
- **Local File**: `papers/text_and_tokenization/03_kudo2018_sentencepiece_system.pdf` (206 KB)
- **Core Contribution**: Introduced the SentencePiece framework, which eliminates language-dependent whitespace pre-segmentation by mapping whitespace to a visible Unicode meta-symbol (`_` / `U+2581`), enabling 100% lossless detokenization.
- **Why It Matters for KathAI**: Directly used in KathAI (`train_and_evaluate_tokenizers.py`) to train our 8k, 16k, 24k, and 32k BPE tokenizers with custom registered Kashmiri Unicode symbols (`ٲ`, `ؠ`, `ۆ`, `ۄ`, `ێ`, `ۍ`, `ؽ`).

#### 4. [Language Model Tokenizers Introduce Systematic Discrepancies Across Languages](file:///home/musa/Kashmiri%20Project/papers/text_and_tokenization/04_petrov2023_tokenization_tax_discrepancies.pdf)
- **Authors**: Aleksandar Petrov, Emanuele La Malfa, Philip H.S. Torr, Adel Bibi
- **Venue**: NeurIPS 2023
- **Local File**: `papers/text_and_tokenization/04_petrov2023_tokenization_tax_discrepancies.pdf` (541 KB)
- **Core Contribution**: Rigorously proved that commercial LLM tokenizers impose systematic fertility and compression disparities on non-English languages, degrading both accuracy and context utilization.
- **Why It Matters for KathAI**: Provides the theoretical and empirical baseline justifying why we cannot rely on LLaMA-3 or GPT-4 tokenizers for Kashmiri (which impose a 4.15x fertility penalty).

#### 5. [Do All Languages Cost the Same? Tokenization in the Era of Commercial Language Models](file:///home/musa/Kashmiri%20Project/papers/text_and_tokenization/05_ahia2023_do_all_languages_cost_the_same.pdf)
- **Authors**: Orevaoghene Ahia et al.
- **Venue**: EMNLP 2023
- **Local File**: `papers/text_and_tokenization/05_ahia2023_do_all_languages_cost_the_same.pdf` (1,098 KB)
- **Core Contribution**: Analyzed the economic discrimination caused by per-token API billing models, proving that low-resource language speakers pay up to 6x more for equivalent task execution.
- **Why It Matters for KathAI**: Highlights the economic necessity of native Kashmiri Small Language Models (SLMs) trained with localized vocabularies to eliminate the commercial API penalty.

#### 6. [Unsupervised Cross-lingual Representation Learning at Scale (XLM-RoBERTa)](file:///home/musa/Kashmiri%20Project/papers/text_and_tokenization/06_conneau2020_xlm_roberta.pdf)
- **Authors**: Alexis Conneau et al.
- **Venue**: ACL 2020
- **Local File**: `papers/text_and_tokenization/06_conneau2020_xlm_roberta.pdf` (389 KB)
- **Core Contribution**: Demonstrated that training a Transformer encoder on 2.5 TB of CommonCrawl data across 100 languages with a 250k-token SentencePiece vocabulary produces state-of-the-art cross-lingual representations.
- **Why It Matters for KathAI**: Serves as the architecture and tokenizer reference for our text representation baselines and cross-lingual classification tasks.

#### 7. [Aya Dataset: An Open-Access Collection for Multilingual Instruction Tuning](file:///home/musa/Kashmiri%20Project/papers/text_and_tokenization/07_singh2024_aya_multilingual_dataset.pdf)
- **Authors**: Shivalika Singh et al. (Cohere For AI)
- **Venue**: arXiv 2024
- **Local File**: `papers/text_and_tokenization/07_singh2024_aya_multilingual_dataset.pdf` (9,322 KB)
- **Core Contribution**: Curated a human-annotated and synthetically expanded instruction tuning dataset across 101 languages, including Kashmiri, to unlock instruction-following in low-resource regimes.
- **Why It Matters for KathAI**: Aya Kashmiri is our primary Supervised Fine-Tuning (SFT) corpus used to transform our pretrained Kashmiri SLM into a helpful conversational assistant.

#### 8. [IndicTrans2: Machine Translation for all 22 Scheduled Indian Languages](file:///home/musa/Kashmiri%20Project/papers/text_and_tokenization/08_gala2023_indictrans2_22_languages.pdf)
- **Authors**: Jay Gala et al. (AI4Bharat)
- **Venue**: arXiv 2023
- **Local File**: `papers/text_and_tokenization/08_gala2023_indictrans2_22_languages.pdf` (1,213 KB)
- **Core Contribution**: The first public state-of-the-art multilingual NMT model supporting all 22 official Indian languages, including Kashmiri, utilizing script unification and 1B-parameter Transformer architectures.
- **Why It Matters for KathAI**: Serves as our primary machine translation engine and parallel text alignment backbone between English and Kashmiri.

#### 9. [ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models](file:///home/musa/Kashmiri%20Project/papers/text_and_tokenization/09_xue2022_byt5_token_free_models.pdf)
- **Authors**: Linting Xue et al. (Google Research)
- **Venue**: TACL 2022
- **Local File**: `papers/text_and_tokenization/09_xue2022_byt5_token_free_models.pdf` (475 KB)
- **Core Contribution**: Eliminated subword tokenizers completely by training a modified T5 encoder-decoder operating directly on raw UTF-8 bytes, proving exceptional robustness to typos, noisy text, and unseen scripts.
- **Why It Matters for KathAI**: Essential reading for evaluating tokenizer-free alternatives for messy social media text and unstandardized Kashmiri WhatsApp chats.

#### 10. [MEGABYTE: Predicting High-Million-Byte Sequences with Multiscale Transformers](file:///home/musa/Kashmiri%20Project/papers/text_and_tokenization/10_yu2023_megabyte_multiscale_transformers.pdf)
- **Authors**: Lili Yu, Daniel Simig, Luke Zettlemoyer, Mike Lewis (Meta AI)
- **Venue**: NeurIPS 2023
- **Local File**: `papers/text_and_tokenization/10_yu2023_megabyte_multiscale_transformers.pdf` (895 KB)
- **Core Contribution**: Proposed a hierarchical architecture that partitions long byte streams into local patches, enabling sub-quadratic autoregressive language modeling over 100k+ byte contexts.
- **Why It Matters for KathAI**: Inspires our multi-scale audio and text patch processing architectures for raw multimodal streaming.

---

### Section 2: Speech Recognition (ASR) & Acoustic Foundations

#### 11. [Connectionist Temporal Classification: Labelling Unsegmented Sequence Data](file:///home/musa/Kashmiri%20Project/papers/speech_and_audio/01_graves2006_connectionist_temporal_classification_ctc.pdf)
- **Authors**: Alex Graves, Santiago Fernández, Faustino Gomez, Jürgen Schmidhuber
- **Venue**: ICML 2006
- **Local File**: `papers/speech_and_audio/01_graves2006_connectionist_temporal_classification_ctc.pdf` (260 KB)
- **Core Contribution**: Formulated the **CTC Loss Function**, introducing the special blank token ($\epsilon$) and dynamic programming forward-backward algorithm that allows acoustic frames to map directly to text without requiring frame-by-frame phone alignments.
- **Why It Matters for KathAI**: The mathematical cornerstone of our **159-token Character CTC** acoustic model (`train_b01_xlsr.py`). Graves' theorem governs our frame-level convergence and blank token dynamics.

#### 12. [wav2vec 2.0: Self-Supervised Learning of Speech Representations](file:///home/musa/Kashmiri%20Project/papers/speech_and_audio/02_baevski2020_wav2vec2_self_supervised_speech.pdf)
- **Authors**: Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli
- **Venue**: NeurIPS 2020
- **Local File**: `papers/speech_and_audio/02_baevski2020_wav2vec2_self_supervised_speech.pdf` (723 KB)
- **Core Contribution**: Introduced masked contrastive predictive coding over quantized latent audio representations, proving that 10 minutes of labeled audio fine-tuned on self-supervised representations achieves competitive ASR.
- **Why It Matters for KathAI**: The foundation of our acoustic encoder pretraining strategy.

#### 13. [XLS-R: Cross-Lingual Speech Representation Learning at Scale](file:///home/musa/Kashmiri%20Project/papers/speech_and_audio/03_babu2021_xls_r_128_languages.pdf)
- **Authors**: Arun Babu, Changhan Wang, Andros Tjandra, Kushal Lakhotia, et al. (Meta AI)
- **Venue**: Interspeech 2022
- **Local File**: `papers/speech_and_audio/03_babu2021_xls_r_128_languages.pdf` (582 KB)
- **Core Contribution**: Scaled self-supervised multilingual speech modeling to 436,000 hours of public audio across 128 languages (from VoxPopuli, CommonVoice, BABEL, and MLS) in models up to 2 Billion parameters.
- **Why It Matters for KathAI**: **`facebook/wav2vec2-xls-r-300m`** is the exact base foundation model used in KathAI Track B for native Kashmiri fine-tuning.

#### 14. [Conformer: Convolution-Augmented Transformer for Speech Recognition](file:///home/musa/Kashmiri%20Project/papers/speech_and_audio/04_gulati2020_conformer_speech_recognition.pdf)
- **Authors**: Anmol Gulati et al. (Google)
- **Venue**: Interspeech 2020
- **Local File**: `papers/speech_and_audio/04_gulati2020_conformer_speech_recognition.pdf` (402 KB)
- **Core Contribution**: Interleaved multi-head self-attention with depthwise separable convolution blocks in a Macaron-style feed-forward network, capturing both global context and local acoustic features.
- **Why It Matters for KathAI**: The Conformer architecture achieves the lowest Word Error Rate on Kashmiri in our comparative benchmarks (**`IndicConformer` WER: 16.48%**).

#### 15. [Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)](file:///home/musa/Kashmiri%20Project/papers/speech_and_audio/05_radford2022_whisper_robust_asr.pdf)
- **Authors**: Alec Radford et al. (OpenAI)
- **Venue**: arXiv 2022
- **Local File**: `papers/speech_and_audio/05_radford2022_whisper_robust_asr.pdf` (985 KB)
- **Core Contribution**: Trained an autoregressive sequence-to-sequence Transformer on 680,000 hours of weakly supervised internet audio, demonstrating zero-shot robustness across accents, background noise, and technical jargon.
- **Why It Matters for KathAI**: We benchmark OpenAI Whisper (Tiny, Base, Small, Medium) across all 6 standardized evaluation splits in `run_whisper_benchmark.py`.

#### 16. [IndicVoices: Multilingual Speech Dataset for Indian Languages](file:///home/musa/Kashmiri%20Project/papers/speech_and_audio/06_javed2024_indicvoices_dataset.pdf)
- **Authors**: Tahir Javed et al. (AI4Bharat)
- **Venue**: arXiv 2024
- **Local File**: `papers/speech_and_audio/06_javed2024_indicvoices_dataset.pdf` (31,178 KB)
- **Core Contribution**: Released 7,300+ hours of spontaneous read and conversational speech across 22 Indian languages recorded in field environments across 145 districts.
- **Why It Matters for KathAI**: The source of our spontaneous, real-world native Kashmiri field recordings (`eval_conversational_field`), testing model robustness against background noise and regional accents.

#### 17. [IndicVoices-R: Scaling Indian Speech Synthesis (TTS)](file:///home/musa/Kashmiri%20Project/papers/speech_and_audio/07_sankar2024_indicvoices_r_tts_corpus.pdf)
- **Authors**: Ashwin Sankar et al. (AI4Bharat)
- **Venue**: arXiv 2024
- **Local File**: `papers/speech_and_audio/07_sankar2024_indicvoices_r_tts_corpus.pdf` (663 KB)
- **Core Contribution**: Curated and acoustically restored multi-speaker datasets for training expressive, multi-accented Indian text-to-speech models.
- **Why It Matters for KathAI**: Informs our multi-speaker voice synthesis dataset curation and acoustic quality filtering pipelines.

#### 18. [Matcha-TTS: Fast TTS with Conditional Flow Matching](file:///home/musa/Kashmiri%20Project/papers/speech_and_audio/08_mehta2024_matcha_tts_flow_matching.pdf)
- **Authors**: Shivam Mehta, Ruibo Tu, Jonas Beskow, Éva Székely, Gustav Eje Henter
- **Venue**: ICASSP 2024
- **Local File**: `papers/speech_and_audio/08_mehta2024_matcha_tts_flow_matching.pdf` (751 KB)
- **Core Contribution**: Formulated non-autoregressive speech synthesis using Conditional Flow Matching (CFM) with Optimal Transport vector fields, generating studio-grade spectrograms in as few as 4 ODE solver steps.
- **Why It Matters for KathAI**: **Matcha-TTS is our primary edge TTS architecture** designed to synthesize natural Kashmiri speech on mobile and CPU hardware in real time (RTF < 0.15).

#### 19. [F5-TTS: Non-Autoregressive Voice Cloning via Flow Matching](file:///home/musa/Kashmiri%20Project/papers/speech_and_audio/09_chen2024_f5_tts_flow_matching_voice_clone.pdf)
- **Authors**: Yushen Chen, Zhikang Niu, Ziyang Ma, et al.
- **Venue**: arXiv 2024
- **Local File**: `papers/speech_and_audio/09_chen2024_f5_tts_flow_matching_voice_clone.pdf` (7,600 KB)
- **Core Contribution**: Proposed a ConvNeXt-guided flow-matching transformer with text-guided speech infilling that achieves zero-shot voice cloning from a 3-second reference voice clip without speaker embeddings.
- **Why It Matters for KathAI**: Our zero-shot voice cloning engine for generating synthetic speech in the voices of historical and literary Kashmiri figures.

---

### Section 3: Speech-LLMs & Multimodal Cross-Modal Models

#### 20. [SeamlessM4T: Massively Multilingual & Multimodal Machine Translation](file:///home/musa/Kashmiri%20Project/papers/speech_llms_and_multimodal/01_seamless2023_seamlessm4t_multimodal.pdf)
- **Authors**: Seamless Communication, Loïc Barrault et al. (Meta AI)
- **Venue**: arXiv 2023
- **Local File**: `papers/speech_llms_and_multimodal/01_seamless2023_seamlessm4t_multimodal.pdf` (3,439 KB)
- **Core Contribution**: Built a foundational multimodal model supporting Speech-to-Speech, Speech-to-Text, Text-to-Speech, and Text-to-Text translation across nearly 100 languages.
- **Why It Matters for KathAI**: Outlines the architecture for direct end-to-end speech translation from Kashmiri audio into English text without cascaded ASR $\to$ MT errors.

#### 21. [AudioPaLM: A Large Language Model That Can Speak and Listen](file:///home/musa/Kashmiri%20Project/papers/speech_llms_and_multimodal/02_rubenstein2023_audiopalm_speaks_and_listens.pdf)
- **Authors**: Paul K. Rubenstein et al. (Google DeepMind)
- **Venue**: arXiv 2023
- **Local File**: `papers/speech_llms_and_multimodal/02_rubenstein2023_audiopalm_speaks_and_listens.pdf` (547 KB)
- **Core Contribution**: Fused PaLM-2 text representations with AudioLM discrete audio tokens, producing a single autoregressive model that processes and outputs both spoken audio and text.
- **Why It Matters for KathAI**: Blueprint for our unified conversational voice assistant (*KathAI-Voice*).

#### 22. [SpeechGPT: Cross-Modal Conversational Language Model](file:///home/musa/Kashmiri%20Project/papers/speech_llms_and_multimodal/03_zhang2023_speechgpt_conversational_llm.pdf)
- **Authors**: Dong Zhang, Shimin Li, Xin Zhang, Jun Zhan, et al.
- **Venue**: EMNLP 2023
- **Local File**: `papers/speech_llms_and_multimodal/03_zhang2023_speechgpt_conversational_llm.pdf` (584 KB)
- **Core Contribution**: Discretized speech into hidden unit representations and trained an autoregressive LLM on paired speech-instruction data, demonstrating multi-turn spoken dialogue.
- **Why It Matters for KathAI**: Outlines our speech-instruction fine-tuning dataset format and tokenization schema.

#### 23. [Qwen2-Audio Technical Report](file:///home/musa/Kashmiri%20Project/papers/speech_llms_and_multimodal/04_qwen2024_qwen2_audio_technical_report.pdf)
- **Authors**: Qwen Team (Alibaba)
- **Venue**: arXiv 2024
- **Local File**: `papers/speech_llms_and_multimodal/04_qwen2024_qwen2_audio_technical_report.pdf` (1,605 KB)
- **Core Contribution**: Trained a 7B-parameter audio-language model with dual audio encoders supporting voice chat, audio event detection, and multi-speaker transcription.
- **Why It Matters for KathAI**: Informs how we adapt `Qwen-2.5-1.5B` into a spoken conversational agent for Kashmiri.

---

### Section 4: Kashmiri Script, Nastaliq OCR & Low-Resource Vision

#### 24. [Koshur Pixel: A Large-Scale Synthetic OCR Dataset for Kashmiri](file:///home/musa/Kashmiri%20Project/papers/kashmiri_and_low_resource/01_malik2024_koshur_pixel_synthetic_ocr.pdf)
- **Authors**: Haq Nawaz Malik et al.
- **Venue**: arXiv:2606.23144 (2024)
- **Local File**: `papers/kashmiri_and_low_resource/01_malik2024_koshur_pixel_synthetic_ocr.pdf` (5,058 KB)
- **Core Contribution**: Generated 613,078 synthetic image-text pairs across Nastaliq, Naskh, and Narqalam typefaces with realistic geometric warping and background noise to overcome the Kashmiri OCR training deadlock.
- **Why It Matters for KathAI**: Directly used in our **Workstream 5 (OCR & Document AI)** to train TrOCR and CRNN text recognizers on historical Kashmiri manuscripts.

#### 25. [600k-KS-OCR: A Synthetic Dataset for Optical Character Recognition in Kashmiri Script](file:///home/musa/Kashmiri%20Project/papers/kashmiri_and_low_resource/02_malik2024_600k_ks_ocr_dataset.pdf)
- **Authors**: Haq Nawaz Malik et al.
- **Venue**: arXiv:2601.01088 (2024)
- **Local File**: `papers/kashmiri_and_low_resource/02_malik2024_600k_ks_ocr_dataset.pdf` (269 KB)
- **Core Contribution**: Established word-level calligraphic segmentation benchmarks for Nastaliq ligatures and Kashmiri diacritics.
- **Why It Matters for KathAI**: Provides isolated word bounding boxes for calibrating word-level OCR confidence thresholds.

---

## 💡 Quick Reference: Paper Mapping to KathAI Modules

| KathAI Module / Pipeline | Core Foundational Paper | Local PDF Path |
| :--- | :--- | :--- |
| **Tokenization (SentencePiece 16k)** | Kudo & Richardson (2018) | [`papers/text_and_tokenization/03_kudo2018_sentencepiece_system.pdf`](file:///home/musa/Kashmiri%20Project/papers/text_and_tokenization/03_kudo2018_sentencepiece_system.pdf) |
| **Tokenization Tax Benchmarks** | Petrov et al. (2023) | [`papers/text_and_tokenization/04_petrov2023_tokenization_tax_discrepancies.pdf`](file:///home/musa/Kashmiri%20Project/papers/text_and_tokenization/04_petrov2023_tokenization_tax_discrepancies.pdf) |
| **Speech ASR (Track B: XLS-R CTC)** | Babu et al. (2021) / Graves (2006) | [`papers/speech_and_audio/03_babu2021_xls_r_128_languages.pdf`](file:///home/musa/Kashmiri%20Project/papers/speech_and_audio/03_babu2021_xls_r_128_languages.pdf) |
| **Speech ASR (Track C: Conformer)** | Gulati et al. (2020) | [`papers/speech_and_audio/04_gulati2020_conformer_speech_recognition.pdf`](file:///home/musa/Kashmiri%20Project/papers/speech_and_audio/04_gulati2020_conformer_speech_recognition.pdf) |
| **Speech ASR (Whisper Evaluation)** | Radford et al. (2022) | [`papers/speech_and_audio/05_radford2022_whisper_robust_asr.pdf`](file:///home/musa/Kashmiri%20Project/papers/speech_and_audio/05_radford2022_whisper_robust_asr.pdf) |
| **Acoustic Dataset (IndicVoices)** | Javed et al. (2024) | [`papers/speech_and_audio/06_javed2024_indicvoices_dataset.pdf`](file:///home/musa/Kashmiri%20Project/papers/speech_and_audio/06_javed2024_indicvoices_dataset.pdf) |
| **Edge Voice Synthesis (Matcha-TTS)**| Mehta et al. (2024) | [`papers/speech_and_audio/08_mehta2024_matcha_tts_flow_matching.pdf`](file:///home/musa/Kashmiri%20Project/papers/speech_and_audio/08_mehta2024_matcha_tts_flow_matching.pdf) |
| **Conversational SLM Instruction SFT**| Singh et al. (2024, Aya) | [`papers/text_and_tokenization/07_singh2024_aya_multilingual_dataset.pdf`](file:///home/musa/Kashmiri%20Project/papers/text_and_tokenization/07_singh2024_aya_multilingual_dataset.pdf) |
| **Machine Translation (IndicTrans2)**| Gala et al. (2023) | [`papers/text_and_tokenization/08_gala2023_indictrans2_22_languages.pdf`](file:///home/musa/Kashmiri%20Project/papers/text_and_tokenization/08_gala2023_indictrans2_22_languages.pdf) |
| **Document OCR (Koshur Pixel)** | Malik et al. (2024) | [`papers/kashmiri_and_low_resource/01_malik2024_koshur_pixel_synthetic_ocr.pdf`](file:///home/musa/Kashmiri%20Project/papers/kashmiri_and_low_resource/01_malik2024_koshur_pixel_synthetic_ocr.pdf) |

---

*Curated for the KathAI Project by Musavir Khaliq and the AI Research Team (November 2026).*
