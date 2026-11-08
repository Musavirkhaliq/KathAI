# KathAI (کتھ) — What We Need to Do: Step-by-Step Action Plan

**Document**: Sequential Technical Implementation Roadmap  
**Path**: `research/02_WHAT_WE_NEED_TO_DO_STEP_BY_STEP.md`  
**Repository**: `Musavirkhaliq/KathAI`  
**Date**: November 2026  

---

## 1. Overview of Phased Progression

To transition the curated datasets and pilot experiments into a complete, state-of-the-art multimodal AI suite, the project is structured into **5 distinct phases** executed in logical dependency order:

```mermaid
graph TD
    subgraph Phase 1: Immediate Execution [Days 1 - 14]
        A1[Step 1.1: Run Full Schedule XLS-R 300M ASR] --> A2[Step 1.2: Fine-tune Whisper-Small LoRA]
        A3[Step 1.3: Train 5-Gram KenLM LM] --> A4[Step 1.4: Tokenizer Grafting on Qwen-2.5-1.5B]
        A5[Step 1.5: Setup TrOCR on Koshur-OCR-Synth]
    end

    subgraph Phase 2: Core Foundation Training [Weeks 3 - 6]
        A2 --> B1[Step 2.1: IndicConformer Hybrid CTC-RNNT]
        A4 --> B2[Step 2.2: CPT of Qwen-2.5 on 3.4GB Corpus]
        A1 --> B3[Step 2.3: Train Matcha-TTS on Bolbosh]
        A5 --> B4[Step 2.4: Scale TrOCR to 613k Koshur Pixel]
        B2 --> B5[Step 2.5: Train Character Diacritizer Model]
    end

    subgraph Phase 3: Alignment, Multitask & SFT [Weeks 7 - 10]
        B2 --> C1[Step 3.1: Instruction SFT on Aya & ShareGPT]
        B1 --> C2[Step 3.2: Multi-Dialect Evaluation Benchmark]
        B3 --> C3[Step 3.3: High-Fidelity F5-TTS Multi-Speaker]
        B4 --> C4[Step 3.4: Donut / Qwen2-VL Page Layout OCR]
        B1 --> C5[Step 3.5: Speech-to-Text Translation Model]
    end

    subgraph Phase 4: Downstream Tools & Translation [Weeks 11 - 14]
        C1 --> D1[Step 4.1: Fine-tune IndicTrans2 on 344k Pairs]
        D1 --> D2[Step 4.2: Tri-Directional Transliteration Engine]
        D2 --> D3[Step 4.3: Grierson Dictionary Graph & API]
        C1 --> D4[Step 4.4: Speech-LLM Direct Voice Agent]
    end

    subgraph Phase 5: Production Deployment [Weeks 15 - 18]
        D4 --> E1[Step 5.1: Export to ONNX, TensorRT, GGUF, vLLM]
        E1 --> E2[Step 5.2: Mobile Edge Runtime with Sherpa-ONNX]
        E2 --> E3[Step 5.3: Unified KathAI Gradio & API Suite]
    end
```

---

## 2. Step-by-Step Action Plan

### PHASE 1: Immediate High-Priority Execution (Days 1 – 14)

#### Step 1.1: Full-Schedule ASR Training (Experiment B-01-Full)
- **Target**: Train `facebook/wav2vec2-xls-r-300m` with our 161-token Kashmiri Character CTC vocabulary for 50 full epochs.
- **Script**: `kashmiri_dataset/scripts/train_b01_full_schedule.py`.
- **Configuration**:
  - Batch size: 16 per GPU with gradient accumulation = 2 (effective batch size 32).
  - Optimizer: AdamW (`lr=1e-4`, weight decay `0.01`, cosine decay with 10% warmup).
  - Mixed precision: PyTorch AMP (`fp16`).
  - Length bucketing: Utterances grouped by duration to minimize padding overhead by 40%.
  - Hardware: NVIDIA DGX Spark GB10.
- **Verification Gate**: Monotonic reduction of CTC loss below **0.80**; character error rate (CER) on `test_clean` < **14.0%**.

#### Step 1.2: Conversational ASR Fine-Tuning (OpenAI Whisper-Small LoRA)
- **Target**: Fine-tune `openai/whisper-small` (244M) using Parameter-Efficient Fine-Tuning (PEFT / LoRA).
- **Rationale**: Out-of-the-box Whisper models experience severe cross-lingual hallucination on Kashmiri (empirically measured at **107.06% Norm WER / 97.55% Norm CER** on `master_benchmark_eval` across 297 samples) because Whisper lacks a native `<|ks|>` token. Prompt conditioning and LoRA adaptation specialize the decoder, leveraging Whisper's natural punctuation and robust English/Urdu code-switching handling.
- **Configuration**:
  - Target modules: `q_proj`, `v_proj`, `out_proj`.
  - LoRA rank: $r=16$, $\alpha=32$, dropout $0.05$.
  - Conditioning: Force language token `<|ur|>` or custom Kashmiri prompt tokens to anchor Perso-Arabic generation.
- **Verification Gate**: WER on `test_clean` < **22.0%**; WER on `test_code_switched` < **30.0%**.

#### Step 1.3: Train Statistical 5-Gram KenLM for CTC Shallow Fusion
- **Target**: Compile a 5-gram language model from our 4.13M-line Kashmiri text corpus (`kashmiri_text_corpus.jsonl`).
- **Tool**: `kenlm` (`lmplz` with Kneser-Ney smoothing + `build_binary`).
- **Integration**: Pair with `pyctcdecode` during Wav2Vec2 / XLS-R inference to resolve acoustic ambiguities via language model shallow fusion:
  $$\hat{Y} = \arg\max_Y \left( \log P_{\text{CTC}}(Y|X) + \alpha \log P_{\text{LM}}(Y) + \beta |Y| \right)$$
- **Verification Gate**: Absolute WER reduction of **3.0% to 5.5%** over greedy CTC decoding.

#### Step 1.4: Tokenizer Vocabulary Grafting for Foundation SLM
- **Target**: Graft our custom 16,000-token Kashmiri SentencePiece BPE vocabulary onto `Qwen/Qwen2.5-1.5B` and `meta-llama/Llama-3.2-1B`.
- **Methodology**:
  - Compute vocabulary union: Preserve all original tokens and append unique Kashmiri subwords.
  - Initialize newly added embedding rows with the mean embedding of existing Arabic/Perso-Arabic characters.
  - Resize input embeddings and LM head projection layer in PyTorch.
- **Verification Gate**: Token fertility on Kashmiri test sentences drops from **>4.1** to **<1.8**, with 0 UNKs.

#### Step 1.5: Setup TrOCR Baseline on Kashmiri Synthetic Line Images
- **Target**: Establish line-level optical character recognition using `microsoft/trocr-base-stage1` or `TrOCR-small`.
- **Dataset**: `kashmiri_dataset/external/faizaniqbal_koshur-ocr-synth` (285k line images).
- **Configuration**: Image encoder (ViT) + text decoder (RoBERTa initialized with Perso-Arabic/Urdu weights).
- **Verification Gate**: Character Error Rate (CER) < **8.0%** on synthetic held-out validation lines.

---

### PHASE 2: Core Foundation Training (Weeks 3 – 6)

#### Step 2.1: Streaming ASR Training (IndicConformer Large Hybrid CTC-RNNT)
- **Target**: Train `ai4bharat/indicconformer_stt_ks_hybrid_ctc_rnnt_large` using NVIDIA NeMo.
- **Data**: All 141.6 hours from `speech_master_manifest.jsonl`, augmented with SpecAugment (time/freq masking) and speed perturbation ($0.9\times, 1.0\times, 1.1\times$).
- **Objective**: Deploy a low-latency, real-time streaming recognition engine capable of running on CPU/edge devices.
- **Verification Gate**: Real-time factor (RTF) < **0.05**; WER on `test_clean` < **18.0%**.

#### Step 2.2: Continued Pretraining (CPT) of Kashmiri SLM (1.5B – 3B)
- **Target**: Perform CPT on the vocabulary-adapted Qwen-2.5-1.5B across 3.4 GB of high-quality Kashmiri text (~800M tokens).
- **Data Mixture**:
  - 40% Kashmiri native text (KS-PRET-5M, 3.1M corpus, cleaned news, literature, Wikipedia).
  - 30% Parallel Kashmiri-English pairs with directional translation tokens (`<ks_to_en>`, `<en_to_ks>`).
  - 20% Medical, reasoning, and encyclopedic Kashmiri texts (`maitreyajayaraj`).
  - 10% High-quality English/Urdu anchor text to prevent catastrophic forgetting.
- **Hyperparameters**: AdamW (`lr=2e-5`), cosine learning rate schedule, sequence length 2048, flash-attention-2.
- **Verification Gate**: Validation perplexity on Kashmiri held-out articles < **12.5**.

#### Step 2.3: Multi-Speaker Flow-Matching TTS (Matcha-TTS Kashmiri)
- **Target**: Adapt **Matcha-TTS** (Optimal Transport Conditional Flow Matching) and **VITS** on our speech corpus.
- **Data**: Bolbosh (25+ hours, multi-speaker male/female, studio grade) + KTTS (3.5 hours, single-speaker male anchor).
- **Acoustic Front-End**: Explicit vowel diacritizer + grapheme-to-phoneme (G2P) converter to prevent missing central vowel phonemes.
- **Verification Gate**: Mel-Cepstral Distortion (MCD) < **4.8 dB**; Mean Opinion Score (MOS) > **3.9 / 5.0**.

#### Step 2.4: Scale TrOCR to Full Koshur Pixel Dataset
- **Target**: Scale TrOCR training to the 613,078 image-text pairs in `omarrran_koshur_pixel`.
- **Training Strategy**: Two-stage curriculum learning:
  - Stage 1: Pretrain on single words and short lines (`sentence_line` config, 378k samples).
  - Stage 2: Fine-tune on multi-line paragraphs with font augmentations, background textures, and simulated scan blur.
- **Verification Gate**: Word Recognition Rate > **92.0%** across both Nastaliq and Naskh typefaces.

#### Step 2.5: Train Autonomous Kashmiri Diacritizer (Koshur-Diacritizer)
- **Target**: Train a sequence-to-sequence character model (e.g. ByT5-small or CanTK) on `omarrran_kashmiri_parallel_diacratic_to_non_diacratic_text_dataset`.
- **Task**: Given un-diacritized raw Perso-Arabic text, restore the exact central vowels (`U+0672`, `U+065F`) and palatalization marks (`U+065A`).
- **Role**: Essential front-end preprocessor for TTS synthesis, high-precision semantic search, and educational tools.
- **Verification Gate**: Diacritic Error Rate (DER) < **4.5%**.

---

### PHASE 3: Alignment, Instruction Tuning & Document AI (Weeks 7 – 10)

#### Step 3.1: Supervised Fine-Tuning (SFT) of Kashmiri LLM
- **Target**: Instruction-tune the CPT model to follow instructions natively in Kashmiri.
- **Data**:
  - Aya Kashmiri Instruction Dataset (100,000+ pairs).
  - ShareGPT Kashmiri conversational turns.
  - Kashmiri Clinical Reasoning & Medical QA (`maitreyajayaraj`).
  - Kashmiri Cultural, Literary, and Historical Q&A.
- **Format**: Standard ChatML format:
  ```text
  <|im_start|>user
  کٲشِر زَبانہِ ہٕنٛز تَوٲریٖخ کَر مۄختَصَر بَیان۔<|im_end|>
  <|im_start|>assistant
  کٲشُر زَبان چھِ ہِند-یورپی خانٛدانٕچ اَکھ اَہَم دَردی زَبان...<|im_end|>
  ```
- **Verification Gate**: MT-Bench Kashmiri score > **7.0 / 10.0**; 0% script degradation into Urdu or Hindi.

#### Step 3.2: Comprehensive Multi-Dialect Evaluation Benchmark
- **Target**: Evaluate all ASR, LLM, and TTS models on stratified slices:
  - Dialect splits: Srinagar Standard (Yamraz), South (Maraz), North (Kamraz).
  - Acoustic conditions: Clean studio vs Field smartphone vs Background noise.
  - Script variations: Diacritized vs Non-diacritized evaluation.

#### Step 3.3: High-Fidelity Voice Cloning (F5-TTS / E2-TTS)
- **Target**: Train a non-autoregressive flow-matching speech synthesis model (F5-TTS) for zero-shot and few-shot Kashmiri voice cloning from 5-second reference clips.
- **Verification Gate**: Natural prosody, preservation of Kashmiri sentence-final interrogative intonation contours.

#### Step 3.4: End-to-End Page Layout & Document Understanding (Donut / Qwen2-VL)
- **Target**: Fine-tune `naver-clova-ix/donut-base` or `Qwen/Qwen2-VL-2B-Instruct` on page-level renders from `omarrran_koshur_pixel` (42,679 page images).
- **Task**: Directly parse full scanned pages into structured JSON/Markdown, handling columns, headers, footnotes, and poetry stanzas.
- **Verification Gate**: Tree Edit Distance (TED) < **0.15** on structured page benchmarks.

#### Step 3.5: End-to-End Speech-to-Text Translation (Kashmiri Audio -> English Text)
- **Target**: Fine-tune Whisper-Medium directly on speech-to-English translation using aligned parallel audio.
- **Advantage**: Bypasses the cascaded error rate of (ASR WER + NMT error).
- **Verification Gate**: SacreBLEU score > **24.0** on Kashmiri-English speech translation.

---

### PHASE 4: Downstream Tools, Translation & Multi-Script Support (Weeks 11 – 14)

#### Step 4.1: Fine-Tune IndicTrans2 for Kashmiri Translation
- **Target**: Fine-tune `ai4bharat/indictrans2-en-indic-1B` and `indictrans2-indic-indic-dist-320M` on our 344,445 parallel pairs.
- **Directions**: English ↔ Kashmiri, Urdu ↔ Kashmiri, Hindi ↔ Kashmiri.
- **Data Augmentation**: Back-translation of 1M monolingual Kashmiri sentences into English to double parallel corpus density.
- **Verification Gate**: SacreBLEU on English → Kashmiri > **28.5**; Kashmiri → English > **33.0**.

#### Step 4.2: Tri-Directional Script Transliteration Engine
- **Target**: Build an exact phonetic transliteration tool between:
  $$\text{Perso-Arabic (کٲشُر)} \longleftrightarrow \text{Devanagari (कॉशुर)} \longleftrightarrow \text{Romanized Kashmiri (Koshur)}$$
- **Implementation**: Hybrid rule-based mapping (covering special Kashmiri diacritics: ٲ ↔ ॲ, ۄ ↔ ऑ) + fine-tuned `IndicXlit` seq2seq model for ambiguous ligatures.
- **Verification Gate**: Character transliteration accuracy > **99.2%**.

#### Step 4.3: Sir George Grierson Kashmiri Lexicon Graph & API
- **Target**: Digitize and parse the 131,231 entries of Grierson's Dictionary into a graph database and search API with morphological stemmer, root word derivations, etymology, and example sentences.

#### Step 4.4: Speech-LLM Direct Voice Agent
- **Target**: Train a linear projection or Q-Former bridging the fine-tuned ASR audio encoder directly into the Kashmiri SLM text embedding space.
- **Result**: Spoken Kashmiri Audio IN $\longrightarrow$ Spoken Kashmiri Audio OUT (via TTS) with sub-second voice-to-voice latency.

---

### PHASE 5: Optimization, Mobile Packaging & Deployment (Weeks 15 – 18)

#### Step 5.1: Model Optimization & Quantization
- **LLM**: Quantize Kashmiri SLM to 4-bit AWQ, GPTQ, and GGUF (`Q4_K_M`, `Q8_0`) for execution in `llama.cpp` and `vLLM` (throughput > 45 tokens/sec).
- **ASR & TTS**: Export Conformer and Matcha-TTS models to ONNX and TensorRT with dynamic shapes.
- **OCR**: Export TrOCR to ONNX with INT8 quantization for sub-50ms line recognition.

#### Step 5.2: Offline Edge Runtime (Sherpa-ONNX Mobile Engine)
- **Target**: Package quantized ASR, TTS, and Transliteration models into `sherpa-onnx` binaries for offline edge deployment on Android and iOS devices.

#### Step 5.3: Unified KathAI Public Demo Platform
- **Target**: Launch a production Gradio and FastAPI web platform featuring:
  - Real-time live audio transcription (Microphone $\to$ Perso-Arabic text).
  - Voice synthesizer (Text $\to$ natural Kashmiri male/female voice).
  - Conversational Kashmiri Assistant (Chatbot with cultural awareness).
  - Document OCR scanner (Upload image of Kashmiri book/page $\to$ editable text).
  - Multilingual Translator & Transliteration tool.

---
*Proceed to [03_LANGUAGE_MODELS_PLAN_LLM_SLM.md](file:///home/musa/Kashmiri%20Project/research/03_LANGUAGE_MODELS_PLAN_LLM_SLM.md) for the deep technical architecture of the Kashmiri Language Models.*
