# Kashmiri AI State of the Art & Model Benchmarks

Comprehensive comparison of existing models, architectures, parameter counts, datasets, and benchmark results for Kashmiri speech and language technologies.

---

## 1. Speech Recognition (ASR) Models

| Model Checkpoint | Architecture | Parameters | Training Data | Reported WER | Reported CER | Access / Repository |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`ai4bharat/indicconformer_stt_ks_hybrid_ctc_rnnt_large`** | Conformer-Large + Hybrid CTC-RNNT | 120M | IndicVoices Kashmiri + multi-lingual Indian speech | **21.4%** (clean) / **32.8%** (noisy) | **8.2%** | Hugging Face (MIT) |
| **`facebook/mms-1b-all` (`kas` adapter)** | Wav2Vec 2.0 (1B) + 1D Conv Adapter | 1,000M | 1,400+ languages (Bible / read speech) | **28.6%** | **11.4%** | Hugging Face (CC-BY-NC 4.0) |
| **`facebook/mms-300m` (`kas` adapter)** | Wav2Vec 2.0 (300M) + Adapter | 300M | MMS Multilingual Pretraining | **34.2%** | **14.1%** | Hugging Face (CC-BY-NC 4.0) |
| **Whisper-Small Fine-tune (Community)** | Encoder-Decoder Transformer | 244M | OpenSLR SLR122 + KTTS | **26.8%** | **10.5%** | Hugging Face |
| **Whisper-Medium Fine-tune** | Encoder-Decoder Transformer | 769M | Multi-task pseudo-labeled Indian speech | **23.1%** | **9.1%** | Hugging Face |

### Technical Analysis: Conformer vs Whisper for Kashmiri ASR
- **IndicConformer Advantages**: Specifically regularized for Indian language phonetics; native support for retroflex and aspirated sounds; CTC-RNNT hybrid architecture enables real-time streaming; significantly lower latency on GPU/CPU.
- **Whisper Advantages**: Naturally punctuated output; robust zero-shot English loanword transcription (vital for code-switched Kashmiri conversations); sequence-to-sequence beam search handles acoustic omissions better.
- **Recommendation**: Deploy **IndicConformer Large** as the high-throughput, low-latency streaming baseline, and **Whisper-Small/Medium** with LoRA as the rich conversational/offline transcription engine.

---

## 2. Text-to-Speech (TTS) Models

| Model Name | Framework / Architecture | Parameters | Training Voice | Reported Quality / MOS | License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`GAASH-Lab/Matcha-TTS-Kashmiri`** | Optimal Transport Conditional Flow Matching (OT-CFM) | ~20M | Bolbosh (Multi-speaker male & female) | **MOS 3.92 / 5.0**; MCD 5.42 dB | CC-BY-4.0 |
| **`facebook/mms-tts` (`kas`)** | VITS (Variational Inference with adversarial learning) | ~35M | Meta MMS single-speaker read corpus | **MOS 3.10 / 5.0** | CC-BY-NC 4.0 |
| **IndicTTS Kashmiri** | FastSpeech2 / Tacotron2 + HiFi-GAN | ~40M | AI4Bharat / Bhashini native speaker | **MOS 3.55 / 5.0** | MIT |

### Key Insight for Kashmiri TTS
Standard multilingual TTS architectures fail to render Kashmiri accurately because they lack Kashmiri-specific vowel graphemes (e.g. central Tschva vowels) and fail to handle Perso-Arabic diacritization. **Bolbosh** from NIT Srinagar is the first model to implement script-aware vocabulary expansion specifically designed for Kashmiri diacritics.

---

## 3. Machine Translation (NMT) Models

| Model | Architecture | Parameters | Parallel Pairs | BLEU (EN → KS) | BLEU (KS → EN) | License |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`ai4bharat/indictrans2-indic-indic-dist-320M`** | Transformer Encoder-Decoder | 320M | BPCC (22 languages) | 26.4 | 29.8 | MIT |
| **`ai4bharat/indictrans2-en-indic-1B`** | Transformer Encoder-Decoder | 1,000M | BPCC + NLLB | 28.1 | 32.4 | MIT |
| **`facebook/nllb-200-distilled-600M`** | Sparse Mixture-of-Experts / Dense | 600M | Seed parallel corpora | 18.2 | 22.0 | CC-BY-NC 4.0 |

---

## 4. Large Language Models (LLM / SLM)

| Model Name | Base Foundation | Kashmiri Tuning Data | Task / Modality | License |
| :--- | :--- | :--- | :--- | :--- |
| **`MISHANM/Kashmiri_text_generation_Llama3_8B_instruct`** | Meta-Llama-3-8B-Instruct | 49K instruction samples | Conversational QA, Translation | Llama 3 Community |
| **`megaketu555/kashmiri-nastaliq-llm`** | Llama / Mistral variant | Nastaliq text | Kashmiri chat | Open |
| **`Omarrran/kashmiri-llm-bert-base-hnm`** | Custom BERT from scratch | KS-PRET-5M | Token classification, fill-mask | Apache-2.0 |
| **`Jibby1729/Koshur-JibDiacritizer-152M`** | Character Transformer | Diacritized parallel text | Automatic Diacritization | Open |
