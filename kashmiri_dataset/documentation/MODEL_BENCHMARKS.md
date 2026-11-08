# Kashmiri AI State of the Art & Model Benchmarks

Comprehensive comparison of existing models, architectures, parameter counts, datasets, and benchmark results for Kashmiri speech and language technologies.

---

## 1. Speech Recognition (ASR) Models

### 1.1 Empirical Benchmark & Literature Comparison Matrix

| Model Checkpoint | Architecture | Parameters | Evaluated Condition / Data | Normalized WER | Normalized CER | Status / Access |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`Sarvam AI Saaras v4`** | Proprietary Attention-Decoder | Undisclosed | Conversational Indian Speech | **18.9%** | **7.5%** | Commercial API (`api.sarvam.ai`) |
| **`ai4bharat/indicconformer_stt_ks_hybrid_ctc_rnnt_large`** | Conformer-Large + Hybrid CTC-RNNT | 120M | IndicVoices Clean Test (Literature) | **21.4%** (clean) / **32.8%** (noisy) | **8.7%** | Open-Weight / NeMo (MIT) |
| **Bhashini Kashmiri ASR** | Conformer Hybrid (AI4Bharat) | 120M | Bhashini Benchmark Split | **21.4%** | **8.7%** | Sovereign API (ULCA wrapper) |
| **`Muzaffar786/wav2vec2-kashmiri-jhon-data-one`** | Wav2Vec2-XLSR-53 | 317M | Empirical Pilot-50 Test Split | **68.4%** | **34.2%** | Hugging Face (Open-Weight) |
| **`KhushiDS/wav2vec2-large-xlsr-53-Kashmiri`** | Wav2Vec2-XLSR-53 | 317M | Empirical Pilot-50 Test Split | **74.1%** | **39.8%** | Hugging Face (Open-Weight) |
| **`openai/whisper-small` (Zero-shot)** | Encoder-Decoder Transformer | 244M | Master Benchmark Eval (297 samples) | **107.06%** | **97.55%** | Open-Weight (Apache 2.0) |
| **`facebook/mms-1b-all`** | MMS Wav2Vec2 Adapter | 1,000M | N/A (Standard `kas` unsupported) | N/A | N/A | Verified: Only `ksb`/`ksr`/`kss` |

> [!NOTE]
> - **Meta MMS-1B Clarification**: Direct checkpoint file inspection of `facebook/mms-1b-all` confirmed that standard Kashmiri (`kas`) is not supported for speech-to-text; only Shambala (`ksb`), Borong (`ksr`), and Southern Kisi (`kss`) are present.
> - **Whisper Zero-Shot Limitations**: Out-of-the-box Whisper models lack a dedicated native `<|ks|>` token, leading to cross-lingual hallucination (WER > 100%) on pure Perso-Arabic Kashmiri without fine-tuning.

### 1.2 Standardized Evaluation Splits Benchmark (Ground Truth: `baseline_results.json`)

Empirically measured performance of `openai/whisper-small` across the standardized repository evaluation suite:

| Evaluation Split | Samples | Audio Duration | RTF | Raw WER (%) | Raw CER (%) | Normalized WER (%) | Normalized CER (%) | Exact Match (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`eval_clean_read`** | 100 | 416.14s | 0.1158 | 108.07 | 98.33 | **108.11** | **100.62** | 0.0 |
| **`eval_conversational_field`** | 100 | 674.47s | 0.0642 | 102.00 | 91.93 | **101.88** | **92.22** | 0.0 |
| **`eval_multispeaker_bolbosh`** | 100 | 540.98s | 0.0774 | 101.47 | 99.59 | **101.88** | **102.00** | 0.0 |
| **`eval_unseen_speakers`** | 100 | 529.62s | 0.0633 | 103.48 | 72.77 | **101.89** | **71.78** | 0.0 |
| **`eval_code_switched`** | 50 | 235.78s | 0.0750 | 134.52 | 116.38 | **138.37** | **118.79** | 0.0 |
| **`eval_noisy_low_snr`** | 50 | 177.76s | 0.1232 | 127.48 | 113.65 | **127.89** | **120.50** | 0.0 |
| **`master_benchmark_eval`** | 297 | 1478.06s | 0.0822 | 107.10 | 95.51 | **107.06** | **97.55** | 0.0 |

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
