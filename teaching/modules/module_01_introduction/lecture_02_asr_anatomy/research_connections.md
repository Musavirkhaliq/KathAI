# Research Connections: Lecture 02

---

## 1. Grounding in the Repository
The anatomical decomposition discussed in this lecture directly mirrors the production architecture implemented across the following codebase modules:
- **`src/inference/pipeline.py`**: The unified inference pipeline implementing audio loading, preprocessing, model inference, and decoding.
- **`src/decoding/beam_search.py`**: Prefix beam search implementation interfacing with KenLM.
- **`configs/vocab.json`**: The 47-token certified Kashmiri character vocabulary.
- **`configs/acoustic/wav2vec2_kashmiri.yaml`**: The acoustic architecture configuration specifying hidden dimensions ($1024$), intermediate dimensions ($4096$), and attention heads ($16$).

---

## 2. Research History: From Hybrid Pipeline Dreams to CTC
- **Stage 01 Baseline**: The original repository attempt tested an off-the-shelf Urdu hybrid Kaldi model on Kashmiri recordings, producing catastrophic failure ($88.4\%$ WER).
- **Stage 04 Architecture Selection**: Explored Whisper (AED) fine-tuning vs. Wav2Vec2/XLS-R (CTC). While Whisper demonstrated fluent Urdu completions, it suffered from hallucination and severe vowel dropping in Kashmiri. CTC was chosen for its strict acoustic grounding and zero-hallucination guarantee.
