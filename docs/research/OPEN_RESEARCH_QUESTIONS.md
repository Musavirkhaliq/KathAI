# Kashmiri ASR Open Research Questions & Future Roadmap

**Project**: EduCasheer Kashmiri Automatic Speech Recognition Research
**Date**: November 16, 2026

---

## 1. Primary Open Research Questions

### RQ1: How to Close the Remaining 18.99% Oracle Headroom Ceiling?
- **Context**: Top-32 beam search reveals an Oracle WER ceiling of **18.99%** (vs 24.40% for KenLM), leaving 5.4 pp of unexploited candidate headroom.
- **Finding from Stage 24**: A 329k-parameter local BiGRU reranker failed to beat KenLM, proving that shallow candidate-level models cannot learn Kashmiri grammar without external pretraining.
- **Hypothesis**: Replacing shallow rankers with a generative Kashmiri Large Language Model (e.g., Llama-3-8B fine-tuned on Kashmiri text) via N-best prompt rescoring will unlock this headroom.

### RQ2: Can Speech Foundation Scaling (1B+ Parameters) Eliminate Central Vowel Confusion?
- **Context**: 78.6% of remaining errors are acoustic representation errors where the correct word is missing from the search beam entirely.
- **Hypothesis**: Scaling from `wav2vec2-xls-r-300m` to `wav2vec2-xls-r-1b` or `mms-1b` with 500+ hours of speech will provide sharper acoustic separation for central vowel contrasts (`U+0672`, `U+065F`).

### RQ3: End-to-End Multimodal Speech-to-Text & Kashmiri TTS
- **Context**: Current production operates on a cascaded architecture (CTC Acoustic Model -> KenLM -> ByT5 Romanizer).
- **Hypothesis**: End-to-end discrete speech tokenizers (e.g., EnCodec / SpeechTokenizer) coupled with autoregressive language models can unify Kashmiri ASR, Speech-to-Text Translation, and Text-to-Speech in a single model.
