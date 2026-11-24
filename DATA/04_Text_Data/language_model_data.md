# Language Model Training & Pretraining Readiness

1. **KenLM / N-gram Models**: Evaluated and trained on `kashmiri_wikipedia_cleaned.jsonl` (34.99 MB) + `kashmiri_text_corpus.jsonl` (3.45 GB), generating vocabulary caches for ASR CTC beam search decoding.
2. **LLM Instruction Fine-Tuning**: `kreedz_qa/qwen_dataset.jsonl` (1.11 MB) and `aya-kashmiri-only` instruction/response pairs ready for conversational tuning.
