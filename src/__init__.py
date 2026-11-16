"""
EduCasheer Kashmiri AI Research Ecosystem - Core Modular Package (src)
======================================================================
Provides verified, modular, high-performance implementations for Kashmiri:
- Normalization (Perso-Arabic script standardization, diacritic policies)
- Audio Processing (16 kHz resampling, mono conversion, chunking)
- ASR (Champion E22-01, Wav2Vec2-XLS-R, Conformer, Whisper)
- Decoding (Exact CTC slicing, calibrated KenLM beam search, temperature scaling)
- Evaluation (Levenshtein WER/CER, character error rate, code-switching analysis)
- Datasets (JSONL manifest parsing, split partitioning, audit validation)
- Translation & Romanization (Perso-Arabic -> Grierson / ASCII / IPA)
- Language Models (KenLM n-gram models, perplexity scoring)
- Shared Utilities (Device management, configuration parsing, logging)
"""

__version__ = "1.0.0"
