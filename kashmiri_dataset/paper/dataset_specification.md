# Kashmiri Multimodal Research Dataset Specification (v1.0.0)

## 1. Overview
This specification details the standardized multimodal corpus for the Kashmiri language (*Koshur*, ISO `ks`/`kas`), designed for acoustic modeling, automatic speech recognition (ASR), large language modeling (LLM), machine translation, and phonetic studies.

---

## 2. Quantitative Summary

| Pillar / Modality | Primary Format | Volume | Target Applications |
| :--- | :--- | :--- | :--- |
| **KASHMIRI-TEXT** | JSONL | **4,120,142 records** | Monolingual Pretraining, Instruction SFT (Aya), Clinical & Mathematical Reasoning |
| **KASHMIRI-SPEECH** | JSONL + 16kHz WAV | **9,180 samples (13.12 hrs)** | Standardized ASR, Acoustic Pretraining, Single-Speaker TTS Alignments |
| **KASHMIRI-PARALLEL**| JSONL | **344,445 pairs** | Machine Translation (En-Ks, Hi-Ks), Transliteration & Code-Mixing |
| **KASHMIRI-LEXICON** | JSONL | **131,231 entries** | Grapheme-to-Phoneme (G2P), Pronunciation Dictionaries, Lexicography |

---

## 3. Acoustic Specifications
- **Container / Encoding**: Linear 16-bit PCM WAV (RIFF Header)
- **Sampling Rate**: 16,000 Hz
- **Channels**: 1 (Mono)
- **VAD Trimming**: 50ms silence padding at phrase boundaries
- **Mean SNR**: 39.05 dB

---

## 4. Orthographic Specifications
- **Unicode NFC / NFKC Normalized**
- **Perso-Arabic Keheh (`U+06A9`) Standardized**
- **Dual Representation**:
  - `normalized_text`: Preserves contrastive vowel marks (`U+0672`, `U+065F`) for acoustic training.
  - `evaluation_text`: Diacritic-free, variant-unified for standard WER scoring.

---

## 5. Splitting Strategy (Zero Leakage)
- `train.jsonl`: 7,307 samples (10.40 hrs) — 214 training speakers.
- `val.jsonl`: 945 samples (1.45 hrs) — 27 disjoint validation speakers.
- `test.jsonl`: 928 samples (1.27 hrs) — 27 disjoint evaluation speakers.
- `train_spks ∩ test_spks = ∅`
