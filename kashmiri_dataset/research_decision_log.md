# Kashmiri Language & Speech Research Decision Log

This document records all architectural, methodological, and scientific decisions made throughout the project lifecycle. Each decision is justified with scientific rationale, alternatives considered, and impact on future model training and academic publications.

---

### DECISION 001: Immutable Raw Data Policy
- **Date**: 2026-11-01
- **Status**: APPROVED & ENFORCED
- **Decision**: The directory `/home/musa/Kashmiri Project/Data` and all raw external downloads in `kashmiri_dataset/external/` must remain strictly immutable (read-only). All transformations must output to separate derivative directories (`processed/`, `cleaned/`, `merged/`, `splits/`).
- **Rationale**: Any accidental modification, in-place deletion, or re-encoding of raw ground-truth data destroys scientific reproducibility and eliminates the ability to audit original acoustic/text characteristics.
- **Alternatives Considered**: In-place conversion to save disk space.
- **Why Rejected**: High risk of irreversible data corruption and inability to trace raw errors back to source.

---

### DECISION 002: Strict Speaker-Disjoint Splitting
- **Date**: 2026-11-01
- **Status**: APPROVED & ENFORCED
- **Decision**: Partition audio training, validation, and testing splits such that `train_speakers ∩ test_speakers = ∅` and `train_speakers ∩ val_speakers = ∅`.
- **Rationale**: Random utterance-level shuffling leaks speaker identity, acoustic environment, and microphone impulse response between train and test splits, artificially deflating Word Error Rate (WER) by 15–30% and yielding scientifically invalid evaluation results.
- **Alternatives Considered**: Random uniform sample splitting (e.g. `train_test_split`).
- **Why Rejected**: Produces overly optimistic benchmarks that collapse when deployed on unseen native speakers in real-world conditions.

---

### DECISION 003: Dual-Mode Text Normalization (Acoustic vs Evaluation)
- **Date**: 2026-11-01
- **Status**: APPROVED & ENFORCED
- **Decision**: Maintain two distinct normalization tracks:
  1. `normalized_transcript`: Preserves script-specific vowel diacritics (Zer, Zabar, Pesh, Kashmiri short/long vowel marks) necessary for acoustic pronunciation modeling and TTS.
  2. `evaluation_transcript`: Standardizes character variants, strips non-contrastive diacritics, normalizes whitespace and punctuation for standardized Levenshtein Word Error Rate (WER) and Character Error Rate (CER) computation.
- **Rationale**: Kashmiri vowels carry critical semantic distinctions in speech (e.g., distinguishing verbal aspect and case inflection), but diacritic inconsistency across annotators artificially spikes WER during evaluation if not systematically handled.
- **Alternatives Considered**: Single aggressive diacritic stripping or unnormalized raw transcripts.
- **Why Rejected**: Stripping vowels damages acoustic/TTS modeling; unnormalized evaluation unfairly penalizes valid predictions due to font/Unicode variants.

---

### DECISION 004: Standard Audio Canonicalization (16 kHz Mono 16-bit PCM WAV)
- **Date**: 2026-11-01
- **Status**: APPROVED & ENFORCED
- **Decision**: Canonicalize all speech audio into single-channel 16,000 Hz, 16-bit linear PCM WAV with standard RIFF headers.
- **Rationale**: 16 kHz is the de facto standard for state-of-the-art foundation speech models (Conformer, Whisper, Wav2Vec 2.0, MMS). Converting compressed formats (WebM Opus, MP3) using lossless PCM decoding eliminates loader overhead and guarantees uniform mel-spectrogram feature extraction.
- **Alternatives Considered**: Retaining Opus/WebM and decoding dynamically in DataLoader.
- **Why Rejected**: Causes 3-5x GPU DataLoader bottlenecks during distributed training and incompatibility with audio libraries that lack native dynamic Opus decoders.

---

### DECISION 005: Quality-Flagging Over Silent Deletion
- **Date**: 2026-11-02
- **Status**: APPROVED & ENFORCED
- **Decision**: Questionable samples (e.g., low SNR, clipping, short duration < 0.5s, code-switched phrases) must receive descriptive quality flags (`is_clipped`, `low_snr`, `is_code_switched`) rather than being silently purged from the master manifest.
- **Rationale**: Different downstream tasks have different tolerance profiles. For instance, robust ASR models benefit from noisy, accented, and code-switched samples, whereas TTS training requires pristine studio audio. Flagging allows dynamic, task-specific filtering without data loss.
- **Alternatives Considered**: Hard threshold filtering (purging all audio with SNR < 15 dB).
- **Why Rejected**: Discards potentially valuable data and makes it impossible to reproduce the selection criteria later.
