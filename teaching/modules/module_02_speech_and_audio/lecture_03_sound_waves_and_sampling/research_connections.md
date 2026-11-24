# Research Connections: Lecture 03

---

## 1. Grounding in the Repository
- **`src/utils/audio.py`**: Enforces strict verification that all incoming training and evaluation audio is formatted as single-channel, 16-bit, 16 kHz PCM WAV.
- **`stages/stage_02/scripts/inspect_audio_specs.py`**: Historical script from Stage 02 that audited sampling rates across 12 raw data sources, uncovering widespread mismatches (8 kHz, 22.05 kHz, 44.1 kHz, 48 kHz).

---

## 2. Research History
In the earliest stages of the EduCasheer project, audio files were collected from diverse web sources, community mobile recordings, and YouTube broadcasts. The failure to understand Nyquist sampling led to naive nearest-neighbor downsampling in early Stage 03 scripts, corrupting high-frequency Kashmiri consonants.
