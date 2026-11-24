# Research Connections: Lecture 05

---

## 1. Repository Grounding
- **`src/utils/audio.py`**: Implements spectrogram generation and RMS energy tracking functions.
- **`stages/stage_05/`**: Explored Log-Mel filterbank front-ends for baseline Conformer experiments before the project shifted decisively to self-supervised raw waveform backbones.
- **`src/models/wav2vec2.py`**: Shows how the 7-layer convolutional feature extractor processes raw 1D audio tensors directly.

---

## 2. Research History
In Stage 05, the team evaluated whether feeding 80-channel Log-Mel filterbanks to a Conformer model could outperform fine-tuned Wav2Vec2 on raw waveforms. The raw waveform model outperformed the filterbank Conformer by 4.8% WER, primarily because the raw convolutional front-end preserved subtle acoustic cues distinguishing Kashmiri central vowels (/ɨ/, /ə/).
