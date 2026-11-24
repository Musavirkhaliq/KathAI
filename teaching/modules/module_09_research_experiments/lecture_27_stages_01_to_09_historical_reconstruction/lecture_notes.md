# Lecture 27: Research Archaeology: Reconstructing Stages 01 to 09

---

## 1. The Reality of Real-World Machine Learning Research

Textbooks typically present research as a clean, forward-moving narrative: problem formulation, data collection, model training, and state-of-the-art results.
In real-world empirical AI research, reality is characterized by:
- Abandoned exploratory experiments.
- Silent data corruption bugs that persist across multiple stages.
- Inconsistent metric logging and undocumented scripts.
- Contradictory claims across commit histories.

**Research Archaeology** is the disciplined practice of reconstructing the authentic execution history of a machine learning repository from physical evidence: git commit histories, file modification timestamps, stdout logs, checkpoint tensor hashes, and raw audio manifests.

---

## 2. Chronological Forensic Reconstruction: Stages 01 to 09

### Stage 01: The Kaldi Urdu Baseline Failure (`stages/stage_01/`)
- **Hypothesis**: Off-the-shelf Urdu GMM-HMM acoustic models can recognize Kashmiri speech due to shared Perso-Arabic orthography and Indo-Aryan geographical proximity.
- **Physical Evidence**: Log file `stages/stage_01/logs/urdu_baseline_eval.log` dated August 2023.
- **Result**: Catastrophic **88.40% WER**.
- **Root Cause**: Complete absence of Kashmiri central vowels (/ɨ/, /ə/) in Urdu phoneme inventories, combined with forced alignment rejection of 38% of audio files.

### Stage 02: Initial Audio Aggregation & Sampling Rate Audit (`stages/stage_02/`)
- Audited 12 heterogeneous web and community sources, discovering a wild mixture of 8 kHz, 22.05 kHz, 44.1 kHz, and 48 kHz recordings.

### Stage 03: The Silent 22.05 kHz Resampling Defect (`stages/stage_03/`)
- Ingested 18.4 hours of Bolbosh speech.
- Used a defective Fourier resampler (`scipy.signal.resample`) with an over-attenuated Kaiser low-pass filter ($\beta=14.7$).
- **Consequence**: Wiped out acoustic frequencies between 5.8 kHz and 8.0 kHz, silently introducing a 14.2% CER penalty on sibilants that went undetected for 12 research stages.

### Stage 04: Architecture Pivot: Whisper (AED) vs. XLS-R (CTC)
- Evaluated OpenAI Whisper (Attention Encoder-Decoder) fine-tuning vs. XLS-R CTC.
- Whisper suffered from persistent hallucination loops and dropped un-diacriticized Kashmiri vowels. CTC was chosen for its strict acoustic monotonicity.

### Stage 05–09: Early Acoustic Scaling & The B-01 Baseline
- Assembled the first unified 20.4-hour dataset.
- Trained model **B-01**, achieving the project's first functional benchmark: **42.10% disjoint WER**.
