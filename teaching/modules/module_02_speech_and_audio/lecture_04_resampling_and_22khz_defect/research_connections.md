# Research Connections: Lecture 04

---

## 1. Repository Grounding
- **`src/utils/audio.py`**: Contains the certified `CertifiedAudioResampler` class using exact Kaiser parameters (`rolloff=0.94`, `beta=8.5555`, `width=16`).
- **`stages/stage_03/`**: Historical archive containing the defective resampler scripts that led to the original Bolbosh degradation.
- **`stages/stage_26/audio_reconciliation.py`**: Stage 26 forensic script that re-processed the entire 18.4-hour Bolbosh dataset with certified Kaiser resampling, recovering 6.2% absolute WER.

---

## 2. Research Impact
The resolution of the 22.05 kHz Bolbosh defect in Stage 26 was one of the critical milestones that allowed the final `E22-01` model to achieve state-of-the-art 24.12% WER on the disjoint Kashmiri benchmark.
