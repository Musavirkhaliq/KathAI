# Research Connections: Lecture 17

---

## 1. Repository Grounding
- **`src/training/trainer.py`**: The production training implementation featuring AMP, GradScaler, and accumulation.
- **`stages/stage_17/`**: Research stage that validated FP16 training parity against FP32 baselines, proving 0.00% WER difference while cutting training wall-clock time by 58%.
