# Research Connections: Lecture 15

---

## 1. Repository Grounding
- **`src/models/wav2vec2.py`**: Interfacing with PyTorch `nn.CTCLoss`.
- **`stages/stage_15/`**: Audit that identified loss spikes caused by utterances where $T_\text{frames} < U_\text{chars}$, prompting the addition of `zero_infinity=True`.
