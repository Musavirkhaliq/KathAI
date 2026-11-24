# Research Connections: Lecture 23

---

## 1. Repository Grounding
- **`src/decoding/beam_search.py`**: Implementation of shallow fusion with calibrated $T$, $\alpha$, and $\beta$.
- **`stages/stage_23/`**: Stage auditing the conflicting WER numbers (24.74% vs. 24.12%), proving that the difference was caused entirely by Config A ($\alpha=0.3, \beta=1.0$) vs. Config B ($\alpha=0.5, \beta=1.5$).
