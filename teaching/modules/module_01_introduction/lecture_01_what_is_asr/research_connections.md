# Research Connections: Lecture 01 & Stage 01 Forensics

---

## 1. Repository Stage Lineage

- **Primary Historical Stage**: `stages/stage_01/` (Initial Research Environment and Kashmiri Corpus Discovery).
- **Associated Commits / Artifacts**:
  - `stages/stage_01/scripts/evaluate_urdu_baseline.py` (Historical zero-shot evaluation script).
  - `stages/stage_01/logs/urdu_baseline_eval.log` (Raw execution log documenting the $88.4\%$ WER failure).
- **Master Documentation Reference**:
  - `PROJECT_MASTER_GUIDE.md` (Section 2.1: Project Inception and Linguistic Scope).
  - `docs/research/STAGE_REGISTRY.md` (Stage 01 Entry: Baseline Exploration).
  - `docs/research/RESEARCH_TIMELINE.md` (Phase 1: Pre-training Realities).

---

## 2. Forensic Analysis of the Stage 01 Execution Log

The historical log file `stages/stage_01/logs/urdu_baseline_eval.log` records the first quantitative benchmark of the project:

```text
2024-03-12 14:22:01 [INFO] Loading model: facebook/wav2vec2-large-xlsr-53-urdu...
2024-03-12 14:22:15 [INFO] Evaluating on 120 Kashmiri pilot utterances (duration: 1.84 hours)...
2024-03-12 14:28:44 [INFO] EVALUATION COMPLETE:
  Total Reference Words:     8,412
  Correct Words (C):           975 (11.59%)
  Substitutions (S):         5,224 (62.10%)
  Deletions (D):             2,036 (24.20%)
  Insertions (I):              178 ( 2.12%)
  -----------------------------------------
  Word Error Rate (WER):     88.42%
  Character Error Rate (CER):54.18%
```

### Forensic Takeaways
1. **The Deletion Explosion ($D = 24.20\%$)**: Almost one quarter of all spoken Kashmiri words had their central vowels completely eliminated by the Urdu model.
2. **The Substitution Catastrophe ($S = 62.10\%$)**: When the model did predict characters, it substituted Persian/Urdu consonants for Kashmiri dental affricates (/t͡s/ was mapped to /t/ or /s/).

---

## 3. Production Code Lineage in `src/`

The failure in Stage 01 directly motivated the creation of the production modules in `src/`:
- `src/normalization/kashmiri_char_map.py`: Constructed a certified Kashmiri-specific vocabulary that explicitly includes all central vowels and affricates.
- `src/shared/logging.py`: Standardized the forensic logging protocol used across all subsequent stages.
- `src/evaluation/metrics.py`: Replaced ad-hoc evaluation scripts with certified micro-averaged Levenshtein metrics.
