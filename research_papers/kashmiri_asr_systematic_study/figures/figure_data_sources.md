# Data Sources and Provenance for Manuscript Figures

This document details the exact source data tables, CSVs, and mathematical transformations used to render all figures in the manuscript.

---

## 1. Figure 1: Speech Corpus Composition and Sample Rate Distribution
- **Generated Files**: `fig1_corpus_and_samplerate.png`, `.svg`, `.pdf`
- **Data Source Files**:
  - `data/source_inventory.csv`
  - `data/source_licenses.csv`
  - `kashmiri_dataset/experiments/stage20s_audio_integrity_recovery/audits/stage20s_sample_rate_distribution.csv`
- **Underlying Metrics**:
  - Corpus duration: Bolbosh (58.40h / 27.2%), IndicVoices-R (41.87h / 19.5%), IndicVoices (29.80h / 13.9%), Mirwaiz Quran (20.91h / 9.7%), Other 8 sources (63.67h / 29.7%). Total: 214.65 hours.
  - Sample rate distribution: Tier A (66.2% 16k, 33.8% 22.05k), Tier B (48.2% 16k, 51.8% 22.05k), Dev Assessment (56.9% 16k, 43.1% 22.05k), Dev Calibration (58.3% 16k, 41.7% 22.05k).

---

## 2. Figure 2: Acoustic Scaling Trajectory (B-01 to B-02C)
- **Generated Files**: `fig2_acoustic_scaling.png`, `.svg`, `.pdf`
- **Data Source Files**:
  - `docs/EXPERIMENT_REGISTRY.md` (Section 2.2)
  - `history/stage_by_stage_experiment_ledger.csv`
- **Underlying Metrics**:
  - B-01: 92.94h, Greedy WER 33.86%, KenLM WER 31.80%
  - B-02A: 104.93h, Greedy WER 29.42%, KenLM WER 26.30%
  - B-02B: 123.45h, Greedy WER 29.39%, KenLM WER 26.41%
  - B-02C: 165.32h, Greedy WER 27.84%, KenLM WER 25.24%
  - Evaluated on canonical `val.jsonl` (6,728 utterances, 104,685 words).

---

## 3. Figure 3: Resampling Integrity and Audio Correction Impact
- **Generated Files**: `fig3_resampling_integrity_impact.png`, `.svg`, `.pdf`
- **Data Source Files**:
  - `results/historical_results_reconciliation.csv`
  - `kashmiri_dataset/experiments/stage20s_audio_integrity_recovery/evaluation/stage20s_dual_protocol_results.csv`
  - `kashmiri_dataset/experiments/stage20s_audio_integrity_recovery/evaluation/stage20s_sourcewise_resampling_impact.csv`
- **Underlying Metrics**:
  - Historical Unresampled Protocol: 3,709 Substitutions, 669 Deletions, 236 Insertions (4,614 errors; 27.39% KenLM WER).
  - Corrected Canonical 16 kHz Protocol: 3,365 Substitutions, 516 Deletions, 240 Insertions (4,121 errors; 24.46% KenLM WER).
  - Net Delta: -493 errors (concentrated 100% in the 431 Bolbosh files; 570 16k files bitwise identical with 0 changes).

---

## 4. Figure 4: N-Best Candidate Oracle WER Ceiling & Headroom
- **Generated Files**: `fig4_oracle_headroom.png`, `.svg`, `.pdf`
- **Data Source Files**:
  - `results/oracle_analysis.md`
  - `kashmiri_dataset/experiments/stage21a_post_b02c_reconstruction/reports/stage21a_final_research_report.md`
- **Underlying Metrics**:
  - $N=1$: 24.46% WER (4,121 errors)
  - $N=4$: 21.54% WER (3,629 errors; -492 errors)
  - $N=8$: 20.26% WER (3,412 errors; -709 errors)
  - $N=16$: 19.23% WER (3,240 errors; -881 errors)
  - $N=32$: 18.51% WER (3,118 errors; -1,003 errors)
  - $N=64$: 18.32% WER (3,086 errors; -1,035 errors)

---

## 5. Figure 5: Canonical Acoustic Checkpoint Leaderboard
- **Generated Files**: `fig5_acoustic_leaderboard.png`, `.svg`, `.pdf`
- **Data Source Files**:
  - `results/canonical_acoustic_results.csv`
- **Underlying Metrics**:
  - Ranks 1–10 audited checkpoints evaluated on `dev_assessment.jsonl` under `EduCasheer-Kashmiri-ASR-Eval-v2`.

---

## 6. Figure Generation Script
The master script reproducing all figures is saved in:
`figures/figure_generation_scripts/generate_all_figures.py`
Execution command: `python3 figures/figure_generation_scripts/generate_all_figures.py`
