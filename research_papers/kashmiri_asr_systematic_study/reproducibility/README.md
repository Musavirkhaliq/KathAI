# EduCasheer Kashmiri ASR Reproducibility Package

Welcome to the independent scientific replication package for the EduCasheer Kashmiri Automatic Speech Recognition systematic study.

This package allows any researcher to:
1. Recompute all benchmark tables, statistical tests, and metrics from saved predictions.
2. Regenerate all publication-quality vector and raster figures.
3. Audit physical checkpoint files and cryptographic manifests on disk.
4. Execute full acoustic model inference or retraining pipelines on the NVIDIA DGX Spark GB10.

---

## 1. Quick-Start Reproduction

To run the complete verification suite:
```bash
cd "/home/musa/Kashmiri Project"
./research_papers/kashmiri_asr_systematic_study/reproducibility/reproduce_paper_results.sh --mode all
```

Individual operational modes:
- **`--mode metrics`**: Recomputes all canonical leaderboard tables, error breakdowns, and statistical $p$-values. Runs in $< 2$ seconds without requiring GPU execution.
- **`--mode figures`**: Reruns the matplotlib rendering scripts, regenerating all PNG, SVG, and PDF figures in `figures/`. Runs in $< 5$ seconds.
- **`--mode audit`**: Scans disk paths for all 23 historical checkpoints, verifying file presence and formats.

---

## 2. Directory Layout of the Replication Package

```
research_papers/kashmiri_asr_systematic_study/
├── evidence/                  # Repository inventories, checkpoint registry, claims matrix
├── history/                   # Complete stage-by-stage ledgers and dependency graphs
├── data/                      # Dataset cards, source inventories, and licensing
├── benchmark/                 # Benchmark construction, protocols, and version audits
├── models/                    # Architecture configs, hyperparameter matrices, loss mathematics
├── results/                   # Canonical results, decoder tables, historical reconciliation
├── analysis/                  # Error taxonomy, phonetic confusion matrices, remaining challenges
├── literature/                # Prior work review, BibTeX references, citation verification
├── figures/                   # Rendered figures (PNG, SVG, PDF) and generation scripts
├── paper/                     # Complete manuscripts (Markdown & LaTeX), abstract, appendix
├── reproducibility/           # Reproduction scripts, environment configs, and manifests
└── publication/               # Readiness assessments, venue options, submission checklists
```

---

## 3. Computational Requirements

- **Lightweight Reproduction (Metrics & Figures)**: Standard CPU, Python 3.10+, `pandas`, `numpy`, `matplotlib`.
- **Acoustic Inference Reproduction**: NVIDIA GPU with $\ge 12$ GB VRAM (or DGX Spark GB10), PyTorch 2.6.0+, CUDA 12.8+, `torchaudio`, `transformers`, `pyctcdecode`, `kenlm`.
- **Full Model Retraining**: NVIDIA DGX Spark (128GB Unified Memory), 20–40 hours per full training run.
