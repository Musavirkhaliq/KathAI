# Required Additional Experiments Prior to Final A* Submission

This document defines the minimal, high-impact set of targeted experiments required to elevate the manuscript from a strong submission-ready draft to a top-tier A* publication.

---

## 1. High-Priority Experiments (Critical for A* Acceptance)

### Experiment 1: Blind Evaluation on the Sealed Benchmark v1.1
- **Objective**: Execute baseline B-02C, E-03, and the Calibrated KenLM / Candidate-Relative Ranker on the sealed 6,315-sample `benchmark_v1_1_primary.jsonl`.
- **Why Required**: Resolves the threat to validity concerning repeated diagnostic inspection of `dev_assessment.jsonl`. Proves that the 24.27%–24.46% accuracy generalizes to completely untouched speakers and domains.
- **Compute Budget**: ~2.5 hours on NVIDIA DGX Spark GB10. Zero retraining required.

### Experiment 2: Supervised Fine-Tuning of Whisper-Large-v3
- **Objective**: Fine-tune `openai/whisper-large-v3` on the identical 165.32h B-02C training corpus using LoRA or full encoder-decoder fine-tuning.
- **Why Required**: Proves whether the zero-shot failure of Whisper (>84% WER) is purely an architectural lack of a `<|ks|>` token, or if supervised Whisper can rival our XLS-R 300M CTC acoustic model (24.47% WER).
- **Compute Budget**: ~16 hours on DGX Spark GB10.

### Experiment 3: In-Domain Fine-Tuning of IndicConformer
- **Objective**: Fine-tune AI4Bharat's 115M Conformer CTC/RNN-T on our standardized 165.32h corpus with diacritics retained.
- **Why Required**: Directly compares the Wav2Vec2/XLS-R Transformer backbone against the popular Conformer architecture on the exact same data.
- **Compute Budget**: ~12 hours on DGX Spark GB10.

---

## 2. Medium-Priority Experiments (Strengthens Analysis)

### Experiment 4: Corpus Size Down-Sampling Ablation
- **Objective**: Train XLS-R CTC on 10h, 25h, 50h, 100h, and 165h subsets.
- **Why Desirable**: Formulates precise empirical power-law scaling coefficients for low-resource Dardic speech.
- **Compute Budget**: ~24 hours total.

### Experiment 5: Repeated-Seed Statistical Variance Audit
- **Objective**: Fine-tune B-02A or B-02C across three random seeds ($S \in \{42, 123, 777\}$).
- **Why Desirable**: Establishes error bars on the acoustic training trajectory, proving convergence stability.
- **Compute Budget**: ~35 hours.
