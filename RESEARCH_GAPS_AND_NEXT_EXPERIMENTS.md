# Research Gaps and Prioritized Future Experiments

This document identifies the smallest, most impactful set of targeted experiments required to strengthen the scientific validity of the EduCasheer Kashmiri ASR systematic study prior to top-tier A* conference or archival journal submission.

---

## 1. Prioritized Experiment Roadmap

```
┌──────┬───────────────────────────────────┬──────────────┬─────────────┬────────────────────────────────────┐
│ Rank │ Experiment Title                  │ Priority     │ Est. Compute│ Target Scientific Contribution     │
├──────┼───────────────────────────────────┼──────────────┼─────────────┼────────────────────────────────────┤
│ 1    │ Blind Evaluation on Benchmark v1.1│ Mandatory    │ 2.5 hours   │ Eliminates dev_assessment threat   │
│ 2    │ Supervised Whisper-Large-v3 Fine-T│ High         │ 16 hours    │ Closes zero-shot baseline critique │
│ 3    │ In-Domain IndicConformer Baseline │ High         │ 12 hours    │ Compares Conformer vs XLS-R        │
│ 4    │ Power-Law Acoustic Scaling Ablat. │ Medium       │ 24 hours    │ Quantifies data efficiency curves  │
│ 5    │ 3-Seed Training Variance Audit    │ Medium       │ 35 hours    │ Establishes acoustic error bars    │
└──────┴───────────────────────────────────┴──────────────┴─────────────┴────────────────────────────────────┘
```

---

## 2. Detailed Experiment Protocols

### Experiment 1: Blind Evaluation on Sealed Benchmark v1.1
- **Current Limitation**: The 1,001-utterance manifest `dev_assessment.jsonl` was examined repeatedly during diagnostic audits in Stages 11–21B.
- **Protocol**:
  - Load the untouched 6,315-sample manifest `benchmark_v1_1_primary.jsonl`.
  - Evaluate locked checkpoints:
    1. Foundation Acoustic Baseline: B-02C (`beam_width=16`, $\alpha=0.35, \beta=-0.50$).
    2. Calibrated KenLM Champion: B-02C / E-03 ($\alpha=0.20, \beta=-0.50$).
    3. Candidate-Relative Sequence Ranker (312k params).
  - Compute WER, CER (with and without space), and 10,000 bootstrap paired confidence intervals.
- **Expected Impact**: Provides definitive proof of held-out generalization on untouched speakers, satisfying the most stringent reviewer demands.

### Experiment 2: Supervised Fine-Tuning of Whisper-Large-v3
- **Current Limitation**: The manuscript audits Whisper in a zero-shot capacity, showing $>84\%-107\%$ WER due to the lack of an explicit `<|ks|>` token. Reviewers may argue that supervised fine-tuning could unlock Whisper's 1.5B parameters.
- **Protocol**:
  - Adapt `openai/whisper-large-v3` by adding a native Kashmiri token `<|ks|>` or mapping to an unused language token.
  - Fine-tune on the identical 165.32h B-02C training corpus using LoRA ($r=16$) or full encoder-decoder optimization for 10 epochs.
  - Evaluate on Benchmark v1.1 and `dev_assessment.jsonl`.
- **Expected Impact**: Establishes whether an autoregressive sequence-to-sequence foundation model can outperform or match our 315M CTC acoustic model (24.46% WER).

### Experiment 3: In-Domain Fine-Tuning of IndicConformer
- **Current Limitation**: Published IndicConformer numbers (41.20% / 66.59% WER) were trained on AI4Bharat's original data mixture.
- **Protocol**:
  - Fine-tune AI4Bharat's 115M Conformer CTC on our standardized 165.32h corpus with diacritics retained.
  - Evaluate using pyctcdecode and the domain KenLM.
- **Expected Impact**: Isolates the architectural effect of Conformer vs XLS-R Transformer on identical speech data.

### Experiment 4: Corpus-Size Down-Sampling Ablation
- **Objective**: Subsample the training corpus to 10h, 25h, 50h, 100h, and 165h.
- **Analysis**: Fit power-law scaling coefficients $\text{WER}(H) = \alpha H^{-\beta} + \gamma$ to predict data volume requirements for reaching sub-15% WER.

### Experiment 5: Multi-Seed Stability Verification
- **Objective**: Retrain B-02A and B-02C across seeds 42, 123, and 777.
- **Analysis**: Report mean $\pm$ standard deviation across seeds to prove that acoustic scaling results are not artifacts of seed initialization.
