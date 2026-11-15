# Experiment Dependency Graph

This document details the architectural and data lineage connecting all 21 research stages of the EduCasheer Kashmiri ASR project.

```mermaid
flowchart TD
    %% Phase 1: Data & Baseline
    RAW["Raw Speech Archives<br/>(Bolbosh 22k, IndicVoices 16k, OpenSLR 16k, XKaab 16k)"] --> PREPROC["Standardized Preprocessing<br/>(NFKC Normalization, 159-char CTC Vocab)"]
    PREPROC --> STAGE8["Stage 8: B-01 Baseline<br/>(92.9h, XLS-R 300M, WER: 33.86%)"]
    
    %% Phase 2: Acoustic Scaling
    STAGE8 --> B02A["Stage 9: B-02A<br/>(+12h Recovered Speech, 104.9h, WER: 29.42%)"]
    B02A --> B02B["Stage 9: B-02B<br/>(+18.5h Mirwaiz Quranic, 123.5h, WER: 29.39%)"]
    B02B --> B02C["Stage 9: B-02C (Champion Foundation)<br/>(+41.9h IndicVoices-R, 165.3h, WER: 27.84% / KenLM: 24.47%)"]
    
    %% Phase 3: Divergent Exploration from B-02C
    B02C --> STAGE12["Stage 12: Robustness Series<br/>(R-01 Reverb, R-02 Multi-Noise: Regressed to 28.94%)"]
    B02C --> STAGE13["Stage 13: Architecture Ablations<br/>(U-01..03 Unfreeze, CR-CTC, Inter-CTC: 24.91% - 25.23%)"]
    B02C --> STAGE14["Stage 14: MWER Pilots<br/>(MWER-01, WA-01..03 Weight Averaging: 24.51%)"]
    
    %% Phase 4: Stage 15 Continuation
    B02C --> STAGE15["Stage 15: MWER Scaling<br/>(E-01..E-04: E-03 achieves 24.46% KenLM WER)"]
    STAGE15 --> STAGE16["Stage 16: Model Merging<br/>(S16-01..05: S16-05 achieves 24.47%)"]
    STAGE15 --> STAGE17["Stage 17 / 17B: Noisy Audio & OTC<br/>(P17-PL, T17-CTC, S17B: Regressed to 25.25%)"]
    STAGE15 --> STAGE18["Stage 18: Decoder Intelligence & Atlas<br/>(Cataloged 4,614 errors on unresampled protocol)"]
    STAGE15 --> STAGE19["Stage 19: Phonetic Confusion Repair<br/>(Step 0 chosen; no weights saved)"]
    
    %% Phase 5: The Integrity Crisis & Root Cause
    STAGE15 --> STAGE20["Stage 20: CARE-ASR<br/>(Reported 24.53% WER; missing ckpts)"]
    STAGE20 --> STAGE20R["Stage 20R: Forensic Audit<br/>(Uncovered 22.05 kHz audio bug & Step-0 save bug)"]
    STAGE20R --> STAGE20S["Stage 20S: Audio Integrity Recovery<br/>(-493 errors on Bolbosh via Kaiser sinc 16k; clean speech 100% parity)"]
    
    %% Phase 6: Canonical Reconstruction & Final Frontier
    STAGE20S --> STAGE20T["Stage 20T: Sequence-Aware Rescoring<br/>(Beam 8 pruning penalty: +84 errors)"]
    STAGE20T --> STAGE21A["Stage 21A: Post-B-02C Reconstruction<br/>(Restored Beam 16: 100% parity; E-03 vs B-02C delta=1 error; Oracle: 19.23%)"]
    STAGE21A --> STAGE21B["Stage 21B: Calibration & Candidate Ranker<br/>(KenLM Calib: 24.27% / 4089 errs; CR-Ranker: 24.30% / 4093 errs)"]

    %% Styling
    classDef baseline fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef scaling fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef negative fill:#ffebee,stroke:#d32f2f,stroke-width:2px;
    classDef audit fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef champion fill:#f3e5f5,stroke:#7b1fa2,stroke-width:3px;

    class STAGE8 baseline;
    class B02A,B02B,B02C scaling;
    class STAGE12,STAGE13,STAGE16,STAGE17,STAGE19 negative;
    class STAGE20,STAGE20R,STAGE20S,STAGE20T,STAGE21A audit;
    class STAGE15,STAGE21B champion;
```

---

## Lineage Matrix

| Node ID | Parent Model / Ingestion | Training Data Source | Primary Dependent Stages |
|:---|:---|:---|:---|
| **B-01** | `facebook/wav2vec2-xls-r-300m` | 92.94h initial verified corpus | B-02A |
| **B-02A** | B-01 Checkpoint | B-01 + 12h recovered speech | B-02B |
| **B-02B** | B-02A Checkpoint | B-02A + 18.5h Mirwaiz translations | B-02C |
| **B-02C** | B-02B Checkpoint | B-02B + 41.9h IndicVoices-R | Stage 12, 13, 14, 15, 21A |
| **E-03** | B-02C Checkpoint | 100 steps MWER ($\lambda=0.10$) | Stage 16, 17, 18, 19, 20, 20R, 20S, 20T, 21A, 21B |
| **Stage 20R/S**| Audit of Stages 15–20 | `dev_assessment.jsonl` (1,001 utts) | Stage 20T, 21A, 21B |
| **Stage 21A** | Full Checkpoint Sweep | Audited all 30+ historical checkpoints | Stage 21B |
| **Stage 21B** | E-03 + Repaired 16-beam | `dev_calibration.jsonl` + 2,500 Tier A tuples | Final Manuscript Systems |
