# Results Integrity and Scientific Audit Report

This report consolidates the findings of the reproducibility audit, certifying which results are validated, which are corrected, and which are invalidated.

---

## 1. Certified Scientific Status of Project Claims

```
┌──────────────────────────────────────┬───────────────────────────────┬────────────────────────────┐
│ Empirical Finding                    │ Historical Claim              │ Certified Audit Verdict    │
├──────────────────────────────────────┼───────────────────────────────┼────────────────────────────┤
│ B-01 -> B-02C Scaling                │ -6.02% WER reduction          │ VERIFIED (p < 0.0001)      │
│ Foundation B-02C Audio Processing    │ Verified 16 kHz training      │ VERIFIED (torchaudio sinc) │
│ E-03 vs B-02C Superiority            │ E-03 ~3% better than B-02C    │ CONTRADICTED (Identical)   │
│ Stage 20 CARE-ASR Improvement        │ 482 errors eliminated by CARE │ CONTRADICTED (Resampling)  │
│ Stage 20 CARE-ASR Checkpoints        │ Checkpoint saved on disk      │ INVALIDATED (Missing)      │
│ Sample Rate Resampling Impact        │ None (Unnoticed)              │ VERIFIED (-493 errors)     │
│ Beam 8 Search Pruning Penalty        │ Attributed to rescorer        │ VERIFIED (+84 search errs) │
│ CER 9.03% vs 11.19% Divergence       │ Model drift                   │ VERIFIED (Denominator bug) │
│ Stage 21B Decoder Calibration        │ Modest error reduction        │ VERIFIED (-28 to -32 errs) │
│ Stage 21B Statistical Significance   │ Claimed p < 0.01              │ CONTRADICTED (p = 0.0847)  │
└──────────────────────────────────────┴───────────────────────────────┴────────────────────────────┘
```

---

## 2. Core Integrity Takeaways

1. **Preprocessing Bugs Mimic Architectural Breakthroughs**: The accidental passage of 22,050 Hz audio into 16 kHz Wav2Vec2 processors in Stages 15–19 created an artificial ~3% WER distortion. Correcting this bug eliminated 493 errors without modifying a single neural weight. Preprocessing integrity must be treated as a first-class scientific variable.
2. **The Perils of Step-Zero Assumptions**: Stage 20 reported positive results from an adapter that never saved its weights, evaluating an in-memory Step 100 model while logging `best_step = 0`. Full audit scripts must inspect file presence on disk.
3. **Statistical Humility in Decoder Rescoring**: Small error reductions (28–32 words across 16,845 words) frequently fail rigorous significance testing ($p = 0.0847$). Transparent reporting prevents misleading claims of breakthrough progress.
