# Historical Evolution of Kashmiri ASR Test Partitions
**EduCasheer Kashmiri AI — Workstream 8, Stage 22C**  
*Principal Speech Recognition Scientist & Reproducibility Auditor*  
*Date: November 15, 2026*

---

## Chronological Evolution of Benchmarks

Across the progression of the EduCasheer project, the evaluation splits underwent several revisions:

1. **Stage 7–9 (Initial Native Baseline)**:
   - Initial 5,251-utterance split (`test.jsonl` early release), consisting of IndicVoices (2,731), Bolbosh (2,000), and Whisper (520).
   - Reported greedy WER on XLS-R 300M was ~34.5%.
2. **Stage 15 (E-03 Model & mWER Scaling)**:
   - E-03 checkpoint (`best_calib`) established as the project's leading acoustic model.
   - Historical calibrated KenLM decoding achieved ~24.30% on internal validation benchmarks.
3. **Stage 16 (Benchmark v1.0 & v1.1 Audit)**:
   - Evaluated external robustness across supplementary collections (OpenSLR 122 and XKaab), appending 672 utterances to create the 5,923-utterance comprehensive test partition.
4. **Stage 22 (WAAZ Data Integration & E22-01 Retraining)**:
   - Added WAAZ Tier A community recordings (501 utts, 1.07 h) to create `train_mixture_2_clean_plus_waaz_tier_a.jsonl` (97,165 utts, 166.39 h).
   - Initial evaluation reported a catastrophic KenLM "regression" (KenLM performing worse than greedy CTC).
5. **Stage 22B (Decoder Slicing & Initial Test Benchmarking)**:
   - Diagnosed the cause of the KenLM regression: batched audio zero-padding without temporal logit slicing.
   - Fixed temporal slicing using `round(N / 320.0)` and evaluated the four systems on the 5,923 test set.
   - Suffered GPU/host memory exhaustion due to the 546-second sermon audio in OpenSLR 122.
6. **Stage 22C (Inference Certification & Exact Calibration)**:
   - Proved mathematically that `round(N / 320.0)` retains 1 frame of zero-padded acoustic artifacts in padded batches.
   - Replaced approximation with exact Conv1d receptive-field calculation `compute_exact_ctc_length(N)`.
   - Verified that recordings $\le 25$s run uninterrupted, and recordings $> 25$s use Context-Preserving Central-Frame Extraction (Method C2).
   - Established the definitive, reproducible benchmark for EduCasheer Kashmiri ASR.
