# Kashmiri ASR Master Experiment Results & Validated Leaderboard

**Project**: EduCasheer Kashmiri Automatic Speech Recognition Research
**Standardized Evaluation Benchmark**: `dev_assessment.jsonl` (1,001 utts, 16,845 words) & `val.jsonl` (6,728 utts, 104,685 words)
**Audit Status**: Fully Reconciled Post-Stage 20R/20S Forensic Audit and Stage 24 Certification

---

## 1. Primary Canonical Leaderboard (`dev_assessment.jsonl`, 16,845 words)

| Rank | Model ID | Research Stage | Architecture | Greedy WER (%) | KenLM WER (%) | CER (%) | Word Errors | Scientific Status |
|:---:|:---|:---|:---|:---:|:---:|:---:|:---:|:---|
| 1 | **E22-01** | Stage 22A | XLS-R 300M + Char CTC | 26.60% | **24.40%** | 11.15% | 4,110 | `VERIFIED (Current Production SOTA)` |
| 2 | **E-03** | Stage 15 | XLS-R 300M + Char CTC | 26.68% | **24.46%** | 11.19% | 4,121 | `VERIFIED (Co-Champion)` |
| 3 | **E-04** | Stage 15 | XLS-R 300M + Char CTC | 26.67% | **24.46%** | 11.21% | 4,121 | `VERIFIED` |
| 4 | **B-02C** | Stage 9 | XLS-R 300M + Char CTC | 26.76% | **24.47%** | 11.23% | 4,122 | `VERIFIED (Foundation Baseline)` |
| 5 | **E-02** | Stage 15 | XLS-R 300M + Char CTC | 26.62% | **24.47%** | 11.21% | 4,122 | `VERIFIED` |
| 6 | **S16-05** | Stage 16 | XLS-R 300M + Char CTC | 26.77% | **24.47%** | 11.24% | 4,122 | `VERIFIED` |
| 7 | **WA-02** | Stage 14 | XLS-R 300M + Char CTC | 26.63% | **24.51%** | 11.22% | 4,129 | `VERIFIED` |
| 8 | **E-01** | Stage 15 | XLS-R 300M + Char CTC | 26.63% | **24.52%** | 11.18% | 4,130 | `VERIFIED` |
| 9 | **S16-02** | Stage 16 | XLS-R 300M + Char CTC | 26.67% | **24.52%** | 11.23% | 4,131 | `VERIFIED` |
| 10 | **MWER-01** | Stage 14 | XLS-R 300M + Char CTC | 26.77% | **24.56%** | 11.21% | 4,137 | `VERIFIED` |
| 11 | **S17B-EXP-A** | Stage 17B | XLS-R 300M + Char CTC | 26.66% | **24.57%** | 11.24% | 4,138 | `VERIFIED` |
| 12 | **U-02** | Stage 13 | XLS-R 300M + Char CTC | 27.15% | **24.91%** | 11.44% | 4,196 | `VERIFIED (Negative Result)` |
| 13 | **I-01** | Stage 13 | XLS-R 300M + Char CTC | 27.41% | **25.02%** | 11.59% | 4,215 | `VERIFIED (Negative Result)` |
| 14 | **P17-PL** | Stage 17 | XLS-R 300M + Char CTC | 27.75% | **25.25%** | 11.68% | 4,253 | `VERIFIED (Negative Result)` |
| 15 | **B-02A** | Stage 9 | XLS-R 300M + Char CTC | 28.11% | **25.71%** | 11.93% | 4,331 | `VERIFIED` |
| 16 | **R-02** | Stage 12 | XLS-R 300M + Char CTC | 30.47% | **28.94%** | 12.58% | 4,875 | `VERIFIED (Negative Result)` |

---

## 2. Multi-Partition Generalization Leaderboard (Stages 22A - 24)

| Model ID | Stage | Evaluation Benchmark | Utterances / Words | Greedy WER (%) | KenLM WER (%) | Errors | Scientific Significance |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---|
| **E22-01** | Stage 22A | `dev_calibration.jsonl (1,001 utts)` | — | 26.82% | **24.74%** | 4,114 | Certified calibration benchmark |
| **E22-01** | Stage 22A | `dev_a_disjoint.jsonl (6,257 utts)` | — | 27.20% | **24.94%** | 26,566 | Disjoint development holdout (106,530 words) |
| **E22-01** | Stage 22A | `dev_b_waaz_val.jsonl (127 utts)` | — | 48.26% | **46.12%** | 582 | Held-out conversational community speech |
| **E24-CTC-01** | Stage 24 | `dev_b_waaz_val.jsonl (127 utts)` | — | 52.38% | **48.42%** | 611 | Acoustic continuation: greedy improved (+0.16pp), KenLM degraded |
| **CARE-ASR** | Stage 20 | `dev_assessment (in-memory)` | — | 26.50% | **24.53%** | 4,132 | Step 0 serialization failure; weights not written to disk; evaluated in-memory model |

---

## 3. Foundation Scaling & External Baseline Leaderboard (`val.jsonl`, 104,685 words)

| Rank | Model ID | Stage / Provider | Parameters | Training Data | Greedy WER (%) | KenLM WER (%) | CER (%) | Relative WERR vs B-01 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **B-02C** | Stage 9 | 315.6M | 165.32 hrs | **27.84%** | **25.24%** | **10.96%** | **-17.78%** |
| 🥈 | **B-02B** | Stage 9 | 315.6M | 123.45 hrs | 29.39% | 26.41% | 11.61% | -13.20% |
| 🥉 | **B-02A** | Stage 9 | 315.6M | 104.93 hrs | 29.42% | 26.30% | 11.66% | -13.11% |
| 4 | **B-01** | Stage 8 | 315.6M | 92.94 hrs | 33.86% | 31.80% | 13.08% | Baseline Reference |
| 5 | **Muzaffar786** | External Community | 315M | Public | 49.80% | 48.10% | 21.60% | +51.3% error |
| 6 | **Khushi v6** | External Community | 315M | Public | 52.40% | 50.80% | 23.10% | +59.7% error |
| 7 | **IndicConformer** | AI4Bharat | 600M | Multilingual | 62.14% | 60.50% | 28.32% | +90.3% error |
| 8 | **Whisper Large-v3**| OpenAI | 1,550M | Multilingual | 84.12% | 84.12% | 42.18% | +164.5% error (Catastrophic Hallucination) |

---

## 4. Search Space Topology & Decoder Oracle Bounds

| Decoding Configuration | Beam ($W$) | $\alpha$ (LM) | $\beta$ (WIP) | WER (%) | Total Errors | Substitutions | Deletions | Insertions | Headroom vs Baseline |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **KenLM Frozen Baseline** | 16 | 0.35 | -0.50 | 24.46% | 4,121 | 3,344 | 521 | 256 | Baseline (0 errors) |
| **KenLM Calibrated Champion** | 16 | 0.20 | -0.50 | 24.30% | 4,093 | 3,369 | 440 | 284 | -28 word errors |
| **Assessment Optimized KenLM** | 16 | 0.20 | 0.00 | 24.27% | 4,089 | 3,365 | 435 | 289 | -32 word errors |
| **Candidate Relative Ranker** | 16 | 0.20 | -0.50 | 24.30% | 4,093 | 3,369 | 440 | 284 | -28 word errors |
| **Linear Feature Ranker** | 16 | 0.20 | -0.50 | 24.29% | 4,092 | 3,368 | 440 | 284 | -29 word errors |
| **Word-Level MBR ($T=1.0$)** | 16 | 0.35 | -0.50 | 24.43% | 4,115 | 3,401 | 424 | 290 | -6 word errors |
| **Oracle $N=16$ Ceiling** | 16 | — | — | **19.23%** | **3,240** | 2,560 | 418 | 262 | **-881 word errors (21.4% recoverable headroom)** |
| **Oracle $N=32$ Ceiling** | 32 | — | — | **18.99%** | **3,158** | 2,480 | 412 | 266 | **-963 word errors (23.4% recoverable headroom)** |

---

## 5. Invalidated & Methodologically Flawed Experiments

| Experiment ID | Stage | Claimed Metric | Flaw / Violation Discovered | Scientific Evidence | Current Status |
|:---|:---:|:---:|:---|:---|:---|
| **CARE-ASR** | Stage 20 | 24.53% WER (4,132 errors) | Checkpoints not serialized to disk; evaluation run on in-memory representations. | Audit in Stage 20R verified empty checkpoint directory on disk. | `INVALIDATED` |
| **MWER Initial Claim** | Stage 15 | 21.68% (tolerant) / 27.39% WER | Bolbosh audio was evaluated without 16kHz resampling, generating 493 phantom errors on B-02C. | Stage 20S re-evaluation proved Delta = 1 error across 16,845 words. | `RECONCILED (ILLUSION)` |
| **Stage 14 Cosine Sim** | Stage 14 | Cosine similarity > 1.0000 | Float32/BF16 accumulation round-off error violated Cauchy-Schwarz bound. | Stage 15 FP64 double-precision accumulator strictly bounds similarity in [-1, 1]. | `CORRECTED` |
| **Stage 20T Extraction**| Stage 20T | 84-word error penalty | Narrow beam width (W=8) caused premature history pruning. | Stage 21A beam expansion (W=16) restored 100% exact parity. | `REPAIRED` |
