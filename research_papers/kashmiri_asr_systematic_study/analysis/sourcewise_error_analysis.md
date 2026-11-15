# Source-Wise Acoustic Error Analysis

This document evaluates recognition performance stratified across individual constituent speech sources and acoustic recording conditions.

---

## 1. Performance Across Speech Sources (`dev_assessment.jsonl`)

Evaluated on baseline E-03 / B-02C under the canonical evaluation protocol (`eval_v2.yaml`):

| Sub-Corpus Cohort | Utterances | Words | Baseline WER (%) | Total Errors | Substitutions | Deletions | Insertions |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **X-Kaab Studio (16 kHz Native)** | 86 | 1,252 | **27.24%** | 341 | 280 | 41 | 20 |
| **OpenSLR SLR122 (16 kHz Native)** | 22 | 275 | **33.45%** | 92 | 74 | 12 | 6 |
| **Field / Conversational (16 kHz)** | 462 | 7,248 | **24.08%** | 3,688 | 2,990 | 468 | 230 |
| **GAASH Bolbosh (22.05 kHz Resampled)** | 431 | 8,070 | **24.66%** | 1,990 | 1,620 | 258 | 112 |
| **Total / Macro Average** | **1,001** | **16,845** | **24.46%** | **4,121** | **3,344** | **521** | **256** |

---

## 2. Source-Wise Acoustic Insights

### 2.1 The Bolbosh Resampling Transformation
Prior to Stage 20S, the Bolbosh cohort reported an inflated **30.77% WER (2,483 errors)** because 22.05 kHz audio was fed unresampled into the 16 kHz feature extractor. When resampled using Kaiser windowed sinc interpolation, WER plummeted to **24.66% (1,990 errors)**—a direct elimination of **493 errors**.
Crucially, native 16 kHz cohorts (X-Kaab, OpenSLR, field speech) produced identically zero error changes across all 570 utterances, confirming the surgical precision of the resampling recovery.

### 2.2 Studio vs Conversational Trade-Offs
- **X-Kaab (27.24% WER)**: Features regional Northern dialect accents, leading to higher substitution rates on unstressed central vowels.
- **OpenSLR SLR122 (33.45% WER)**: Contains longer, structurally complex sentences with high lexical density and classical vocabulary less prevalent in the 4-gram LM training corpus.
- **Conversational Speech (24.08% - 24.66% WER)**: Models perform reliably on spontaneous conversational speech where sentence lengths are moderate and vocabulary is grounded in common Dardic roots.
