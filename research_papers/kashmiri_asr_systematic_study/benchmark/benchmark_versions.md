# Kashmiri ASR Benchmark Versions and Iterative Refinements

This document chronicles the evolution of benchmark releases, detailing the differences between Benchmark v1.0 and Benchmark v1.1.

---

## 1. Version Comparison Overview

| Metric / Dimension | Benchmark v1.0 (Original Release) | Benchmark v1.1 (Audited Release) | Rationale for Revision |
|:---|:---:|:---:|:---|
| **Release Date** | November 14, 2026 (13:30) | November 14, 2026 (16:22) | Full audit and sample quarantine |
| **Directory Path** | `.../benchmark_construction/releases/v1_0_original/` | `.../benchmark_construction/releases/v1_1_corrected/` | Versioned isolation |
| **Total Audio Candidates** | 8,917 recordings | 6,315 verified recordings | Elimination of low-quality/infeasible samples |
| **Excluded Samples** | 0 (Uncurated release) | **2,602 quarantined samples** | Documented in `excluded_samples.jsonl` |
| **Tier A Samples** | 4,226 utterances | 3,742 utterances | Excluded short bursts (< 0.6s) & low SNR |
| **Tier B Samples** | 1,820 utterances | 1,480 utterances | Removed ambiguous Urdu code-switching |
| **Tier C Samples** | 2,871 utterances | 1,093 utterances | Deduplicated verse repeats & Arabic-only phrases |
| **Audio Verification** | Unverified container headers | **100% SHA-256 audited** | Cryptographic verification (`SHA256SUMS`) |
| **Canonical Resampling** | Inconsistent loading | **Kaiser sinc 16 kHz mandated** | Resolves sample-rate distortion |

---

## 2. Detailed Root Causes of Sample Exclusion in v1.1

The 2,602 samples excluded during the transition to v1.1 fall into four audited categories:

1. **Acoustic Noise Violations (1,422 samples)**:
   Recordings exhibiting SNR below 10.0 dB where the acoustic energy of speech was obliterated by background automotive noise, domestic chatter, or heavy electrical hum.
2. **Language Ambiguity / Non-Kashmiri (904 samples)**:
   Recordings where the speaker uttered pure standard Urdu or Hindi sentences without Kashmiri phonology, grammar, or lexical morphemes. Retaining these would evaluate cross-lingual Urdu ASR rather than Kashmiri ASR.
3. **CTC Length Violations (94 samples)**:
   Clips where speaking rate exceeded 15 characters per second, causing the tokenized text length to exceed available 50 Hz acoustic frames ($L > T$).
4. **Duration Boundary Violations (182 samples)**:
   Clips with duration $< 0.6$ seconds (isolated breathing sounds, false microphone triggers) or $> 25.0$ seconds.

---

## 3. Benchmark Manifest Artifacts in v1.1

The audited release resides in `kashmiri_dataset/experiments/benchmark_construction/releases/v1_1_corrected/manifests/`:
- `benchmark_v1_1_primary.jsonl`: The core 6,315-sample unified benchmark.
- `tier_a.jsonl`: Clean studio and near-field conversational benchmark.
- `tier_b.jsonl`: Field and noisy conversational benchmark.
- `tier_c.jsonl`: Formal religious and classical translation benchmark.
- `excluded_samples.jsonl`: The 2,602 quarantined samples with exclusion rationale metadata.
- `pilot_corrected.jsonl`: A 150-sample stratified diagnostic pilot set.
