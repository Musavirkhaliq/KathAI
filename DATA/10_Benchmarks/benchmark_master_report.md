# Kashmiri ASR Benchmark Forensic Audit & Master Evaluation Report

## 1. Executive Summary

The EduCasheer benchmark suite evaluates Kashmiri speech recognition across multi-speaker, spontaneous community, and scriptural domains.

## 2. Benchmark Corpus Statistics

| Benchmark Release / Tier | Clips | Duration (Hours) | Speakers | Words | Characters | Primary Domains |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Frozen Test Set (v1 Clean)** | 866 | 1.21h | 40 | 11,817 | 63,311 | `xkaab_asr_4hrs, mendeley_ktts, programindz_audio_corpus, openslr_slr122` |
| **Frozen Test Set (v1 Master)** | 928 | 1.27h | 41 | 12,400 | 66,297 | `xkaab_asr_4hrs, mendeley_ktts, programindz_audio_corpus, openslr_slr122` |
| **Frozen Validation Set (v1)** | 945 | 1.45h | 28 | 12,970 | 70,337 | `xkaab_asr_4hrs, mendeley_ktts, programindz_audio_corpus, openslr_slr122` |
| **Benchmark Tier A (IndicVoices-R)** | 3,731 | 9.19h | 60 | 91,091 | 427,664 | `indicvoices_r_quarantined_test` |
| **Benchmark Tier B (WAAZ Online)** | 668 | 1.38h | 43 | 7,616 | 38,111 | `DataCollection_Online` |
| **Benchmark Tier C (Quran King Fahd)** | 1,916 | 2.39h | 2 | 18,436 | 88,010 | `Quran_King_Fahd` |

## 3. Tier-Wise Ground Truth Provenance & Forensic Analysis

### Tier A (Speaker Generalization — IndicVoices-R Quarantined)
- **Clips**: 3,731 clips (9.19 hours, 60 speakers)
- **Ground Truth**: Human-verified field transcriptions from AI4Bharat IndicVoices-R.
- **Isolation**: Quarantined from all acoustic and language model training splits.

### Tier B (Real-World Spontaneous Kashmiri — EduCasheer Online)
- **Clips**: 668 clips (1.38 hours, 43 speakers)
- **Ground Truth**: Native crowdsourced recordings with verified prompt matching.
- **Isolation**: Completely held out and unreleased.

### Tier C (Domain Robustness — King Fahd Quranic Translation)
- **Clips**: 1,916 clips (2.39 hours, 2 speakers)
- **Forensic Audit Finding**: In early unverified versions, some clips contained Arabic recitation or model-predicted references. In this audit, all 1,916 clips have been audited to ensure pure Kashmiri translation references.
