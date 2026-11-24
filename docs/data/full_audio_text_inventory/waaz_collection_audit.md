# WAAZ Community Kashmiri Speech Collection — Forensic Audit Report
**Corpus Key**: `SRC_COMM_WAAZ`  
**Database Path**: `/home/musa/Desktop/Projects/DataCollection/data/datacollect.db`  
**Physical Audio Storage**: `/home/musa/Desktop/Projects/DataCollection/data/raw/contributions/`  
**Standardized Audio Storage**: `/home/musa/Desktop/Projects/DataCollection/data/processed/audio/16k_mono/`  
**Audit Date**: November 17, 2026  
**Auditor**: Principal Speech Dataset Researcher & Computational Linguist  

---

## 1. Executive Summary

WAAZ is the EduCasheer crowdsourced Kashmiri speech collection platform, engineered to capture native, multi-dialectal, conversational, and read speech from local speakers across all districts of Jammu & Kashmir.

Improving conversational Kashmiri ASR requires real community speech because existing legacy datasets (e.g., Bolbosh, OpenSLR) are heavily biased toward read studio speech and formal broadcast registers.

### Key Numerical Findings

| Metric | Measured Value | Forensic Evidence |
| :--- | :---: | :--- |
| **Total Physical Audio Files on Disk** | **2,944 files** | 1,472 raw WebM/WAV uploads + 1,472 16kHz Mono standardized WAVs. |
| **Total Live Database Recordings** | **1,453 unique recordings** | `recordings` table in `datacollect.db` (excluding 27,000 seeded external benchmark imports). |
| **Total Recorded Audio Duration** | **3.989 hours (14,361.8 s)** | Sum of `duration_seconds` across all native community contributions. |
| **Average Recording Duration** | **9.49 seconds** | Typical sentence or short phrase reading length. |
| **Total Registered Native Speakers** | **53 contributors** | Distinct `speaker_id` values in community `contributions`. |
| **District Geographic Spread** | **3 active districts** | Srinagar (30 speakers), Baramulla (19 speakers), Pulwama (1 speaker), Unspecified (3 speakers). |
| **Total Used in E22-01 Champion Training** | **501 utterances (1.070 hrs)** | Tier A high-confidence train partition (`train_waaz_tier_a.jsonl`). |
| **Total Held Out in Dev B Benchmark** | **90 utterances (0.140 hrs)** | Zero-speaker-overlap validation split (`dev_b_waaz_val.jsonl`). |
| **Total Remaining Unused Community Speech** | **862 utterances (2.779 hrs)** | Tier B (130 utts / 0.29h), Tier C (54 utts / 0.12h), Tier D (238 utts / 1.21h), plus ~440 newly contributed unreviewed utterances. |

---

## 2. Quality Tier Distribution & Audit Findings

WAAZ community contributions are subjected to multi-stage signal and phonetic validation:

```
[Community Upload (1,453 clips / 3.99h)]
       │
       ├──> Tier A: High Confidence (591 clips / 1.21h, CER <= 0.30, Approved)
       │         ├──> E22-01 Training Set: 501 clips (1.07h)
       │         └──> Dev B Held-Out Val: 90 clips (0.14h)
       │
       ├──> Tier B: Moderate Quality (167 clips / 0.35h, CER <= 0.55, Approved)
       │         └──> Included in experimental E22-02, but NOT in E22-01
       │
       ├──> Tier C: Acoustic Noise / Accent Deviation (54 clips / 0.12h)
       │         └──> Quarantined awaiting speech enhancement / denoiser
       │
       ├──> Tier D: Signal Defects / Clipping / Silence (238 clips / 1.21h)
       │         └──> Rejected (RMS < 0.005 or clipping ratio > 0.05)
       │
       └──> Tier E: Recent Unreviewed Backlog (~403 clips / ~1.06h)
                 └──> Contributed Nov 11–17, awaiting validation in web portal
```

### Forensic Tier Table

| Quality Tier | Total Recordings | Total Hours | Distinct Speakers | Mean SNR (dB) | Validation Status | Recommendation |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Tier A (Pristine)** | 591 | 1.207h | 35 | 24.12 dB | Double-human verified | 501 trained in E22-01; 90 preserved as Dev B val. |
| **Tier B (Good)** | 167 | 0.354h | 27 | 18.65 dB | Human approved | **Immediately ready for E22 continuation training.** |
| **Tier C (Noisy)** | 54 | 0.121h | 20 | 11.40 dB | Single verification | Requires spectral subtraction or Demucs filtering. |
| **Tier D (Defective)** | 238 | 1.213h | 36 | 6.82 dB | Rejected by QA | Unusable (silence, cutoffs, microphone distortion). |
| **Recent Backlog** | ~403 | ~1.060h | 18 | 19.80 dB | Pending review | **Execute human validation campaign in DataCollection UI.** |

---

## 3. Demographics, Modality, and Campaign Analysis

### Campaign Modality
1. **Kashmiri 100: Community Speech Challenge** (`kashmiri-100`):
   - Modality: `SCRIPTED_READING`
   - Purpose: Multi-district phonetic balance covering all Kashmiri vowel phonemes.
   - Recordings: 1,372 utterances.
2. **Voices of Every District** (`every-district`):
   - Modality: `SCRIPTED_READING`
   - Purpose: North Kashmir (Kupwara, Baramulla) vs. South Kashmir (Anantnag, Pulwama) dialect capture.
3. **Everyday Spontaneous Kashmiri** (`everyday-spontaneous`):
   - Modality: `SPONTANEOUS_CONVERSATION`
   - Purpose: Capturing unscripted memories, village folklore, culinary traditions, and natural speech.
   - Recordings: 27 utterances (0.410 hours, avg duration: 54.7 seconds).
4. **Kashmiri Lexicon & Vocabulary** (`kashmiri-lexicon`):
   - Modality: `WORD_LEXICON`
   - Purpose: Isolated phonetic articulation of Kashmiri words and proverbs.

### Critical Speaker Analysis
The collection features significant contributor concentration:
- Top speaker `SPK_KAS_D5586E` (Srinagar): 285 recordings (48.3 minutes).
- Speaker `SPK_KAS_GUEST_B93B` (Srinagar): 30 spontaneous recordings (25.0 minutes).
- Speaker `SPK_KAS_5747F8` (Female, 18–29, Baramulla / Wagoora): 91 recordings (13.4 minutes).
- Speaker `SPK_KAS_E37C27` (Male, 30–45, Pulwama): 59 recordings (8.7 minutes).

> [!WARNING]
> **Prompt-Reading Deviation Warning:**  
> A text prompt cannot automatically be treated as a verbatim transcript of what a contributor spoke. Our audit revealed that contributors occasionally substituted colloquial dialect words (e.g., reading *yem* as *yim*, or omitting clitic particles like *hye*). All prospective training additions must be validated against human reviewer audio adjudication rather than raw prompt strings.

---

## 4. Recommendations for WAAZ Data Incorporation

1. **Incorporate Tier B Immediately**: The 130 approved Tier B utterances (0.291 hrs) provide verified phonetic diversity and tested successfully in stage 22 (`DATA-22B`).
2. **Adjudicate the 403-Utterance Backlog**: Running a 2-hour native review session in the existing DataCollection UI (`/home/musa/Desktop/Projects/DataCollection`) will unlock approximately **0.85 hours of pristine Tier A Kashmiri speech**.
3. **Expand Spontaneous Speech Collections**: The 27 spontaneous recordings show high conversational realism. Expanding this campaign is the single highest-value action for conversational ASR improvement.
