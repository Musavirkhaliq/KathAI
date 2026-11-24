# Scientific Opportunity Ranking of Prospective Kashmiri Speech Data
**Project**: EduCasheer Kashmiri AI Research  
**Document Identification**: `EDU-KAS-RANK-2026-FINAL`  
**Execution Date**: November 17, 2026  
**Auditor**: Principal ASR Training Scientist & Computational Linguist  

---

## 1. Executive Summary & Ranking Framework

To maximize word error rate (WER) reduction on conversational and broadcast Kashmiri ASR, we must prioritize data quality, speaker diversity, and conversational relevance over raw dataset volume.

We evaluated all **unused, underused, or recoverable Kashmiri speech sources** across 10 empirical criteria:
1. **Verified Speech Duration (Hours)**
2. **Transcript Accuracy / Ground Truth Provenance**
3. **Conversational Realism & Spontaneity**
4. **Speaker & Gender Diversity**
5. **Dialect & Geographic Coverage (All 10 Valley Districts)**
6. **Acoustic Quality & Signal-to-Noise Ratio (SNR)**
7. **Contamination Risk (Benchmark Protection)**
8. **Alignment / Recovery Effort Cost**
9. **Licensing & Legal Commercialization Freedom**
10. **Acoustic Generalization Value (Out-of-Domain Robustness)**

---

## 2. Master Opportunity Ranking Table

| Rank | Candidate Data Source | Available Hours | Status / Readiness | Conversational Value | Recovery Cost | Contamination Risk | Primary Contribution to Model Generalization |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 🥇 **1** | **WAAZ Community Crowdsourced Speech (Tier B + Backlog)** | **2.63h** | High (Semi-ready) | ⭐⭐⭐⭐⭐ (Highest) | Very Low | Zero | Captures real-world smartphone microphone acoustics, native dialectal vowel shifts, and multi-generational speakers. |
| 🥈 **2** | **Meta Omnilingual Kashmiri Corpus (Train Partition)** | **7.41h** | Ready to Train | ⭐⭐⭐⭐ (High) | Zero | Zero | Pristine Perso-Arabic transcripts, studio prompt responses, zero benchmark overlap. |
| 🥉 **3** | **Mirwaiz Bayan-ul-Furqan Unaligned Commentary Recovery** | **15.11h** | Recoverable (Forced Align) | ⭐⭐⭐⭐ (High) | Medium | Zero | Natural unscripted Kashmiri narrative; massive expansion in classical vocabulary and discourse connectors. |
| **4** | **IndicVoices Kashmiri (Leak-Safe Extra Slice)** | **3.54h** | Ready to Train | ⭐⭐⭐⭐⭐ (Highest) | Zero | Zero | Spontaneous village conversations from rural valley districts; 100% leak-free. |
| **5** | **Gaash Bolbosh (Leak-Safe Extra Slice)** | **9.18h** | Ready to Train | ⭐⭐⭐ (Medium) | Zero | Zero | Studio & quiet room native read speech from IUST; high SNR phonetic regularization. |
| **6** | **Dr. War Gulistan News Commentary (Human Review)** | **3.81h** | Recoverable (Human Review) | ⭐⭐⭐⭐⭐ (Highest) | Medium | Zero | Natural television commentary; rapid speaking rate and extemporaneous discourse. |
| **7** | **Programindz / KTTS Clean Phrase Corpus** | **4.42h** | Ready to Train | ⭐⭐ (Low - Read TTS) | Zero | Zero | Studio single-speaker clean speech; excellent phonetic baseline anchor. |
| **8** | **King Fahd Spoken Commentary Recovery** | **1.08h** | Recoverable (ASR + Review) | ⭐⭐⭐ (Medium) | Low | Zero | Highest fidelity studio acoustics; covers complex Arabic-Kashmiri religious code-mixing. |
| **9** | **UChicago S. Hassan Dictionary Pronunciations** | **0.34h** | Lexicon Ready | ⭐ (Isolated Words) | Low | Zero | Rare archaic lexical terms; improves vocabulary coverage for long-tail nouns. |
| **10** | **Long Unsegmented Broadcast Archive** | **25h – 35h** | Unprocessed Raw Audio | ⭐⭐⭐⭐ (High) | High (VAD + Pseudo-label) | Low | Massive acoustic scaling through semi-supervised self-training. |

---

## 3. Scientific Rationale for Top-Ranked Sources

### Rank 1: WAAZ Community Crowdsourced Speech (+2.63 Hours)
- **Why #1**: Conversational Kashmiri ASR models trained exclusively on studio speech catastrophically fail when deployed on real users speaking through smartphone microphones in varying acoustic environments.
- WAAZ represents the **only corpus in our repository containing live smartphone submissions from 53 local native speakers** across Srinagar, Baramulla, and Pulwama.
- Including Tier A in E22-01 established our current SOTA champion. Unlocking the 167 Tier B recordings and the 403 unreviewed submissions directly targets conversational error modes.

### Rank 2: Meta Omnilingual Kashmiri Slice (+7.41 Hours)
- **Why #2**: Meta FAIR's field collection contains 788 clean, studio-verified clips in Perso-Arabic Nastaliq script.
- The train partition was completely isolated from historical baseline models B-01/B-02C. It has zero speaker collision with our test benchmarks and is immediately ready to ingest without any preprocessing delay.

### Rank 3: Mirwaiz Bayan-ul-Furqan Spoken Commentary Recovery (+15.11 Hours)
- **Why #3**: This is the **largest single pool of high-prestige, fluent native Kashmiri speech in existence** (17,728 clean audio segments).
- Mirwaiz speaks continuous, coherent Kashmiri without hesitation or disfluency. Because the digital text of the tafsir is already in our repository (`QURAN_WRITTEN_TEXT_INVENTORY.jsonl`), forced alignment requires zero manual transcription cost.

### Rank 4 & 5: Leak-Safe IndicVoices (+3.54h) and Bolbosh (+9.18h)
- **Why #4 & #5**: These slices were already verified, standardized into 16kHz mono, and cryptographically hashed.
- They were quarantined from early baselines solely because naive manifests mixed them with validation splits. With our clean isolation filters, they provide **12.72 hours of immediate, zero-cost training expansion**.

---

## 4. Gated and Known-Not-Present Sources (External Potential)

The audit surveyed several prominent external Kashmiri datasets that are known but currently inaccessible:

1. **UrduSpeech (Kashmiri Partition)** (`humairawan/UrduSpeech`):
   - Estimated ~25,000 clips (~30–40 hours).
   - Status: Blocked by gated HuggingFace access; authorization pending provider approval.
2. **Chaashini Multi-Dialectal Speech** (`kapturecx/Chaashini`):
   - 4,733 clips (11.04 hours).
   - Status: Gated commercial license; non-commercial research access request submitted.
3. **Kashmiri Shruk ASR** (`Omarrran/100_shruk-speech_to_Text__ASR_dataset`):
   - 100 poetic couplets (0.5 hours).
   - Status: Gated access.
4. **External Quranic Audio** (`SRC_QURAN_ALFAJR` [62.5h] and `SRC_QURAN_SALAFI` [41.9h]):
   - Fully cataloged in `QURAN_AUDIO_ACQUISITION_INVENTORY.csv`.
   - Hosted on external internet mirrors; can be scheduled for acquisition in future workstreams if external downloading is authorized.
