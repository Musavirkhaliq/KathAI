# Stage 22C Long-Audio Inventory and Alignment Audit
**EduCasheer Kashmiri AI — Workstream 8, Stage 22C**  
*Principal Speech Recognition Scientist & Reproducibility Auditor*  
*Date: November 15, 2026*

---

## 1. Overview and Manifest Distribution

In Stage 22B, unconstrained batching over un-bucketed audio lengths caused a severe Out-Of-Memory (OOM) incident when a 546-second audio sample was padded alongside 31 short audio clips. This audit examines the entire duration distribution across the four standard evaluation partitions to quantify the prevalence of long audio.

### Duration Distribution Summary Table

| Partition | Total Utterances | Total Hours | Max Duration | $> 10$s | $> 20$s | $> 25$s | $> 30$s | $> 60$s | $> 300$s |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Test (`test.jsonl`)** | 5,923 | 13.04 h | 546.63 s | 1,744 (29.4%) | 278 (4.69%) | 13 (0.22%) | 4 (0.07%) | 1 (0.02%) | 1 (0.02%) |
| **Dev A (`dev_a_legacy_val.jsonl`)** | 7,258 | 13.06 h | 49.99 s | 1,560 (21.5%) | 78 (1.07%) | 16 (0.22%) | 5 (0.07%) | 0 (0.00%) | 0 (0.00%) |
| **Dev B (`dev_b_waaz_val.jsonl`)** | 127 | 0.20 h | 17.46 s | 13 (10.2%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| **Dev C (`dev_c_challenging.jsonl`)** | 93 | 0.16 h | 23.28 s | 19 (20.4%) | 1 (1.08%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |

All 29 recordings exceeding 25.0 seconds across all partitions have been cataloged in `stage22c_long_audio_inventory.csv`.

---

## 2. Qualitative Transcript & Acoustic Verification of Long Recordings (> 30s)

In the 5,923-utterance independent test set, exactly **four recordings exceed 30.0 seconds**:

1. **`slr122_YawarHassan_192_0004` (Index 1801, 546.63s / 9.11 min, 1,777 words)**:
   - **Source**: OpenSLR 122 (Kashmiri ASR Corpus).
   - **Acoustic Content**: Contiguous, uninterrupted single-speaker religious lecture / sermon (waaz) delivered in native standard Srinagar Kashmiri. Clear acoustics, minimal background noise, natural pacing.
   - **Transcript Alignment**: Full multi-paragraph Perso-Arabic text beginning with the Basmala (*بِسمِلاہ ہِراہ مانِہ رحیم*) and concluding with the Hamdala (*وَ آخرُداوانا انل حمدولللاہ رببل آلمین*). The transcript is complete and aligns chronologically with the audio from start to finish.
2. **`bolbosh_007294` (Index 2180, 54.04s, 114 words)**:
   - **Source**: Bolbosh (Kashmiri Wikipedia / News narration).
   - **Acoustic Content**: Biographical narration regarding Yusuf al-Qaradawi. High SNR studio/clean speech.
   - **Transcript Alignment**: 100% complete multi-sentence reference text without truncation.
3. **`bolbosh_039614` (Index 2988, 42.95s, 80 words)**:
   - **Source**: Bolbosh literature reading. Clean literary Kashmiri recitation.
   - **Transcript Alignment**: Complete narrative excerpt.
4. **`bolbosh_008931` (Index 4597, 35.29s, 71 words)**:
   - **Source**: Bolbosh encyclopedic narration regarding the Wikipedia asteroid naming.
   - **Transcript Alignment**: 100% complete reference text.

### Finding
None of the long recordings exhibit transcript truncation, corrupt audio bytes, or multi-speaker crosstalk. They represent legitimate long-form continuous Kashmiri speech that must be evaluated fairly without crashing the GPU inference harness.
