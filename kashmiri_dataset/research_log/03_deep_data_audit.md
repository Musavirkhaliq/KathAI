# Research Log Entry 03: Comprehensive Multi-Modal Data Audit

- **Timestamp**: 2026-11-02T11:30:00+05:30
- **Author/Investigator**: Antigravity AI Pair Programmer & Research Team
- **Phase**: Phase 3 — Deep Acoustic & Orthographic Audit

---

## 1. Objective
Perform an exhaustive statistical, acoustic, and orthographic characterization of all processed Kashmiri speech and textual records, quantifying distributions of duration, SNR, speaker representation, gender, script, and character sets.

---

## 2. Speech & Acoustic Audit Findings

- **Total Segmented Utterances**: 9,180 records
- **Total Speech Duration**: 13.12 hours (47,249 seconds)
- **Duration Statistics**:
  - Mean Duration: 5.15 seconds
  - Median Duration: 4.32 seconds
  - Standard Deviation: 3.86 seconds
  - Min Duration: 0.50 seconds
  - Max Duration: 24.18 seconds
  - Zero-duration / Empty audio: 0 (100% valid)
- **Acoustic Signal Quality**:
  - Mean Signal-to-Noise Ratio (SNR): **39.05 dB** (Indicates high acoustic quality, suitable for both ASR and acoustic pretraining)
  - Zero-clipping violations: 9,173 clips (99.92% headroom compliant)
- **Speaker & Demographic Distribution**:
  - Unique Identified Speakers: 268 speakers
  - Top Speakers: `KTTS_spk01` (2,984 samples / single-speaker TTS), followed by 267 distinct multi-speaker field contributors.
  - Gender Balance: Male (4,616 samples / 50.3%), Female (1,097 samples / 11.9%), Unknown/Crowdsourced (3,463 samples / 37.7%), Other (4 samples).
  - Geographic Coverage: Srinagar, Anantnag, Baramulla, Budgam, Pulwama, Kupwara, Ganderbal, Kulgam, Shopian, Bandipora.

---

## 3. Orthographic & Textual Characterization

- **Total Characters Analyzed**: 654,771 characters
- **Script Composition**:
  - Perso-Arabic (`kas_Arab`): 536,250 characters (81.9%)
  - Inter-word Whitespace: 115,683 characters (17.7%)
  - Latin / English Code-Switching: 2,344 characters (0.36%)
  - Punctuation & Diacritic Markers: 493 characters
  - Digits: Standardized to word forms in ASR manifests.
- **Top Kashmiri Unicode Graphemes**:
  1. `U+0627` (Arabic Letter Alef): 72,130
  2. `U+0646` (Arabic Letter Noon): 43,446
  3. `U+0631` (Arabic Letter Reh): 39,122
  4. `U+06cc` (Arabic Letter Farsi Yeh): 35,765
  5. `U+0648` (Arabic Letter Waw): 35,444
  6. `U+06C1` (Arabic Letter Heh Goal): 26,192
  7. `U+06A9` (Arabic Letter Keheh - Kashmiri Standard): 24,930
  8. `U+06D2` (Arabic Letter Yeh Barree): 17,440

---

## 4. Key Discoveries & Technical Implications
1. **ASR / TTS Division**: `mendeley_ktts` (2,984 samples, 4.54 hrs) is a single, clean speaker ideal for TTS acoustic alignment, while `openslr122`, `programindz`, and `xkaab_asr` provide multi-speaker, diverse geographic variety ideal for generalizable ASR acoustic modeling.
2. **Code-Switching Presence**: 2,344 Latin characters exist in the raw transcripts (predominantly English technical, medical, and administrative terms). These must be preserved in a dedicated code-switched evaluation split (`test_code_switched.jsonl`) rather than discarded.

---

## 5. Artifacts Generated
- `kashmiri_dataset/quality_reports/deep_data_audit_speech_and_text.json`
- `kashmiri_dataset/research_log/03_deep_data_audit.md`
