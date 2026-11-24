# Unresolved Inventory Questions & Recommended Future Actions
**Project**: EduCasheer Kashmiri AI  
**Document Identification**: `EDU-KAS-OPEN-2026-FINAL`  
**Execution Date**: November 17, 2026  
**Auditor**: Principal Speech Dataset Researcher & Computational Linguist  

---

## 1. Executive Summary

While this forensic audit has definitively resolved the exact volume of physical audio, verified transcripts, training overlap, and benchmark collisions, several open scientific questions require explicit research decisions and human review campaigns.

This document catalogs these unresolved items, explains their technical ambiguity, and outlines actionable resolution protocols.

---

## 2. Inventory of Unresolved Forensic Questions

### Question 1: How should Dr. War's 3,678 ASR hypotheses be handled in future training?
- **Ambiguity**: Dr. War's broadcast series (`SRC_QURAN_DR_WAR`, 4.01h) was included in historical models B-02B, B-02C, and E22-01 (via `Baseline_Corpus`), but its transcript text is identical to `asr_hypothesis` (`tier1_machine_detected`).
- **Research Dilemma**: Leaving it in future models perpetuates historical acoustic errors. Removing it would eliminate 4 hours of rich broadcast speech.
- **Recommended Action**: Deploy the existing review package (`QURAN_HUMAN_REVIEW_PACKAGE/pilot_review_samples.jsonl`). A native Kashmiri reviewer should proofread the hypotheses. Proofread records become genuine Category A ground truth.

### Question 2: Can the 17,728 unaligned Mirwaiz commentary segments be recovered automatically?
- **Ambiguity**: Mirwaiz's spoken commentary (`SRC_QURAN_MIRWAIZ`, 23.78h on disk) lacks verse-level boundary markers, but the printed book text is available in `QURAN_WRITTEN_TEXT_INVENTORY.jsonl`.
- **Technical Challenge**: Because Mirwaiz occasionally paraphrases the printed commentary, standard exact-string forced alignment may fail on segments where spoken wording diverges from the classical book text.
- **Recommended Action**: Implement an acoustic-phonetic CTC search (using the E22-01 model) with a sliding character window and local normalized Levenshtein distance $\le 0.25$.

### Question 3: Should WAAZ Tier B (130 utts) be promoted to production training?
- **Ambiguity**: Stage 22 trained model E22-02 on Tier A + Tier B (`data_22b_full_waaz.jsonl`: 97,295 utts). The report noted slight WER improvement on Dev B with negligible regression on Dev A.
- **Decision Required**: Formalize whether Tier B is considered fully certified for production, or if all non-Tier-A crowdsourced speech should remain experimental.

### Question 4: External Gated Datasets on HuggingFace
- **Status**:
  - `humairawan/UrduSpeech` (Kashmiri Partition: ~30–40h estimated)
  - `kapturecx/Chaashini` (11.04h)
  - `Omarrran/100_shruk-speech_to_Text__ASR_dataset` (0.5h)
- **Bottleneck**: These datasets are gated behind manual organization approval on HuggingFace.
- **Action**: Follow up with data stewards. Once granted, apply the same forensic deduplication and benchmark isolation scripts before ingestion.

### Question 5: Licensing of Programindz / KTTS Phrase Corpus
- **Ambiguity**: `programindz_tts` (2,984 files, 4.42h) is licensed under CC-BY 4.0 on HuggingFace, but its twin repository on Mendeley (`ktts`) specifies an academic non-commercial license.
- **Action**: If EduCasheer models are intended for commercial or open-weight distribution, verify which license takes precedence or restrict the KTTS slice to research-only checkpoints.

### Question 6: Orthographic Normalization of UChicago S. Hassan Dictionary
- **Ambiguity**: The 1,876 audio files in `uchicago_S_Hassan` have transcripts in Latin romanization (`S_Hassan_dictionary.csv`) rather than Perso-Arabic Nastaliq (`kas_Arab`).
- **Action**: Run a rule-based deterministic Grapheme-to-Grapheme (G2G) transliteration script to convert the Latin IPA/Romanized headwords into canonical Perso-Arabic Nastaliq script before CTC loss training.

---

## 3. Prioritized Human Review Package

To convert the highest-value recoverable pools into verified supervision, we recommend executing human review tasks in the following order:

```text
Priority 1: WAAZ 403-utterance community upload backlog (DataCollection web UI)
            -> Effort: ~2 hours of reviewer time.
            -> Yield: ~0.85 hours of pristine Tier A speech.

Priority 2: Pilot review of 505 King Fahd / Dr. War segments (QURAN_HUMAN_REVIEW_PACKAGE)
            -> Effort: ~3 hours of reviewer time.
            -> Yield: ~1.20 hours of studio and broadcast speech.

Priority 3: Full proofreading of remaining Dr. War ASR hypotheses
            -> Effort: ~12 hours of reviewer time.
            -> Yield: ~3.81 hours of broadcast conversational speech.
```
