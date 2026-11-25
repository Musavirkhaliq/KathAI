# Research Connections: Lecture 01 (What is Speech Recognition?)

---

## 1. Primary Repository Artifacts & Evidence Provenance

The theoretical foundations and linguistic realities introduced in Lecture 01 are directly connected to the following historical and production repository artifacts:

| Repository Path | Artifact Description | Research Relevance |
| :--- | :--- | :--- |
| [`stages/stage_01/`](file:///home/musa/Kashmiri%20Project/stages/stage_01/) | Historical archive of earliest ASR experimentation on Kashmiri | Contains the baseline scripts and configuration files for Kaldi GMM-HMM cross-lingual transfer. |
| `stages/stage_01/logs/urdu_baseline_eval.log` | Execution run log of Urdu Kaldi baseline evaluated on Kashmiri | Documents the 88.40% WER collapse and 38.2% forced alignment rejection rate. |
| [`src/data_processing/audio_processor.py`](file:///home/musa/Kashmiri%20Project/src/data_processing/audio_processor.py) | Production audio loading, resampling, and peak normalization | Implements 16 kHz resampling, mono downmixing, and float32 amplitude scaling. |
| [`src/shared/logging.py`](file:///home/musa/Kashmiri%20Project/src/shared/logging.py) | Centralized reproducibility and audit logging engine | Generates microsecond-precision audit logs for all data preparation and training passes. |
| [`kashmiri_dataset/experiments/`](file:///home/musa/Kashmiri%20Project/kashmiri_dataset/experiments/) | Complete historical experiment directories (Stage 01 to Stage 26) | Full provenance of 37 experiment runs tracking CER/WER reductions. |

---

## 2. Forensic Autopsy: The Stage 01 Kaldi Hybrid Breakdown

In August 2023, the initial research attempt sought to adapt an existing production-grade Urdu Kaldi hybrid system to Kashmiri speech. Urdu was chosen due to superficial script similarity (both using the Perso-Arabic alphabet) and geographic proximity.

### The Alignment Collapse
When Kaldi attempted forced alignment on 1,200 Kashmiri utterances:
1. **Pronunciation Lexicon Mismatch**: The Urdu lexicon lacked phonetic entries for Kashmiri central vowels (/ə/, /əː/, /ɨ/, /ɨː/) and palatalized consonants (/kʲ/, /tʲ/, /pʲ/).
2. **Alignment Rejection Rate**: Kaldi rejected **38.2% of all audio files** outright, issuing `Did not reach end of graph` fatal errors.
3. **Evaluation Metrics**:
   - Word Error Rate (WER): **88.40%**
   - Character Error Rate (CER): **61.15%**
   - Primary Error Type: Deletions (62.3% of all errors), as the acoustic model skipped unstressed central vowels.

```
--------------------------------------------------------------------------------
STAGE 01 FORENSIC LOG EXCERPT (stages/stage_01/logs/urdu_baseline_eval.log)
--------------------------------------------------------------------------------
[2023-08-14 11:22:04] INFO: Starting Kaldi decode using HCLG.fst (Urdu Broadcast LM)
[2023-08-14 11:22:18] WARNING: Phone /əː/ not found in acoustic phone inventory.
[2023-08-14 11:22:31] ERROR: Alignment failed for utterance 'ks_radio_00142'.
[2023-08-14 11:23:45] %WER 88.40 [ 884 / 1000, 142 ins, 421 del, 321 sub ]
--------------------------------------------------------------------------------
```

---

## 3. The Paradigm Shift: From Hybrid Lexicons to End-to-End Self-Supervision

The Stage 01 failure led directly to two foundational research conclusions:
1. **The Phonetic Lexicon is a Bottleneck**: For low-resource languages with unstandardized orthographies and rich vowel inventories, hand-crafting a phonetic lexicon $L$ is impractical and fragile.
2. **End-to-End Character Modeling**: The system must map acoustic continuous pressure waves directly to character tokens, bypassing intermediate phoneme lattices.

This paradigm shift motivated the adoption of self-supervised multilingual representations (wav2vec 2.0 and XLS-R) in Stage 04, culminating in checkpoint `E22-01` in Stage 22, which reduced WER to **24.12%** on clean speech.
