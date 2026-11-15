# Stage 22C Test Set Provenance and Independence Audit Report
**EduCasheer Kashmiri AI — Workstream 8, Stage 22C**  
*Principal Speech Recognition Scientist & Data Provenance Auditor*  
*Date: November 15, 2026*

---

## 1. Test Manifest Identity & Provenance

The canonical independent test manifest evaluated in Stages 22B and 22C is located at:
`/home/musa/Kashmiri Project/kashmiri_dataset/experiments/native_asr_training/data_splits/test.jsonl`

- **Total Utterances**: 5,923
- **Total Speech Duration**: 13.04 hours (46,950.4 seconds)
- **Total Reference Words**: 120,816
- **Total Reference Characters**: 578,895

---

## 2. Partition Composition & Reconciliation (5,251 vs 5,923 Utterances)

Historical documentation across Stages 15, 16, 21, and 22 frequently alternates between references to a 5,251-utterance test set and a 5,923-utterance test set. This audit reconciles the exact relationship:

| Source Identifier | Sub-Corpus Name | Utterance Count | Hours | Reference Words | Description |
|---|---|:---:|:---:|:---:|---|
| `source: null (iv_ks_*)` | IndicVoices Kashmiri | 2,731 | 5.56 h | 50,078 | Standard reading/prompted community speech across multiple districts |
| `source: null (bolbosh_*)` | Bolbosh Literary & News | 2,000 | 4.88 h | 42,912 | Studio and semi-professional readings of Kashmiri articles and literature |
| `source: null (whisper_ks_*)` | Whisper Kashmiri Curated | 520 | 1.83 h | 16,212 | Synthetic and verified clean Kashmiri speech slices |
| **Subtotal (Legacy Core Test)** | **Historic Benchmark v1.0** | **5,251** | **12.27 h** | **109,202** | **The canonical legacy test split created in early project stages** |
| `source: xkaab_asr_4hrs` | XKaab Conversational ASR | 348 | 0.40 h | 4,284 | Natural multi-party conversational dialogues |
| `source: openslr_slr122` | OpenSLR 122 Kashmiri | 324 | 0.37 h | 7,330 | Crowdsourced and sermon recordings (including the 546s waaz) |
| **Total (Full Independent Test)** | **Comprehensive Test Release** | **5,923** | **13.04 h** | **120,816** | **The comprehensive held-out test set evaluated in Stage 22B/22C** |

---

## 3. Data Contamination & Independence Audit

Cross-referencing `test.jsonl` against the primary training mixture `train_mixture_2_clean_plus_waaz_tier_a.jsonl` (97,165 utterances, 166.39 hours) and development partitions:

1. **Acoustic Audio Overlap**: **0 overlapping files (0.00%)**. Every audio filepath in `test.jsonl` is physically disjoint from all training and validation audio.
2. **Speaker Overlap**: **0 overlapping named speakers (0.00%)**. The 86 distinct named speakers in the test manifest do not appear in the training partition.
3. **Prompt / Text Overlap**: 1,652 out of 5,613 unique test sentences (29.43%) share identical text prompts with training utterances. This is not contamination; it is an inherent property of fixed-prompt speech corpora (IndicVoices), where distinct speakers record identical target sentences.
4. **Limits of Test Independence Claim**:
   - Because `test.jsonl` was evaluated in Stage 22B to diagnose the KenLM padding regression, **it must be acknowledged that the test set has been previously unblinded**.
   - To maintain scientific integrity, **no hyperparameters ($\alpha, \beta, \text{beam\_width}$) were tuned or optimized on `test.jsonl`**. All decoding parameters were frozen exclusively on `dev_calibration.jsonl`, and the long-audio chunking policy was frozen exclusively on `dev_a_legacy_val.jsonl`.
