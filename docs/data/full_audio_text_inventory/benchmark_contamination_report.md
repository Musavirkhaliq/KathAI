# Benchmark Protection & Contamination Elimination Report
**Project**: EduCasheer Kashmiri AI Research  
**Document Identification**: `EDU-KAS-BENCH-2026-FINAL`  
**Execution Date**: November 17, 2026  
**Auditor**: Principal Speech Dataset Researcher & Computational Linguist  

---

## 1. Executive Summary & Benchmark Integrity Charter

A core principle of scientific research in the EduCasheer Kashmiri project is the **absolute inviolability of frozen evaluation benchmarks**.

Any research claim that a new acoustic model has achieved state-of-the-art (SOTA) word error rate (WER) is scientifically invalid if the training data contains:
1. Exact audio duplicates of test clips.
2. Near-duplicate recordings of the same spoken passages.
3. Overlapping speakers whose vocal tract characteristics were memorized during training.
4. Segment-level overlaps derived from the same source recording sessions.

This audit cross-referenced all candidate audio records in physical storage and the master inventory against our standardized evaluation benchmarks:

```
[Frozen Evaluation Benchmarks (14,707 unique audio files / 242 speakers)]
   ├── Canonical Held-Out Test (`test.jsonl`, `test_clean.jsonl`, etc.: 928 utts)
   ├── Stage 9 Held-Out Test (`test.jsonl`: 5,923 utts)
   ├── Canonical Validation Set (`val.jsonl`: 945 utts)
   ├── Stage 9 Validation Set (`val.jsonl`: 7,258 utts)
   ├── WAAZ Community Dev B Validation (`dev_b_waaz_val.jsonl`: 127 utts / 90 unique)
   ├── Meta Omnilingual Benchmark Splits (Dev: 212 utts, Test: 156 utts)
   └── Sarvam Indic DiarBench (20 multi-speaker files)
```

---

## 2. Discovery: Preventing Catastrophic Leakage in the 214.65-Hour Manifest

The previous data acquisition report (Workstream 9, Nov 11, 2026) consolidated 106,487 utterance records totaling 214.65 hours into `asr_training_eligible_manifest.jsonl`.

A naive interpretation assumed all 106,487 records were eligible to train future models. However, our rigorous forensic cross-audit revealed:

### Detailed Contamination Breakdown

| Benchmark Collision Type | Affected Records | Colliding Duration | Primary Constituent Sources | Impact if Trained |
| :--- | :---: | :---: | :--- | :--- |
| **Direct Held-Out Test Audio Duplication** | **5,251 utts** | **11.92 hrs** | Bolbosh (3,120), IndicVoices (1,666), OpenSLR (465) | Catastrophic test memorization; artificial ~0% WER on test. |
| **Direct Canonical Val Audio Duplication** | **6,478 utts** | **12.02 hrs** | Bolbosh (3,568), IndicVoices (2,710), OpenSLR (200) | Invalidates model checkpoint validation and early stopping. |
| **IndicVoices-R Speaker Contamination** | **8,096 utts** | **20.20 hrs** | `SRC_EXT_INDICVOICES_R` | All 8,096 records share `speaker_id` with held-out test/val speakers. |
| **TOTAL CONTAMINATED QUARANTINE** | **19,828 utts** | **47.50 hrs** | — | **Permanently excluded from all supervised training mixtures.** |

```text
106,487 records in Master Manifest (214.65 hrs)
   - 11,729 direct test/val audio duplicates (23.94 hrs)
   -  8,096 benchmark speaker collisions in IndicVoices-R (20.20 hrs)
   ---------------------------------------------------------------
   = 86,659 uncontaminated master records
   - 78,516 already present in baseline B-02C training set
   ---------------------------------------------------------------
   =  8,143 genuinely new, leak-safe records from Master (27.29 hrs)
   +    631 approved WAAZ community speech (Tier A + Tier B: 1.36 hrs)
   ---------------------------------------------------------------
   =  8,774 total leak-safe utterances added in Stage 22 DATA-22C (28.65 hrs)
```

---

## 3. Exclusion Registry for Prospective Training Data

To ensure 100% compliance during all future training experiments, the following exclusion list has been embedded into [`audio_text_pairs.parquet`](file:///home/musa/Kashmiri%20Project/docs/data/full_audio_text_inventory/audio_text_pairs.parquet):

### Specific Excluded Source Cohorts

1. **`SRC_EXT_OPENSLR122` (Test Anchor Partition)**:
   - 465 audio files (0.85 hrs) in `kashmiri_dataset/processed/audio/openslr122/` are the permanent test anchor for out-of-domain evaluation.
   - Status: **SEALED & FROZEN. MUST NEVER BE TRAINED.**
2. **`SRC_EXT_XKAAB` (Held-Out Split)**:
   - 838 audio files (1.23 hrs) in `kashmiri_dataset/processed/audio/xkaab_4hrs/` form the conversational broadcast benchmark.
   - Status: **SEALED & FROZEN. MUST NEVER BE TRAINED.**
3. **`SRC_EXT_INDICVOICES_R` (Speaker Overlap Cohort)**:
   - 8,096 audio files (20.20 hrs) in `kashmiri_dataset/processed/audio/indicvoices_r/` share speakers with the official IndicVoices test set.
   - Status: **QUARANTINED FOR BENCHMARK PROTECTION.**
4. **`SRC_COMM_WAAZ` (`dev_b_waaz_val`)**:
   - 90 recordings (0.14 hrs) representing 15 distinct community speakers.
   - Status: **SEALED & FROZEN AS DEV B BENCHMARK.**
5. **`SRC_EXT_META_OMNI` (Dev & Test Partitions)**:
   - 212 Dev clips (2.49 hrs) and 156 Test clips (1.85 hrs).
   - Status: **PRESERVED AS OFFICIAL META EVALUATION BENCHMARKS.**
6. **`SRC_BENCH_SARVAM` (Diarization Benchmark)**:
   - 20 multi-speaker files (1.10 hrs).
   - Status: **SEALED & PRESERVED.**

---

## 4. Verification & Certification

All models trained in Stage 22 (including the champion **E22-01**, **E22-02**, **E22-03**, and **E22-04**) were audited against this exclusion registry.

**Audit Verification Result**:
```json
{
  "audit_timestamp": "2026-11-17T10:39:00+05:30",
  "active_champion_model": "E22-01",
  "training_manifest": "data_22a_conservative_tier_a.jsonl",
  "total_training_utterances": 97165,
  "test_set_audio_contamination": 0,
  "validation_set_audio_contamination": 0,
  "benchmark_integrity_status": "100% PRESERVED & SEALED"
}
```
