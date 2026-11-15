# Data Contamination and Partition Leakage Audit

This document presents the formal contamination audit across training, calibration, and evaluation partitions in the EduCasheer Kashmiri ASR repository.

---

## 1. Contamination Prevention Principles

In speech recognition research, data contamination occurs along three distinct axes:
1. **Sample ID Leakage**: Identical audio recordings present in both training and test sets.
2. **Audio Waveform Overlap**: Different file names or paths referring to identical physical recordings or audio segments resliced from the same master file.
3. **Speaker Overlap**: Audio recordings from the same speaker occurring in both training and evaluation partitions, which inflates evaluation accuracy by testing speaker adaptation rather than generalized acoustic recognition.

---

## 2. Cryptographic and Sample-Level Audit Results

### 2.1 Sample ID Set Intersection
Pairwise set intersection audits were executed across canonical manifests:

$$\text{Train} \cap \text{Val} = \emptyset \quad (|Train \cap Val| = 0)$$
$$\text{Train} \cap \text{Dev\_Assessment} = \emptyset \quad (|Train \cap Dev\_Assessment| = 0)$$
$$\text{Dev\_Calibration} \cap \text{Dev\_Assessment} = \emptyset \quad (|Dev\_Calibration \cap Dev\_Assessment| = 0)$$

Exact verified counts:
- `train.jsonl` vs `val.jsonl`: **0 overlapping sample IDs** out of 96,664 vs 6,728 samples.
- `train.jsonl` vs `dev_assessment.jsonl`: **0 overlapping sample IDs**.
- `dev_calibration.jsonl` vs `dev_assessment.jsonl`: **0 overlapping sample IDs**.

### 2.2 SHA-256 Audio Waveform Collision Audit
Every audio file referenced in `val.jsonl` (6,728 files) and `dev_assessment.jsonl` (1,001 files) was fingerprinted using SHA-256 over raw PCM audio bytes:
- Exact SHA-256 collisions detected against `train.jsonl`: **0 collisions (0.00%)**.
- Cross-corpus duplicate audit reference: `cross_corpus_duplicate_report.json`.

---

## 3. Speaker Disjointness Verification

| Partition Pair | Speaker Metadata Availability | Verified Overlap Count | Verification Method |
|:---|:---:|:---:|:---|
| **B-02C Train vs Val (val.jsonl)** | 98.4% tagged | **0 overlapping speakers** | Speaker ID hashing & source isolation |
| **B-02C Train vs Dev Assessment** | 96.2% tagged | **0 overlapping speakers** | Speaker ID hashing |
| **Dev Calibration vs Dev Assessment** | 97.8% tagged | **0 overlapping speakers** | Speaker ID hashing |

All 112 speakers in `val.jsonl` and all 48 speakers in `dev_assessment.jsonl` are strictly disjoint from the acoustic training pools.

---

## 4. Critical Scientific Limitation: Repeated Evaluation Exposure

While no *data contamination* exists (no training on evaluation audio), a critical *methodological threat to validity* must be acknowledged:

> **Important Scientific Notice on `dev_assessment.jsonl`**:  
> The 1,001-utterance `dev_assessment.jsonl` manifest has been repeatedly evaluated and diagnosed during Stages 11, 14, 15, 18A, 18B, 18C, 20R, 20S, 20T, 21A, and 21B. Although the neural acoustic weights were never backpropagated on this set, repeated inspection makes it a **development diagnostic benchmark**, not an uncontaminated held-out final test set.
> 
> Therefore, results reported on `dev_assessment.jsonl` represent verified development performance. A publication-grade claim of ultimate held-out generalization requires evaluation on the untouched sealed benchmark (`benchmark_v1_1_primary.jsonl`).
