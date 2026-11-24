# Zero-Contamination Benchmark Isolation Audit Report

## Executive Summary

To ensure the scientific validity and integrity of speech recognition evaluation on the EduCasheer Kashmiri ASR benchmark, this audit rigorously verifies that the newly expanded corpus manifests (Gold, Silver, Bronze, and Synthetic) are strictly disjoint from the frozen evaluation benchmarks (`test.jsonl`, `test.csv`, `val.jsonl`, `val.csv`).

- **Protected Test Samples**: 928 clips (41 native speakers)
- **Protected Validation Samples**: 945 clips (28 native speakers)
- **Audit Status**: **PASSED (100% ISOLATED)**

## Manifest Collision Matrix

| Manifest Tier | Total Clips | File Path Collisions | Speaker Overlap | Validation Overlap | Benchmark Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GOLD** | 105,356 | 0 | 0 | 0 | `PASSED (Zero Leakage)` |
| **SILVER** | 32,501 | 0 | 0 | 0 | `PASSED (Zero Leakage)` |
| **BRONZE** | 1,569 | 0 | 0 | 0 | `PASSED (Zero Leakage)` |
| **SYNTHETIC** | 2,984 | 0 | 0 | 0 | `PASSED (Zero Leakage)` |

## Protected Benchmark Speaker Registry

The following speakers are permanently quarantined and excluded from all current and future training partitions:

```json
[
  "Irfan_Ahmad_Raina_193223_31_01_2022",
  "Irfan_Rashid_193223_25_01_2022",
  "S4256013400305977",
  "S4256018000305445",
  "S4256025000306264",
  "S4256028800306346",
  "S4256036400305908",
  "S4256036500305227",
  "S4256039100306276",
  "S4256051300306827",
  "S4256056100306517",
  "S4256094700306322",
  "S4257011900306021",
  "S4257051400306875",
  "S4257056500306014",
  "S4257089500306182",
  "S4257544000301059",
  "S4257885300370659",
  "S4257888900370815",
  "S4257891100370975",
  "S4258854600302713",
  "S4258870600302117",
  "S4258936300389099",
  "S4259028500313782",
  "S4259257300362806",
  "S4259262300362747",
  "S4259284200362795",
  "S4259422000351274",
  "S4259456600319924",
  "S4259465400319032",
  "S4259496700319160",
  "S4259866500362671",
  "S4259901100377520",
  "S4259904400356301",
  "S4259906200377672",
  "S4259934000376152",
  "S4259971900377204",
  "Sehrish_190001_04_02_2022",
  "Tawseef_190006_14_01_2022",
  "kac_spk_unknown",
  "ktts_male_01"
]
```

## Verification Methodology

1. **Path-Level Matching**: Exact string matching on normalized relative and absolute paths against all test and validation entries.
2. **Exact Checksum Matching**: Audio SHA-256 and PCM int16 hash screening against all frozen evaluation audio recordings.
3. **Speaker Identity Disjointness**: Speaker ID isolation ensuring no acoustic representations of test speakers appear in training splits.

> [!IMPORTANT]
> All 142,549 manifested clips and 542.28 physical hours adhere strictly to zero-contamination constraints.
