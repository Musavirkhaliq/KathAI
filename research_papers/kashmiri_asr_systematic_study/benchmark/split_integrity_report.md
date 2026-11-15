# Split Integrity & Verification Report

This report documents the verification checks performed to guarantee speaker and sample disjointness across all benchmark partitions.

---

## 1. Audit Summary Across Benchmark Releases

| Audit Gate | Verification Procedure | Result | Integrity Status |
|:---|:---|:---:|:---:|
| **Sample ID Disjointness** | Cross-manifest set intersection | 0 overlapping IDs | **VERIFIED** |
| **Audio Hash Disjointness** | SHA-256 hash collision scan | 0 colliding hashes | **VERIFIED** |
| **Speaker Overlap** | Speaker ID cross-referencing | 0 overlapping speakers | **VERIFIED** |
| **Duration Compliance** | Boundary audit ($0.6\text{s} \le t \le 25\text{s}$) | 100% compliant in v1.1 | **VERIFIED** |
| **CTC Feasibility** | $T \ge L$ verification | 100% compliant in v1.1 | **VERIFIED** |
| **Diacritic Integrity** | NFKC normalization & token check | 0 OOV tokens | **VERIFIED** |

---

## 2. Cross-Corpus Contamination Matrix

```
                      Train (B-02C)   Val (6,728)   Dev_Assess (1,001)   Benchmark v1.1 (6,315)
Train (B-02C)              —               0                 0                     0
Val (6,728)                0               —                 0                     0
Dev_Assess (1,001)         0               0                 —                     0
Benchmark v1.1 (6,315)     0               0                 0                     —
```

Pairwise overlap between all active manifests is identically zero.
Artifact reference: `kashmiri_dataset/manifests/deduplication/cross_corpus_duplicate_report.json`.
