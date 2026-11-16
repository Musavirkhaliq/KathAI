# Kashmiri Unicode Orthography & Script Harmonization

- **Stage Index**: 02
- **Verified Chronology**: 2026-11-02 to 2026-11-02
- **Status**: `COMPLETED`
- **Evidence Confidence**: `VERIFIED`

## 1. Objective & Context
Standardize Kashmiri Perso-Arabic orthography, eliminate Arabic substitutions, harmonize diacritics and ZWNJ.

## 2. Key Code References
- [`kashmiri_dataset/scripts/normalize_transcripts.py`](file:///home/musa/Kashmiri Project/kashmiri_dataset/scripts/normalize_transcripts.py)
- [`src/normalization/normalizer.py`](file:///home/musa/Kashmiri Project/src/normalization/normalizer.py)

## 3. Associated Datasets
- `All text corpora and transcript manifests`

## 4. Checkpoints & Models
- *No persistent checkpoints created or checkpoint evaluation only.*

## 5. Verified Metrics & Empirical Outcomes
```json
{
  "normalized_samples": 106487,
  "script": "perso_arabic_kashmiri"
}
```

## 6. Scientific Conclusions
Canonical character mapping rules established (Arabic Kaf -> Keheh, Heh -> Goal, Yeh -> Kashmiri Yeh).

## 7. Failures, Bugs & Identified Limitations
Inconsistent diacritic usage across sources required strict stripping for acoustic modeling.

## 8. Complete Forensic Audit Cross-Reference
For exhaustive line-by-line code audits, training curves, confusion matrices, and audit logs, see [`PROJECT_MASTER_GUIDE.md`](file:///home/musa/Kashmiri Project/PROJECT_MASTER_GUIDE.md) and [`docs/research/STAGE_REGISTRY.md`](file:///home/musa/Kashmiri Project/docs/research/STAGE_REGISTRY.md).
