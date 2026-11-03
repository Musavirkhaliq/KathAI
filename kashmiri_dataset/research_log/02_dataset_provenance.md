# Research Log Entry 02: Traceable Dataset Provenance & Lineage Model

- **Timestamp**: 2026-11-02T11:29:00+05:30
- **Author/Investigator**: Antigravity AI Pair Programmer & Research Team
- **Phase**: Phase 2 — Dataset Provenance & Lineage Architecture

---

## 1. Objective
Establish an end-to-end provenance architecture that satisfies strict peer-reviewed academic criteria. Every single data point in our training, validation, and evaluation corpora must be traceable to its original creator, licensing terms, raw container hash, and exact transformation sequence.

---

## 2. Lineage Architecture

Every sample in our unified manifests incorporates the following provenance schema:

```json
{
  "sample_id": "KS_SPK04_SLR122_00123",
  "provenance": {
    "source_id": "SRC_RAW_OPENSLR122",
    "dataset_name": "OpenSLR SLR122 Kashmiri Speech",
    "original_repo": "https://www.openslr.org/122/",
    "license": "CC-BY-SA 4.0",
    "raw_relative_path": "Data/Kashmiri Project-20261101T102755Z-1-001/.../audio_session_04.wav",
    "raw_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "processing_pipeline_version": "v1.2.0",
    "transformations_applied": [
      "ffmpeg_resample_16000_mono_pcm16",
      "timestamp_segmentation_50ms_pad",
      "unicode_nfkc_normalization",
      "perso_arabic_orthographic_canonicalization"
    ]
  }
}
```

---

## 3. Provenance Verification Capabilities

With this model, our pipeline can formally answer:
1. *"Where did this exact training sample originate?"* -> Direct mapping to raw archive shard and original URL.
2. *"What modifications were made to the original acoustic signal?"* -> Re-sampling parameters, padding, VAD trimming, and peak normalization records.
3. *"Why was a specific character modified in the transcript?"* -> Documented rule IDs in the orthographic canonicalization engine.
4. *"Can this subset be released commercially?"* -> Dynamic filtering by license (`Apache-2.0` vs `CC-BY-SA-4.0` vs `Research-Only`).

---

## 4. Artifacts Generated
- `kashmiri_dataset/provenance/dataset_provenance_registry.csv`
- `kashmiri_dataset/provenance/dataset_provenance_registry.json`
- `kashmiri_dataset/research_log/02_dataset_provenance.md`
