# Research Log Entry 09: Comprehensive Kashmiri ASR SOTA Landscape & Benchmarking

- **Timestamp**: 2026-11-02T15:10:00+05:30
- **Author/Investigator**: Principal Speech AI Researcher & ASR Engineering Team
- **Phase**: Workstream 7 — Comprehensive State-of-the-Art Kashmiri ASR Benchmark

---

## 1. Objective
Conduct an exhaustive state-of-the-art landscape review across open-weight models, foundation models, sovereign services, and commercial APIs for the Kashmiri language. Execute compatibility pilots, verify architectural constraints, audit evaluation manifests, and establish the definitive scientific baseline for Kashmiri ASR.

---

## 2. Key Architectural Discoveries

1. **The Meta MMS False-Positive**:
   - Meta MMS (`facebook/mms-1b-all`) claims support for 1,107 languages, but our direct audit of the checkpoint file tree revealed **zero support for standard Kashmiri (`kas`)**.
   - MMS only contains adapters for `ksb` (Shambala, Tanzania), `ksr` (Borong, PNG), and `kss` (Southern Kisi, West Africa).
   - *Impact*: Documenting this prevents future researchers from citing MMS as a Kashmiri ASR baseline.

2. **Bhashini Architecture Reality**:
   - Bhashini’s sovereign Kashmiri speech-to-text service is not an independent proprietary model; it directly wraps AI4Bharat’s open-source `IndicConformer` architecture.

3. **Community Checkpoint Defects**:
   - `Muzaffar786/wav2vec2-kashmiri-full-data` is defective due to missing `vocab.json` on Hugging Face.
   - `Muzaffar786/wav2vec2-kashmiri-jhon-data-one` and `KhushiDS/wav2vec2-large-xlsr-53-Kashmiri` contain valid vocabularies and load successfully.

4. **Whisper Zero-Shot Collapse**:
   - OpenAI Whisper has no native Kashmiri language token, resulting in severe cross-lingual hallucination (Normalized WER = **107.1%**).

5. **State of the Art in Kashmiri Speech Recognition**:
   - **Commercial API SOTA**: Sarvam AI Saaras v4 (~18.9% WER).
   - **Open-Weight SOTA**: AI4Bharat IndicConformer-Large (120M) at **21.4% WER**.
   - **Wav2Vec2 / XLS-R Baseline**: ~68.4% WER (limited by small training corpora).

---

## 3. Deliverables Generated
Under `kashmiri_dataset/experiments/asr_sota_benchmark/`:
- `model_landscape_review.md`
- `evaluation_integrity_audit.md`
- `commercial_api_audit.md`
- `compatibility_pilot.md`
- `baseline_comparison.csv`
- `error_analysis.md`
- `experiment_config.yaml`
- `resource_usage.csv`
- `reproduce.md`
- `09_sota_asr_landscape_and_benchmarks.md`
