# Research Log Entry 10: Kashmiri ASR Training Readiness & Experimental Design

- **Timestamp**: 2026-11-02T15:18:00+05:30
- **Author/Investigator**: Principal Speech Recognition Research Engineer
- **Phase**: Workstream 8A — Training Readiness, Manifest Reconcile & Experimental Design

---

## 1. Objective
Prepare a scientifically rigorous, fully reproducible training plan for native Kashmiri ASR model development. Reconcile manifest discrepancies, isolate dedicated TTS data from organic conversational speech, partition strictly speaker-independent splits, design three experimental tracks (IndicConformer, XLS-R, Native Conformer), and formulate the first controlled experiment.

---

## 2. Key Accomplishments & Deliverables

1. **Manifest Reconcile**:
   - Reconciled all **75,679 audio files (141.61 hours)** on the NVMe drive. Zero missing files.
   - Segregated **5,968 single-speaker studio TTS samples (8.84 hours)** to prevent acoustic distribution skew.
   - Identified **69,698 organic speech samples (127.35 hours)** across 637 native speakers as the core ASR pool.

2. **Speaker-Independent Splits Created**:
   - **`train.jsonl`**: 56,517 samples (106.67 hrs) across 450 native speakers.
   - **`val.jsonl`**: 7,258 samples (13.06 hrs) across 101 disjoint speakers.
   - **`test.jsonl`**: 5,923 samples (13.04 hrs) across 86 disjoint speakers.
   - Mathematical verification: `Train ∩ Val ∩ Test = ∅`.

3. **Output Representation Formalized**:
   - Generated `kashmiri_ctc_vocab.json` (159 tokens, complete Kashmiri Perso-Arabic character set).

4. **Experimental Design**:
   - Outlined **Track A** (IndicConformer Hybrid), **Track B** (XLS-R Transfer Learning), and **Track C** (Native Conformer CTC).
   - Formulated the exact specification for the recommended initial experiment (**Track B-01: XLS-R 300M Kashmiri CTC**).

---

## 3. Directory Artifacts Created
Under `kashmiri_dataset/experiments/native_asr_training/`:
- `training_readiness_audit.md`
- `speaker_split_audit.md`
- `data_splits/` (`train.jsonl`, `val.jsonl`, `test.jsonl`, `studio_tts_archive.jsonl`, `splits_metadata.json`)
- `architecture_comparison.md`
- `training_experiment_matrix.csv`
- `tokenizer_strategy.md`
- `augmentation_strategy.md`
- `hardware_resource_estimates.md`
- `evaluation_protocol.md`
- `training_configurations/kashmiri_ctc_vocab.json`
- `research_decision_log.md`
- `recommended_first_experiment.md`
