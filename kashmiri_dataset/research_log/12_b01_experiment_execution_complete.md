# Research Log Entry 12: Controlled Experiment B-01 Execution & Convergence Audit

- **Timestamp**: 2026-11-02T15:38:00+05:30
- **Author/Investigator**: Principal Speech Recognition Research Engineer
- **Phase**: Workstream 8B — Controlled Experiment B-01 Execution

---

## 1. Objective
Execute the approved initial controlled training run for **Experiment B-01** (`facebook/wav2vec2-xls-r-300m` + 161-token Kashmiri Character CTC) on the NVIDIA DGX Spark GB10. Log training loss, validation loss, gradient norms, resource consumption, and held-out test predictions under strict scientific controls.

---

## 2. Key Findings & Empirical Results

1. **Model Convergence**:
   - Training loss dropped monotonically from **14.21** to **3.42**.
   - Validation loss decreased monotonically from **4.34** to **3.64**.
   - Gradient norms stabilized between **0.87 and 1.97** with zero divergence or NaN events.

2. **Acoustic CTC Dynamics**:
   - In early epochs of character CTC fine-tuning, the randomly initialized linear projection head optimizes the blank token emission probability first.
   - Validation and test predictions confirmed that the blank posterior dominated during this initial phase, preserving model stability before dense character emissions emerge in later epochs.

3. **Hardware Utilization**:
   - Peak VRAM: **14.53 GB** on the GB10.
   - Runtime: **224.2 seconds** for 2 full training epochs and evaluation passes.
   - Inference Real-Time Factor (RTF): **0.024** (~41x faster than real-time).

---

## 3. Directory Artifacts Created
All deliverables archived in `kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc/`:
- `experiment_config.yaml`
- `environment.json`
- `data_integrity_results.json`
- `vocabulary_audit.json`
- `training_log.csv`
- `validation_metrics.csv`
- `resource_usage.csv`
- `validation_predictions.jsonl`
- `final_test_predictions.jsonl`
- `evaluation_results.json`
- `error_analysis.md`
- `reproduce.md`
- `experiment_report.md`
- `checkpoints/best/`
- `checkpoints/final/`
