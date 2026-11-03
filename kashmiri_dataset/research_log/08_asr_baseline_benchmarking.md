# Research Log Entry 08: Kashmiri ASR Baseline Benchmarking & Error Analysis

- **Timestamp**: 2026-11-02T14:55:00+05:30
- **Author/Investigator**: Antigravity AI Pair Programmer & Research Team
- **Phase**: Workstream 7 — Research-Grade ASR Baseline Benchmarking

---

## 1. Objective
Establish an empirical, reproducible baseline evaluation across all standardized evaluation categories before fine-tuning any acoustic models. Quantify real-world performance of off-the-shelf models on native Kashmiri speech, identify structural failure modes, and rigorously verify evaluation set integrity.

---

## 2. Integrity Audit Verification
- Verified **74,739 accessible, standardized 16 kHz Mono PCM WAV files** on local storage.
- Isolated **30 distinct held-out native speakers** with zero overlap in training splits.
- Generated 6 distinct challenge evaluation categories (Clean, Field, Multi-speaker, Unseen Speakers, Code-switched, Low SNR) and a 297-sample Master Unified Benchmark.

---

## 3. Benchmark Results
- Evaluated `openai/whisper-small` (244M parameters) on CUDA:
  - **Inference Speed**: Real-Time Factor (RTF) = **0.082** (~12x faster than real-time on NVIDIA GB10).
  - **Normalized WER**: **107.1%**
  - **Normalized CER**: **97.5%**
- **Failure Analysis**: Complete cross-lingual hallucination due to the absence of a native Kashmiri token in Whisper's multilingual vocabulary. Whisper routes Kashmiri acoustics into Latin Slavic, Portuguese, or Devanagari repetition loops.
- **Published Literature Reference**: `ai4bharat/indicconformer-large` reports **21.4% WER** on clean read speech.

---

## 4. Artifacts Generated
- `kashmiri_dataset/experiments/asr_baselines/evaluation_integrity_audit.md`
- `kashmiri_dataset/experiments/asr_baselines/transcript_quality_audit.md`
- `kashmiri_dataset/experiments/asr_baselines/model_compatibility_report.md`
- `kashmiri_dataset/experiments/asr_baselines/baseline_results.json`
- `kashmiri_dataset/experiments/asr_baselines/baseline_comparison.csv`
- `kashmiri_dataset/experiments/asr_baselines/error_analysis.md`
- `kashmiri_dataset/experiments/asr_baselines/experiment_config.yaml`
- `kashmiri_dataset/experiments/asr_baselines/reproduce.md`
- `kashmiri_dataset/experiments/asr_baselines/predictions/` (Detailed prediction JSONL files for all 7 evaluation splits)
