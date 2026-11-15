# Benchmark Limitations and Threats to Validity

This document outlines the methodological boundaries, structural limitations, and threats to validity inherent in the current Kashmiri ASR benchmarks.

---

## 1. Threat 1: Repeated Evaluation Exposure on `dev_assessment.jsonl`

The 1,001-utterance manifest `dev_assessment.jsonl` was created during Stage 11 and subsequently utilized as the target evaluation testbed in Stages 14, 15, 18, 19, 20, 20R, 20S, 20T, 21A, and 21B.

### Scientific Implications:
1. **Hyperparameter Over-Fitting**: Grid searches for KenLM parameters ($\alpha, \beta$), beam widths, and ranker thresholds on this subset—even when tuned on `dev_calibration.jsonl`—have been repeatedly observed and validated against `dev_assessment.jsonl`.
2. **Not a Blind Test Set**: In accordance with rigorous machine learning reporting standards, `dev_assessment.jsonl` must be reported as a **diagnostic development benchmark**.
3. **Requirement for Future Publication**: A definitive claim of independent state-of-the-art accuracy requires running final models on the untouched sealed benchmark (`benchmark_v1_1_primary.jsonl`).

---

## 2. Threat 2: Domain and Dialectal Concentration

- **Dominance of Central Dialects**: The benchmark is heavily weighted toward Srinagar / Yamraz and Maraz Kashmiri. Dialectal Dardic varieties such as Kamraz (North), Kishtwari, and Poguli are under-represented.
- **Genre Distribution**: Approximately 45% of the benchmark utterances originate from read sentences or semi-formal discourse. Highly spontaneous multi-speaker overlapping conversations represent less than 15% of the evaluation words.

---

## 3. Threat 3: Speaker Demographics

- **Speaker Metadata Incompleteness**: While all benchmark speakers are verified disjoint from training data, complete demographic metadata (precise age, socio-economic background, native district) is available for only ~78% of benchmark speakers.
- **Gender Imbalance**: The benchmark contains approximately 60% male and 40% female speakers, reflecting historical collection biases in public speech archives.

---

## 4. Threat 4: Acoustic Noise Profiling

While Benchmark v1.1 successfully quarantined samples with SNR $< 10$ dB, real-world field deployment involves acoustic degradation (e.g., street noise, cafeteria chatter, low-bitrate VoIP compression). The benchmark contains limited coverage of extreme acoustic stress conditions.
