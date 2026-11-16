# Certified Evaluation Protocol & Standardization Standard

**Project**: EduCasheer Kashmiri Automatic Speech Recognition Research  
**Document Standard**: Canonical Scientific Evaluation Protocol for Kashmiri ASR  
**Effective Date**: November 16, 2026 (Certified Baseline)  

---

## 1. Principles of Scientific Evaluation

To prevent regression, contamination, and metric inflation, all future speech recognition models developed in this repository must strictly adhere to this standardized, immutable protocol.

### Fundamental Scientific Safeguards:
1. **Never Tune Decoder Hyperparameters on the Test Set**: Grid search sweeps for language model weight ($\alpha$), insertion bonus ($\beta$), and acoustic temperature ($T$) must be conducted strictly on [`dev_calibration.jsonl`](file:///home/musa/Kashmiri%20Project/kashmiri_dataset/experiments/stage18a_kashmiri_asr_error_atlas/manifests/dev_calibration.jsonl).
2. **Never Report Contaminated Partitions**: Models fine-tuned on expanded speech mixtures (such as Tier A + WAAZ) must **never** be evaluated on deprecated early splits (`legacy_val`, `legacy_test`, `test_clean`, or `benchmark_v1_primary_6315`).
3. **Mandatory Exact Frame Slicing**: CTC output logits must be sliced using the certified mathematical feature extractor recurrence. Heuristic frame rounding ($\text{round}(N/320)$) is strictly prohibited.
4. **Statistical Rigor**: All claims of superiority between models must report paired bootstrap uncertainty (10,000 resamples), 95% confidence intervals, and two-tailed Wilcoxon signed-rank $p$-values.

---

## 2. Standardized Audio Preprocessing Pipeline

Every evaluation script must process raw audio waveforms through the following verified pipeline implemented in [`src/data_processing/audio_processor.py`](file:///home/musa/Kashmiri%20Project/src/data_processing/audio_processor.py):

```python
from src.data_processing.audio_processor import load_audio, normalize_audio

# 1. Load audio with automatic 16 kHz sinc resampling and mono mixdown
waveform, sample_rate = load_audio(audio_path, target_sample_rate=16000, to_mono=True)

# 2. Peak normalize to prevent digital clipping
waveform = normalize_audio(waveform, target_peak=0.95)
```

- **Resampling Filter**: High-quality windowed sinc interpolation (scipy / Kaiser).
- **Format Requirements**: 16,000 Hz, 1 channel (mono), 32-bit floating point.

---

## 3. Certified CTC Slicing & Chunking Policy

To eliminate trailing frame token hallucination and memory crashes on long recordings:

### 3.1 Exact Receptive Field Frame Calculation
For `facebook/wav2vec2-xls-r-300m` (7 Conv1d layers with kernels `[10, 3, 3, 3, 3, 2, 2]` and strides `[5, 2, 2, 2, 2, 2, 2]`):
$$L_{\text{out}}^{(i)} = \left\lfloor \frac{L_{\text{in}}^{(i)} - k_i}{s_i} \right\rfloor + 1$$
Implemented in [`src/decoding/ctc_slice.py`](file:///home/musa/Kashmiri%20Project/src/decoding/ctc_slice.py):
```python
from src.decoding.ctc_slice import slice_exact_logits

# Slice logit tensor (B, T, V) or (T, V) to exact valid frames
valid_logits = slice_exact_logits(raw_logits, num_audio_samples)
```

### 3.2 Long Audio Chunking (Method C2)
For recordings exceeding 30.0 seconds, inference must be executed using context-preserving chunking (18.0s core window, 2.0s acoustic context pad) via [`src/decoding/chunking.py`](file:///home/musa/Kashmiri%20Project/src/decoding/chunking.py) to prevent quadratic attention OOM crashes.

---

## 4. Decoder Configuration Standards

| Configuration Name | Model Backbone | Decoder Strategy | $\alpha$ | $\beta$ | Beam Width | Acoustic Temp ($T$) | Primary Purpose |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`CALIBRATED_PRODUCTION`** | XLS-R-300M | KenLM 4-gram Beam Search | 0.50 | 1.50 | 100 | **1.08** | Certified SOTA deployment on GPU |
| **`BENCHMARK_COMPARABILITY`** | XLS-R-300M | KenLM 4-gram Beam Search | 0.30 | -1.00 | 64 | **1.00** | Strict matched comparison to historical stages |
| **`GREEDY_BASELINE`** | Any | Argmax CTC | — | — | 1 | 1.00 | Pure acoustic frontend evaluation |

---

## 5. Text Normalization for WER / CER Scoring

All references and hypotheses must be passed through [`src/normalization/normalizer.py`](file:///home/musa/Kashmiri%20Project/src/normalization/normalizer.py):
```python
from src.normalization.normalizer import normalize_for_wer_evaluation
from src.evaluation.metrics import evaluate_predictions

clean_ref = normalize_for_wer_evaluation(raw_reference)
clean_hyp = normalize_for_wer_evaluation(raw_hypothesis)

# Exact Dynamic Programming Levenshtein Metrics
result = evaluate_predictions([clean_ref], [clean_hyp], normalize=False)
```

### Normalization Rules:
1. **Perso-Arabic Standardization**: Arabic Kaf (`ك`) $\rightarrow$ Keheh (`ک`); Arabic Heh (`ه`) $\rightarrow$ Heh Goal (`ہ`); Arabic Yeh (`ي`) $\rightarrow$ Farsi Yeh (`ی`).
2. **Contrastive Vowels Preserved**: Distinct Kashmiri broad vowels and central diacritics (`U+0672`, `U+065F`, `U+065A`) are strictly preserved. Non-contrastive Arabic tashkeel are stripped.
3. **Numerals**: Eastern Arabic-Indic numerals (`۰-۹`) mapped to ASCII digits (`0-9`).
4. **Punctuation**: All punctuation marks removed; multiple spaces collapsed to a single space.

---

## 6. Certified Uncontaminated Evaluation Partitions

Researchers must report performance on one or more of these **verified zero-contamination benchmarks**:

| Partition Manifest Path | Utterances | Ref Words | Domain & Linguistic Role |
| :--- | :---: | :---: | :--- |
| [`kashmiri_dataset/releases/asr_training_v2_waaz/validation_manifests/dev_a_legacy_val.jsonl`](file:///home/musa/Kashmiri%20Project/kashmiri_dataset/releases/asr_training_v2_waaz/validation_manifests/dev_a_legacy_val.jsonl) | 7,258 | 123,162 | Comprehensive secular validation benchmark |
| [`kashmiri_dataset/experiments/stage23_acoustic_lm_interaction/evaluation/dev_a_disjoint.jsonl`](file:///home/musa/Kashmiri%20Project/kashmiri_dataset/experiments/stage23_acoustic_lm_interaction/evaluation/dev_a_disjoint.jsonl) | 6,257 | 106,530 | Disjoint multi-speaker held-out evaluation set |
| [`kashmiri_dataset/releases/asr_training_v2_waaz/validation_manifests/dev_b_waaz_val.jsonl`](file:///home/musa/Kashmiri%20Project/kashmiri_dataset/releases/asr_training_v2_waaz/validation_manifests/dev_b_waaz_val.jsonl) | 127 | 1,262 | Spontaneous conversational community discourse |
| [`kashmiri_dataset/experiments/stage18a_kashmiri_asr_error_atlas/manifests/dev_assessment.jsonl`](file:///home/musa/Kashmiri%20Project/kashmiri_dataset/experiments/stage18a_kashmiri_asr_error_atlas/manifests/dev_assessment.jsonl) | 1,001 | 16,845 | Diagnostic phonetic error atlas benchmark |
| [`kashmiri_dataset/experiments/stage18a_kashmiri_asr_error_atlas/manifests/dev_calibration.jsonl`](file:///home/musa/Kashmiri%20Project/kashmiri_dataset/experiments/stage18a_kashmiri_asr_error_atlas/manifests/dev_calibration.jsonl) | 1,001 | 16,632 | Dedicated decoder parameter calibration partition |
| [`kashmiri_dataset/experiments/stage22b_decoder_and_independent_test/final_test/stage22b_final_test_predictions.jsonl`](file:///home/musa/Kashmiri%20Project/kashmiri_dataset/experiments/stage22b_decoder_and_independent_test/final_test/stage22b_final_test_predictions.jsonl) | 5,923 | 120,816 | Sealed final independent test set |

Any publication, technical report, or model card generated from this repository must cite this protocol.
