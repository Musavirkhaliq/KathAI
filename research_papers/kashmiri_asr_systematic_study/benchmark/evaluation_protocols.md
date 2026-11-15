# Standard Evaluation Protocol: EduCasheer-Kashmiri-ASR-Eval-v2

This document defines the strict, reproducible evaluation standard required for all Kashmiri ASR benchmarks and model comparisons.

---

## 1. Protocol Specification: `EduCasheer-Kashmiri-ASR-Eval-v2`

To ensure that reported error rates reflect true acoustic and linguistic modeling capabilities rather than preprocessing bugs, all evaluations must adhere to the following contract (`eval_v2.yaml`):

```yaml
protocol_version: "EduCasheer-Kashmiri-ASR-Eval-v2"
target_sampling_rate: 16000
channel_mode: "mono"

resampling:
  method: "sinc_interp_kaiser"
  filter_width: 64
  rolloff: 0.9475937167399596
  beta: 14.769656459379492
  bypass_on_exact_match: true

text_normalization:
  unicode_form: "NFKC"
  preserve_diacritics: true
  strip_punctuation: true
  expand_numerals: true
  lowercase: false

decoding:
  framework: "pyctcdecode"
  language_model: "lm_word_4gram_domain_d.arpa"
  beam_width: 16
  prune_history: true
  alpha_range: [0.20, 0.35]
  beta_range: [-0.50, 0.00]
  unk_score_offset: -10.0

metrics:
  word_error_rate: true
  character_error_rate_with_space: true
  character_error_rate_no_space: true
  total_substitutions: true
  total_deletions: true
  total_insertions: true
  statistical_significance: "bootstrap_paired_10000"
```

---

## 2. Audio Resampling Enforcement

Every audio file is loaded through `load_canonical_audio()`:
1. Multi-channel audio is averaged to single-channel mono.
2. If the native sample rate is 16,000 Hz, resampling is completely bypassed.
3. If the sample rate differs (e.g., 22,050 Hz Bolbosh), the 64-point Kaiser windowed sinc interpolator converts the waveform to 16,000 Hz before passing it to the neural network.

---

## 3. Decoding and Search Contracts

1. **Beam Width Contract**:
   Evaluations must use `beam_width=16` with history pruning enabled. Beam widths smaller than 16 (e.g., $B=4$ or $B=8$) suffer from search pruning failures that artificially inflate word error rates by 60–84 errors.
2. **Language Model Calibration**:
   - Baseline: $\alpha=0.35, \beta=-0.50$.
   - Calibrated Champion: $\alpha=0.20, \beta=-0.50$.
   - A negative word insertion penalty ($\beta < 0$) is mandatory to prevent the KenLM decoder from over-inserting short Kashmiri clitics and copulas (`چھ`, `تہ`, `نہ`).

---

## 4. Dual CER Metric Standard

To resolve historical discrepancies, all publications must report both CER definitions:
- **CER (with space)**: Character edit distance including whitespace divided by total characters including whitespace:
  $$\text{CER}_{\text{with\_space}} = \frac{S_{\text{char}} + D_{\text{char}} + I_{\text{char}}}{N_{\text{chars}} + N_{\text{spaces}}}$$
- **CER (no space)**: Character edit distance computed exclusively on non-whitespace graphemes:
  $$\text{CER}_{\text{no\_space}} = \frac{S_{\text{nospace}} + D_{\text{nospace}} + I_{\text{nospace}}}{N_{\text{chars}}}$$
