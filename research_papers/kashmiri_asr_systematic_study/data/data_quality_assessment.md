# Data Quality Assessment and Acoustic Health Audit

This document summarizes the acoustic and linguistic health metrics across the Consolidated Kashmiri Speech Corpus.

---

## 1. Corpus-Wide Acoustic Health Metrics

| Metric Dimension | Target Standard | Measured Value Across Master Corpus | Status |
|:---|:---:|:---:|:---:|
| **Sample Rate Uniformity** | 100% 16,000 Hz | 62.4% native 16k; 37.6% 22.05k (resampled via Kaiser sinc) | **AUDITED & SECURED** |
| **Channel Uniformity** | 100% Mono PCM | 100.00% single-channel mono | **PASSED** |
| **Duration Distribution** | $0.6 \text{ s} \le t \le 25.0 \text{ s}$ | Mean: $7.25 \text{ s}$, Median: $5.80 \text{ s}$, Min: $0.62 \text{ s}$, Max: $24.88 \text{ s}$ | **PASSED** |
| **Average SNR (Tier A)** | $> 20.0 \text{ dB}$ | $26.42 \pm 4.1 \text{ dB}$ | **EXCELLENT** |
| **Average SNR (Tier B)** | $> 12.0 \text{ dB}$ | $16.85 \pm 3.2 \text{ dB}$ | **ACCEPTABLE** |
| **Clipping Rate** | $< 0.1\%$ samples | $0.02\%$ samples flagged and repaired | **PASSED** |
| **Empty / Corrupt Files** | 0 files | 0 files detected in training manifests | **PASSED** |

---

## 2. Textual and Linguistic Health Metrics

| Linguistic Dimension | Evaluation Metric | Result | Verification Detail |
|:---|:---:|:---:|:---|
| **Character Tokenizer Coverage** | Out-of-Vocabulary (OOV) Rate | **0.000%** | Zero OOV characters across all 106k manifests |
| **Diacritic Retention** | Central Vowel Presence | **100.0%** | Vowels `ٲ`, `ؠ`, `ۆ`, `ۄ`, `ێ` fully retained |
| **Kashmiri Grapheme Consistency** | Percentage Keheh `ک` vs Arabic Kaf `ك` | **100.0% Keheh** | All legacy Arabic Kaf forms canonicalized |
| **Doachashmee Heh Integrity** | Aspirated Consonant Preservation | **100.0%** | Aspirated stops (`کھ`, `تھ`, etc.) intact |
| **Digit Expansion Rate** | Unexpanded Numeric Digits | **0.000%** | All numerals mapped to Kashmiri word strings |

---

## 3. Signal-to-Noise Ratio (SNR) Distribution by Source

```
Source                  | Mean SNR (dB) | StdDev (dB) | Dominant Acoustic Environment
------------------------|---------------|-------------|--------------------------------------
KTTS Studio Read        | 32.4          | 2.1         | Soundproof recording booth
Mirwaiz Quran Studio    | 29.8          | 2.8         | Broadcast vocal microphone
OpenSLR SLR122          | 28.5          | 3.4         | Near-field directional microphone
XKaab ASR Studio        | 27.2          | 3.1         | Controlled quiet room
Bolbosh Read Speech     | 24.6          | 4.2         | Headset microphone in university lab
IndicVoices Primary     | 18.2          | 5.4         | Field recordings / ambient noise
IndicVoices-R Dialog    | 19.5          | 4.8         | Mobile handset in domestic spaces
Whisper Crowdsourced    | 16.4          | 6.1         | Mixed consumer smartphone microphones
```

---

## 4. Summary of Quarantined Samples

During the construction of Benchmark v1.1, a total of 2,602 samples were quarantined into `excluded_samples.jsonl`:
- **Duration Violations**: 182 samples with duration $<0.6$ seconds or $>25.0$ seconds.
- **CTC Alignment Infeasibility**: 94 samples where acoustic feature frames $T < L$ (transcript characters).
- **Excessive Acoustic Noise**: 1,422 samples with estimated $\text{SNR} < 10.0 \text{ dB}$.
- **Language Identification Ambiguity**: 904 samples containing unmixed pure Urdu or Hindi phrases without Kashmiri Dardic morphosyntax.
