# Kashmiri ASR Speech Data Processing Pipeline

This document describes the end-to-end multi-stage pipeline used to ingest, audit, standardize, and filter heterogeneous Kashmiri speech corpora.

```mermaid
flowchart TD
    A[Heterogeneous Raw Audio<br/>12 Sources, 106,487 Files, 214.65 Hours] --> B[Format Standardization & Header Audit]
    B --> C{Audio Sample Rate?}
    C -- 22,050 Hz (Bolbosh 40k utts) --> D[Kaiser Sinc Resampler to 16,000 Hz<br/>filter_width=64, beta=14.77]
    C -- 16,000 Hz Native (66k utts) --> E[Zero-Distortion Bypass]
    D --> F[Single-Channel Mono Normalization]
    E --> F
    F --> G[5-Stage Quality Validation Gates]
    G -- Pass --> H[Transcript Unicode NFKC & Diacritic Canonicalizer]
    G -- Reject (SNR < 10dB, CTC infeasible) --> Q[Quarantined Exclusion Pool]
    H --> I[Speaker-Disjoint Stratified Partitioning]
    I --> J[Training Splits (B-01, B-02A/B/C)]
    I --> K[Validation Split (val.jsonl, 6,728 utts)]
    I --> L[Diagnostic Assessment Split (dev_assessment, 1,001 utts)]
    I --> M[Sealed Benchmark v1.1]
```

---

## 1. Stage 1: Audio Ingestion and Header Forensics

Input recordings span diverse physical formats including standard WAV, Opus/WebM encapsulated with `.wav` extensions (KTTS), 22,050 Hz 16-bit PCM (Bolbosh), and variable bit-rate MP3s.

The standardization pass executes:
1. **Container Extraction**: Probing true container codecs via `torchaudio.info` and `soundfile.info`.
2. **Channel Downmixing**: Dual-channel stereo recordings are averaged to single-channel mono:
   $$x_{\text{mono}}[n] = \frac{1}{C} \sum_{c=1}^C x_c[n]$$
3. **Sample Rate Standardization**:
   - For native 16,000 Hz audio: waveform passes through unmolested.
   - For 22,050 Hz audio: downsampled using a 64-tap Kaiser windowed sinc interpolator.

---

## 2. Stage 2: Audio Quality Validation Gates

Implemented in `kashmiri_dataset/scripts/pretraining_validation_gates.py`, every sample is audited against five sequential gates:

### Gate 1: Decode Feasibility
Ensures the audio stream contains no corrupt frames, non-numeric `NaN` or `Inf` samples, and decodes completely without buffer underflow.

### Gate 2: Temporal Duration Limits
$$0.60 \text{ s} \le t_{\text{duration}} \le 25.0 \text{ s}$$
Clips shorter than 600 ms typically contain truncated phone fragments; clips longer than 25.0 s exceed GPU memory limits during batch acoustic training.

### Gate 3: CTC Length Inequality
Under the Wav2Vec2 convolutional feature extractor (stride factor 320 at 16 kHz, producing 50 Hz frame representations), the number of acoustic frames $T$ is:
$$T = \left\lfloor \frac{16000 \times t_{\text{duration}}}{320} \right\rfloor = \lfloor 50 \times t_{\text{duration}} \rfloor$$
For CTC loss to be mathematically defined, the acoustic frames must equal or exceed the tokenized transcript length $L$:
$$T \ge L$$
Any sample violating this constraint ($T < L$) is rejected, as CTC cannot align more labels than available frames.

### Gate 4: Signal-to-Noise Ratio (SNR) Thresholding
SNR is estimated using Welford's algorithm to separate high-energy speech regions from low-energy noise floors:
$$\text{SNR}_{\text{est}} = 10 \log_{10} \left( \frac{P_{\text{speech}}}{P_{\text{noise}}} \right)$$
- **Tier A (High Confidence)**: $\text{SNR} \ge 20 \text{ dB}$
- **Tier B (Permissible Conversational)**: $12 \text{ dB} \le \text{SNR} < 20 \text{ dB}$
- **Quarantined**: $\text{SNR} < 10 \text{ dB}$

---

## 3. Stage 3: Transcript Normalization & Diacritic Preservation

Kashmiri Perso-Arabic text poses a critical challenge: standard Arabic/Urdu normalization tools strip short vowel diacritics, which in Kashmiri destroys lexical meaning by collapsing contrastive central vowels into plain consonants.

Our pipeline (`kashmiri_dataset/scripts/normalize_transcripts.py`) enforces:
1. **Preservation of Distinctive Kashmiri Vowels**:
   - `U+0672` (ٲ, Alef with Wavy Hamza Above — /ɨː/)
   - `U+065F` (Wavy Hamza Below — /ɨ/)
   - `U+065A` (Small V Above — Kashmiri Palatalization mark)
   - `U+06C4` (ۄ, Kashmiri Waaw with circle)
   - `U+06D5` (ە, Kashmiri Ae)
2. **Canonical Mapping of Graphemic Variants**:
   - Arabic Kaf `ك` (`U+0643`) $\rightarrow$ Keheh `ک` (`U+06A9`)
   - Arabic Yeh with two dots `ي` (`U+064A`) $\rightarrow$ Farsi Yeh `ی` (`U+06CC`)
   - Arabic Heh `ه` (`U+0647`) $\rightarrow$ Heh Goal `ہ` (`U+06C1`)
   - Aspirated Consonants strictly preserved using Doachashmee Heh `ھ` (`U+06BE`): `کھ`, `تھ`, `پھ`, `چھ`.
3. **Digit Expansion**: Arabic/Indic numerals (`۰۱۲۳۴۵۶۷۸۹`) are converted into spoken word sequences (e.g., `۱` $\rightarrow$ `اَکھ`, `۲` $\rightarrow$ `زٕ`).

---

## 4. Stage 4: Cross-Corpus Deduplication and Splitting

Audio recordings are fingerprinted using MD5 audio chunk hashes and fuzzy text matching:
1. **Exact Audio Duplicate Elimination**: Audio pairs sharing identical waveform energy profiles are deduplicated.
2. **Speaker Isolation**: Speakers are isolated using metadata tags (`spk_XXXX`). Where speaker tags are missing, source isolation prevents leakage.
3. **Partition Stratification**: Splits are stratified to balance domain proportions across read, conversational, and broadcast speech genres.
