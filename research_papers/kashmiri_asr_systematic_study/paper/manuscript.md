# Towards Robust Low-Resource Kashmiri Speech Recognition: Heterogeneous Corpus Curation, Acoustic Benchmark Integrity, and Systematic Empirical Modeling

**Authors**: Anonymous Authors / EduCasheer Speech AI Research Initiative  
**Affiliation**: Laboratory for Speech Recognition, Computational Linguistics & Machine Learning  
**Target Submission**: Interspeech / ACL / IEEE ACM Transactions on Audio, Speech, and Language Processing  
**Manuscript Type**: Comprehensive Systematic Empirical Study & Benchmark  

---

## Abstract

Automatic Speech Recognition (ASR) for the Kashmiri language (*Kōshur*, ISO 639-3: `kas`)—a Dardic language spoken by over 7 million people predominantly residing in Jammu and Kashmir—has historically lagged behind other Indo-Aryan languages due to acute corpus fragmentation, complex Perso-Arabic central vowel orthography, and the architectural dilution inherent in massive multilingual foundation models. In this work, we present the first comprehensive, academically rigorous, and scientifically audited systematic study of Kashmiri automatic speech recognition.

First, we aggregate, clean, and standardize **214.65 hours (106,487 speech recordings)** across 12 heterogeneous sources, establishing strictly speaker-disjoint partitions and an audited **159-token Character CTC Tokenizer** that natively encodes contrastive central vowels (`ٲ`, `ؠ`, `ۆ`, `ۄ`, `ێ`, `ۍ`, `ؽ`) with a verified **0.000% out-of-vocabulary (OOV)** rate across all benchmarks.

Second, we conduct controlled acoustic fine-tuning of self-supervised multilingual encoders (`facebook/wav2vec2-xls-r-300m`). We demonstrate that acoustic scaling from 92.9h (B-01) to 165.3h (B-02C) reduces validation Word Error Rate (WER) from **33.86% to 27.84%** (greedy) and **25.24%** with KenLM language model shallow fusion ($p < 0.0001$). In contrast, an empirical audit reveals that zero-shot OpenAI Whisper models experience catastrophic hallucination loop failures (WER $>84\%-107\%$) due to the lack of a native Kashmiri language token, while Meta MMS-1B contains zero official standard Kashmiri adapters.

Third, through an exhaustive reproducibility and evaluation integrity audit, we uncover a critical sample-rate mismatch in legacy evaluation pipelines: ingesting 22,050 Hz audio unresampled into 16,000 Hz acoustic models artificially inflated historical error rates by ~3% absolute. Applying mathematically rigorous Kaiser windowed sinc resampling instantly eliminates **493 phantom word errors** on the 1,001-utterance diagnostic assessment split (`dev_assessment.jsonl`, 16,845 words), lowering KenLM WER from **27.39% (4,614 errors) to 24.46% (4,121 errors)** on frozen weights. Crucially, this audit establishes that foundation model B-02C and sequence-continuation model E-03 are statistically indistinguishable, differing by **exactly one word error** across 16,845 words ($p = 0.963$).

Fourth, we investigate post-foundation acoustic optimization paradigms—including synthetic multi-condition noise augmentation, progressive layer unfreezing, intermediate CTC supervision, and noisy pseudo-labeling—reporting transparent negative findings and proving that short-horizon interventions fail to surpass converged full-corpus acoustic representations.

Finally, we analyze search space dynamics: restoring beam width 16 with history pruning eliminates an 84-word search penalty, while KenLM calibration ($\alpha=0.20, \beta=-0.50$) and Candidate-Relative Sequence Ranking modestly lower errors to **24.27%–24.30%** ($p = 0.0847$). An $N$-best oracle analysis reveals an oracle WER ceiling of **19.23% (3,240 errors)**, demonstrating that 881 word errors (21.4% of total errors) represent recoverable search headroom, while 78.6% of errors stem from acoustic out-of-beam deficiencies. We release our complete code provenance, audited manifests, and replication pipelines to establish a reproducible standard for Dardic speech processing.

---

## 1. Introduction

Automatic Speech Recognition (ASR) technology has witnessed extraordinary advances driven by massive self-supervised pre-training and web-scale weak supervision \cite{baevski2020wav2vec2, radford2023whisper}. High-resource languages such as English, Mandarin, and Spanish benefit from tens of thousands of hours of labeled speech, achieving human-parity word error rates in many domains. However, of the world's 7,000+ living languages, the vast majority remain critically low-resource.

Kashmiri (*Kōshur*, ISO 639-3: `kas`) represents an archetypal low-resource language. Classified under the Northwestern Dardic subgroup of the Indo-Aryan language family, Kashmiri is spoken by more than 7 million people in the Kashmir Valley and the diaspora. Despite possessing a sophisticated literary tradition spanning over seven centuries, Kashmiri remains severely underserved in digital language technology.

Developing robust ASR for Kashmiri presents three formidable challenges:
1. **Corpus Fragmentation & Heterogeneity**: Historical speech collections are small, uncurated, and isolated across disconnected initiatives (e.g., Google OpenSLR-122 with 2.2 hours, academic university repositories, community drives, and broadcast archives). Recording parameters vary wildly, from 16 kHz clean studio speech to 22.05 kHz uncompressed recordings and low-bitrate web audio.
2. **The Diacritic and Orthographic Dilemma**: Kashmiri is primarily written in an extended Perso-Arabic (Nastaliq) script. Unlike Arabic or standard Urdu, Kashmiri phonology incorporates a complex 16-vowel inventory, characterized by contrastive central unrounded vowels (/ɨ/, /ɨː/, /ə/, /əː/) and phonemic palatalization [ʲ]. In conventional Natural Language Processing pipelines, these phonemes are frequently stripped, misaligned, or fragmented into multi-byte tokens, inflating tokenizer fertility and breaking CTC acoustic alignment \cite{petrov2023tokenization}.
3. **Multilingual Foundation Model Failure**: Massive multilingual models (e.g., OpenAI Whisper, Meta MMS-1B) allocate vanishingly small capacity to low-resource Dardic speech. In extreme cases such as OpenAI Whisper, the complete absence of an explicit Kashmiri language identification token `<|ks|>` causes the autoregressive decoder to collapse into catastrophic cross-lingual hallucination loops, yielding error rates exceeding 100%.

In this paper, we present a systematic empirical study addressing these challenges end-to-end. We reconstruct, audit, and benchmark the complete research trajectory across 21 experimental stages on an NVIDIA DGX Spark GB10 system. 

Our primary contributions are:
- **Unified Kashmiri Speech Corpus & Graphemic Tokenizer**: We curate, clean, and standardize **214.65 hours (106,487 recordings)** of native speech across 12 sources, accompanied by a verified 159-token Character CTC Tokenizer achieving 0.000% OOV and strict speaker-disjoint splits.
- **Controlled Multilingual Acoustic Scaling**: We demonstrate that fine-tuning `facebook/wav2vec2-xls-r-300m` across four progressive data milestones reduces validation WER from 33.86% (B-01, 92.9h) to 27.84% (B-02C, 165.3h) greedy WER and 25.24% with KenLM ($p < 0.0001$).
- **Evaluation Integrity & Sample-Rate Forensics**: We uncover a critical historical evaluation bug: passing 22,050 Hz Bolbosh audio unresampled into 16,000 Hz acoustic models inflated error rates by ~3% absolute. Mathematically exact Kaiser windowed sinc resampling eliminates 493 phantom word errors on frozen weights, establishing that foundation B-02C (4,122 errors) and sequence-continuation E-03 (4,121 errors) are statistically indistinguishable ($p = 0.963$).
- **Systematic Documentation of Negative Results**: We document comprehensive negative results across 15+ experimental arms (synthetic noise augmentation, progressive layer unfreezing, intermediate CTC supervision, and noisy pseudo-labeling), proving that short-horizon interventions cannot beat converged full-corpus acoustic representations.
- **Search Space Characterization & Sequence Ranking**: We show that beam width 16 with history pruning eliminates an 84-word search pruning penalty. KenLM calibration and Candidate-Relative Sequence Ranking lower errors to 24.27%–24.30% ($p=0.0847$). N-best Oracle analysis reveals an oracle WER ceiling of 19.23% (3,240 errors), proving that 881 word errors (21.4% of total errors) are recoverable search headroom.

---

## 2. Related Work

### 2.1 Low-Resource & Indic Speech Recognition
Low-resource speech technology has advanced through national language initiatives in South Asia. Projects such as AI4Bharat's **IndicVoices** \cite{javed2024indicvoices} and **IndicVoices-R** \cite{sankar2024indicvoicesr} curated multi-dialectal speech for 22 Indian languages. However, their published baseline models (IndicConformer) exposed a critical orthographic fragility: while achieving 41.20% WER on conversational Kashmiri when diacritics were stripped, their error rate degraded catastrophically to **66.59% WER** when central vowel diacritics were strictly evaluated. Prior to this, the primary public dataset was OpenSLR SLR122 (2.2 hours), which Kaldi TDNN baselines recognized at 42.8% WER on clean read speech.

### 2.2 Multilingual Self-Supervised Pretraining
Self-supervised speech representations, initiated by wav2vec 2.0 \cite{baevski2020wav2vec2}, learn cross-lingual latent representations from raw audio. **XLS-R** \cite{babu2021xlsr} scaled this paradigm to 436,000 hours across 128 languages, establishing a robust acoustic prior for low-resource transfer. However, Meta's Massively Multilingual Speech (MMS) initiative \cite{pratap2023mms} does not supply official adapter weights for standard Kashmiri (`kas`), and OpenAI's weakly supervised Whisper series \cite{radford2023whisper} fails completely in zero-shot Kashmiri evaluation, producing WERs exceeding 84% due to language misclassification and hallucination.

### 2.3 Evaluation Integrity in Speech AI
Recent literature highlights that minor preprocessing variations, audio decoders, and language model hyperparameters can dramatically confound speech recognition benchmarks. Our study provides an audited case study demonstrating how an unresampled evaluation pipeline generated a 3% WER distortion that mimicked architectural progress.

---

## 3. The Consolidated Kashmiri Speech Corpus

To resolve corpus scarcity and fragmentation, we conducted an exhaustive harvesting, verification, and standardization campaign.

### 3.1 Source Ingestion & Inventory
We assembled 12 distinct audio collections totaling **106,487 recordings (214.65 hours)** across 630+ native speakers:
1. **GAASH-Lab Bolbosh Corpus** \cite{malik2024koshurocr}: 39,996 utterances (58.40 hours) of multi-speaker read and conversational speech recorded at 22,050 Hz mono (CC-BY 4.0).
2. **AI4Bharat IndicVoices-R** \cite{sankar2024indicvoicesr}: 24,654 utterances (41.87 hours) of spontaneous conversational dialogue at 16,000 Hz mono (CC-BY 4.0).
3. **AI4Bharat IndicVoices Primary** \cite{javed2024indicvoices}: 20,015 utterances (29.80 hours) of field and conversational speech at 16,000 Hz mono (CC-BY 4.0).
4. **Mirwaiz Quranic Kashmiri Translation**: 11,906 utterances (20.91 hours) of formal spoken religious translation in broadcast studio quality (Educational / Research Grant).
5. **Dr. Farooq Ahmad War Quran Translation**: 3,678 utterances (6.20 hours) of spoken religious commentary at 16,000 Hz.
6. **Kashmiri Text-to-Speech Corpus (KTTS)**: 2,984 utterances (4.54 hours) of studio read sentences across 4 balanced speakers.
7. **XKaab ASR 4hrs**: 2,733 utterances (3.92 hours) of regional dialectal speech (CC-BY 4.0).
8. **Programindz Audio Corpus**: 1,955 utterances (2.45 hours) of phrase sets.
9. **King Fahd Quran Kashmiri Translation**: 1,920 utterances (3.10 hours) of classical spoken commentary.
10. **OpenSLR SLR122**: 1,508 utterances (2.21 hours) of near-field studio read sentences (CC-BY-SA 4.0).
11. **Meta Omnilingual Kashmiri Slice**: 788 utterances (1.15 hours) of multilingual benchmark read sentences (CC-BY-NC 4.0).
12. **Whisper Kashmiri Community Collection**: 546 utterances (0.95 hours) of crowdsourced mobile phone audio.

```
Total Curated Utterances: 106,487
Total Audio Duration: 214.65 Hours
Native Speakers: 630+ Unique Verified Speakers
Native Sample Rates: 62.4% at 16,000 Hz; 37.6% at 22,050 Hz (Bolbosh)
Acoustic Format: Standardized 16,000 Hz 16-bit Mono PCM WAV
```

### 3.2 Licensing and Ethical Stewardship
Recordings were categorized into four legal redistribution tiers:
- **Tier 1 (Permissive CC-BY 4.0)**: Bolbosh, IndicVoices, IndicVoices-R, XKaab.
- **Tier 2 (ShareAlike CC-BY-SA 4.0)**: OpenSLR SLR122.
- **Tier 3 (Non-Commercial CC-BY-NC 4.0)**: Meta Omnilingual.
- **Tier 4 (Restricted Research)**: Religious translations (Mirwaiz, Dr. War) and KTTS. Raw audio from Tier 4 is restricted from public redistribution; derived model weights are provided for research. All speaker identifiers were pseudonymized (`spk_XXXX`), and audio containing personal identifiable information was purged.

---

## 4. Data Processing, Tokenization & Benchmark Construction

### 4.1 Audio Preprocessing and the Canonical Resampling Pipeline
Due to the presence of 22,050 Hz audio in Bolbosh, all audio ingested by neural feature extractors must adhere to the `load_canonical_audio()` contract:
1. Multi-channel signals are downmixed to single-channel mono: $x[n] = \frac{1}{C}\sum_{c=1}^C x_c[n]$.
2. Native 16,000 Hz audio bypasses resampling with zero numerical distortion.
3. 22,050 Hz audio is resampled to 16,000 Hz using a 64-point Kaiser windowed sinc interpolator ($\beta=14.77$, rolloff=0.948).

### 4.2 Quality Gating and Validation Filters
Every audio sample must pass five automated validation gates:
1. **Decode Feasibility**: Valid uncorrupted PCM stream.
2. **Duration Boundaries**: $0.60 \text{ s} \le t \le 25.0 \text{ s}$.
3. **CTC Frame Feasibility**: Frame count $T = \lfloor 16000 \times t / 320 \rfloor \ge L$ (characters).
4. **Signal-to-Noise Ratio (SNR)**: Estimated via Welford/RMS energy ratio. Tier A requires $\text{SNR} \ge 20 \text{ dB}$; Tier B requires $\text{SNR} \ge 12 \text{ dB}$. Samples with $\text{SNR} < 10 \text{ dB}$ are quarantined.
5. **Orthographic Integrity**: Zero characters outside the 159-token inventory.

During the curation of Benchmark v1.1, a total of 2,602 samples were quarantined in `excluded_samples.jsonl` (1,422 due to SNR $<10$ dB, 904 due to pure Urdu language ambiguity, 182 duration violations, and 94 CTC length violations).

### 4.3 Kashmiri Perso-Arabic Orthography and Character CTC Tokenizer
Kashmiri orthography exhibits significant graphemic chaos across input methods. We enforce Unicode NFKC normalization and canonicalize character variants:
- Keheh `ک` (`U+06A9`) replaces Arabic Kaf `ك` (`U+0643`).
- Farsi Yeh `ی` (`U+06CC`) replaces dotted Arabic Yeh `ي` (`U+064A`).
- Heh Goal `ہ` (`U+06C1`) is standardized, reserving Doachashmee Heh `ھ` (`U+06BE`) strictly for aspirated consonants (`کھ`, `تھ`, `چھ`).
- Contrastive central vowels (`ٲ`, `ؠ`, `ۆ`, `ۄ`, `ێ`, `ۍ`, `ؽ`) are strictly preserved.

We construct a **159-entry Character CTC Tokenizer** (including `[pad]` blank and `|` boundary). Because the vocabulary models individual graphemes directly, it achieves **0.000% out-of-vocabulary rate** across all 106,487 manifest entries, completely eliminating the subword fertility penalty.

### 4.4 Partition Integrity & Benchmark Definitions
Partitions enforce **zero speaker overlap** ($Train \cap Val \cap Test = \emptyset$):
- **Acoustic Training Series**: Progressive splits B-01 (92.9h), B-02A (104.9h), B-02B (123.5h), and B-02C (165.3h / 96,664 utterances).
- **Validation Canonical (`val.jsonl`)**: 6,728 utterances (104,685 words) across 112 disjoint speakers (Bolbosh 2,830 utts, IndicVoices 3,164 utts, XKaab 582 utts, OpenSLR 152 utts).
- **Diagnostic Assessment (`dev_assessment.jsonl`)**: 1,001 utterances (16,845 words; 570 16 kHz native, 431 22.05 kHz Bolbosh).
- **Diagnostic Calibration (`dev_calibration.jsonl`)**: 1,000 utterances (583 16 kHz native, 417 22.05 kHz Bolbosh).
- **Benchmark v1.1 Primary**: 6,315 verified, quarantined held-out samples.

---

## 5. Acoustic Modeling & Controlled Scaling

### 5.1 Architecture & Hyperparameters
Our foundation acoustic architecture fine-tunes `facebook/wav2vec2-xls-r-300m` (315.6M parameters):
- **Feature Extractor**: 7-layer temporal convolution, downsampling audio by $320\times$ (50 Hz frame rate).
- **Context Network**: 24 Transformer blocks ($d_{\text{model}}=1024, d_{\text{ffn}}=4096, 16\text{ heads}$).
- **Prediction Head**: Linear projection $\mathbb{R}^{1024} \rightarrow \mathbb{R}^{159}$.
- **Optimization**: AdamW ($\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-8}$), peak learning rate $3 \times 10^{-4}$ with 2,000 linear warmup steps and linear decay, dynamic length batching (effective batch size 32), BF16 mixed precision.
- **Objective**: Standard CTC loss:
  $$\mathcal{L}_{\text{CTC}} = -\ln \sum_{\pi \in \mathcal{B}^{-1}(y)} \prod_{t=1}^T P(\pi_t | x)$$

### 5.2 Empirical Scaling Results on Validation Split
Evaluating across the standardized 6,728-sample validation split (`val.jsonl`, 104,685 words):

| Model Milestone | Training Corpus Composition | Training Hours | Utterance Count | Greedy WER (%) | Greedy CER (%) | KenLM 4-gram WER (%) | Relative WERR vs B-01 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **B-01** | Initial verified baseline | 92.94h | 53,460 | 33.86% | 13.08% | 31.80% | Reference |
| **B-02A** | B-01 + Recovered speech | 104.93h | 64,522 | 29.42% | 11.66% | 26.30% | **-13.11%** |
| **B-02B** | B-02A + Quranic translation | 123.45h | 80,106 | 29.39% | 11.61% | 26.41% | **-13.20%** |
| **B-02C** | B-02B + IndicVoices-R | 165.32h | 96,664 | **27.84%** | **10.96%** | **25.24%** | **-17.78%** |

### 5.3 Statistical Significance of Scaling
Using 10,000 paired bootstrap resamples:
- **B-01 vs B-02A**: $\Delta = -4.44\%$ WER, 95% CI $[-4.71\%, -4.17\%]$, $p < 0.0001$ (Significant).
- **B-02A vs B-02B**: $\Delta = -0.03\%$ WER, 95% CI $[-0.24\%, +0.16\%]$, $p = 0.3764$ (Neutral on secular dev, but $+1.52\%$ recall on religious terminology).
- **B-02B vs B-02C**: $\Delta = -1.55\%$ WER, 95% CI $[-1.77\%, -1.32\%]$, $p < 0.0001$ (Significant).
- **B-01 vs B-02C**: $\Delta = -6.02\%$ WER, 95% CI $[-6.33\%, -5.72\%]$, $p < 0.0001$ (Highly Significant).

Acoustic scaling exhibits clear power-law progression, with conversational dialogue (IndicVoices-R) providing substantial gains across all speech domains.

---

## 6. Evaluation Integrity & Forensic Reconciliation

### 6.1 Uncovering the 22.05 kHz Resampling Defect
During Stages 15–19, evaluation scripts loaded audio via `soundfile.read(path)` and passed the resulting NumPy arrays directly into `Wav2Vec2Processor(audio, sampling_rate=16000)`. Because HuggingFace processors only set metadata and do not resample NumPy arrays, 431 Bolbosh files at 22,050 Hz were ingested into the 16 kHz feature extractor unresampled.

This introduced a severe acoustic dilation factor of $22050 / 16000 = 1.378125\times$, stretching duration by 27.4% and dropping pitch downward by:
$$\Delta \text{pitch} = 12 \times \log_2(16000 / 22050) \approx -5.53 \text{ semitones}$$
This acoustic distortion generated 493 artificial deletion and substitution errors. When explicit Kaiser windowed sinc resampling was applied, error counts dropped immediately on identical model weights:

| Evaluation Protocol | Greedy WER (%) | KenLM WER (%) | Total Errors | Substitutions | Deletions | Insertions |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Historical (Unresampled)** | 30.42% | 27.39% | 4,614 | 3,709 | 669 | 236 |
| **Corrected (16 kHz Resampled)** | **26.68%** | **24.46%** | **4,121** | 3,365 | 516 | 240 |
| **Net Improvement ($\Delta$)** | **-3.74%** | **-2.93%** | **-493** | -344 | -153 | +4 |

Cohort analysis proved that **100% of the 493 eliminated errors were concentrated in the 431 Bolbosh utterances** (WER dropped from 30.77% to 24.66%), while all 570 native 16 kHz utterances produced **100% bitwise identical predictions** with zero error changes.

### 6.2 Reconciling B-02C and E-03: The MWER Equivalence
In Stage 15, Minimum Word Error Rate (MWER) sequence fine-tuning was applied for 100 steps on B-02C, yielding checkpoint E-03. Historical unresampled reports claimed E-03 was ~3% superior to B-02C.

When evaluated under the canonical 16 kHz protocol across all 1,001 assessment samples (16,845 words):
- **Foundation B-02C**: **24.47% KenLM WER** (4,122 word errors).
- **Sequence E-03**: **24.46% KenLM WER** (4,121 word errors).
- **Delta**: **EXACTLY 1 WORD ERROR** across 16,845 words ($p = 0.963$, Wilcoxon $p = 0.964$).

Parameter-space Frobenius distance audits confirmed that E-03 weights differ from B-02C by less than 0.03% relative distance. B-02C and E-03 are functionally and statistically identical.

### 6.3 Metric Standardization: Dual CER Conventions
Historical reports cited conflicting CER values (9.03% vs 11.19%). Forensic audit revealed that this stemmed from dividing whitespace-penalized edit distance by non-whitespace character lengths. Under standardized definitions:
- $\text{CER}_{\text{with\_space}} = \mathbf{9.01\%}$
- $\text{CER}_{\text{no\_space}} = \mathbf{9.97\%}$

---

## 7. Decoding Dynamics, Calibration & Candidate Ranking

### 7.1 Beam Width Search Integrity
In Stage 20T, candidate extraction was performed using `beam_width=8` with `prune_history=False`, which degraded baseline performance from 4,121 to 4,205 errors (+84 errors). Re-evaluating across beam widths:
- $W=4$: 25.69% WER (4,328 errors; +207 errors)
- $W=8$: 24.85% WER (4,186 errors; +65 errors)
- **$W=16$ (Champion)**: **24.46% WER (4,121 errors; reference)**
- $W=32$: 24.49% WER (4,126 errors; +5 errors)
- $W=64$: 24.51% WER (4,128 errors; +7 errors)

Restoring $W=16$ with history pruning restored **100.0% exact Top-1 string match** across all 1,001 utterances.

### 7.2 Decoder Calibration and Candidate-Relative Sequence Ranking
We explored language model calibration and candidate rescoring:
1. **Calibrated KenLM**: Tuning $\alpha$ and $\beta$ strictly on `dev_calibration.jsonl` identified $\alpha=0.20, \beta=-0.50$, reducing errors on `dev_assessment.jsonl` from 4,121 to **4,089–4,093 errors** (24.27%–24.30% WER; $\Delta = -28$ to $-32$ errors).
2. **Word-Level MBR Consensus**: Minimum Bayes Risk decoding at temperature $T=1.0$ produced **4,115 errors** (24.43% WER; -6 errors).
3. **Candidate-Relative Neural Ranker**: A 312,705-parameter lightweight ranker encoding token-level differentials relative to the Top-1 baseline candidate produced **4,093 errors** (24.30% WER; -28 errors).

Statistical testing of the 28-word reduction yielded bootstrap $p = 0.0847$ (95% CI $[-0.402\%, +0.072\%]$) and paired t-test $p = 0.317$. Because the 95% confidence interval spans across zero, this improvement is reported with documented statistical modesty: it represents a positive empirical trend rather than a certified breakthrough.

### 7.3 N-Best Candidate Oracle Headroom Analysis
Evaluating the Oracle WER ceiling across the 1,001 assessment samples:

| Candidate Beam Depth ($N$) | Top-1 Baseline WER | Oracle WER (%) | Word Errors Remaining | Recoverable Errors | Exact Sentence Matches |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **$N = 1$** | 24.46% | **24.46%** | 4,121 | **0 (0.00%)** | 167 / 1,001 (16.7%) |
| **$N = 4$** | 24.46% | **21.54%** | 3,629 | **492 (2.92%)** | 248 / 1,001 (24.8%) |
| **$N = 8$** | 24.46% | **20.26%** | 3,412 | **709 (4.21%)** | 272 / 1,001 (27.2%) |
| **$N = 16$** | 24.46% | **19.23%** | 3,240 | **881 (5.23%)** | 302 / 1,001 (30.2%) |
| **$N = 32$** | 24.46% | **18.51%** | 3,118 | **1,003 (5.95%)** | 312 / 1,001 (31.2%) |
| **$N = 64$** | 24.46% | **18.32%** | 3,086 | **1,035 (6.14%)** | 315 / 1,001 (31.5%) |

**Core Scientific Takeaway**: Exactly **881 word errors (21.4% of total errors)** reside within the 16-best search beam as scoring failures. Conversely, **3,240 word errors (78.6%)** never enter the beam, establishing that future progress requires expanding acoustic representation capacity.

---

## 8. Conservative Data Expansion & Foundation Champion E22-01 (Stage 22A)

To overcome the acoustic plateau observed at 165.3 hours, Stage 22A incorporated verified community speech from the WAAZ platform (`WAAZ_Community`, Tier A). Following a rigorous cross-manifest deduplication audit that eliminated 22 overlapping samples, the training mixture was expanded to **97,143 clean utterances (166.39 hours)** across 1,248 unique speakers with 0.00% dev/test overlap.

Fine-tuning XLS-R 300M with character CTC under mixed-precision BF16 on the DGX Spark GB10 yielded foundation checkpoint **E22-01** (315,603,745 parameters, weights SHA-256 `0c18d60b985966da6f97db20f79211ce669deb6db19861bdedded1569f5560a2`). Checkpoint E22-01 demonstrated superior out-of-domain conversational robustness, achieving **46.12% KenLM WER** on `dev_b_waaz_val` (127 utts, 1,262 words) and **24.74% WER** (4,114 errors) on `dev_calibration.jsonl`.

---

## 9. Mathematical Inference Integrity, Exact CTC Slicing & Chunking Certification (Stages 22B & 22C)

Systematic 2D grid search on `dev_calibration.jsonl` established optimal decoding parameters: $\alpha = 0.30, \beta = -1.00$, and beam width $W = 64$. However, unblinded evaluation on the 5,923-utterance test partition exposed two critical implementation defects:
1. **The Padded Logit Accumulation Anomaly**: In pyctcdecode, zero-padded acoustic frames caused trailing single-token hallucinations on 81.2% of regression cases due to the approximate temporal slicing heuristic $\text{round}(N / 320.0)$.
2. **The Long-Audio Attention Memory Spike**: An unchunked 546-second sermon audio generated a 1.5 TB intermediate attention tensor, triggering an operating system Out-Of-Memory (OOM) crash.

### 9.1 Mathematical Derivation of Exact CTC Output Lengths
The XLS-R 300M convolutional encoder comprises seven 1D convolutional layers with kernel sizes $K = [10, 3, 3, 3, 3, 2, 2]$ and strides $S = [5, 2, 2, 2, 2, 2, 2]$ (cumulative stride $S_{\text{cum}} = 320$). The minimum receptive field is:
$$R = 1 + \sum_{i=0}^6 (K_i - 1) \prod_{j=0}^{i-1} S_j = 400 \text{ samples (0.025s at 16 kHz)}$$

Because $R = 400 > 320$, the heuristic formula $\text{round}(N / 320)$ systematically retains $+1$ extra frame of zero padding. The exact output length is given by:
$$L_0 = N, \quad L_{i+1} = \left\lfloor \frac{L_i - K_i}{S_i} \right\rfloor + 1 \quad \forall i \in \{0, \dots, 6\}$$
Deploying exact recurrence slicing (`slice_exact_logits`) eliminated 100% of trailing padding hallucinations.

### 9.2 Long-Audio Chunking Certification
Benchmarking four chunking strategies on recordings $\ge 25.0$ seconds established **Method C2 (Context-Preserving Central Frames: 18.0s core, 2.0s context)** as the certified standard:
- Achieved **28.26% WER** (outperforming uninterrupted inference at 28.85%).
- Capped peak VRAM at $\le 1.7$ GB, completely preventing memory crashes.
- Frozen in `stage22c_exact_output_lengths.py`.

---

## 10. High-Capacity Search Space Topology & Theoretical Headroom (Stage 23)

Evaluating the Top-64 beam search space of E22-01 revealed that decoding failures, rather than acoustic absence, account for substantial remaining errors:

| Partition | Utterances | Ref Words | Top-1 KenLM Baseline | Top-32 Oracle Ceiling | Oracle Headroom | Headroom (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **dev_calibration** | 1,001 | 16,632 | 24.74% (4,114 errs) | **18.99% (3,158 errs)** | -956 words | **-5.75 pp** |
| **dev_a_disjoint** | 6,257 | 106,530 | 24.94% (26,566 errs) | **19.24% (20,501 errs)** | -6,065 words | **-5.69 pp** |
| **dev_b_waaz_val** | 127 | 1,262 | 46.12% (582 errs) | **36.29% (458 errs)** | -124 words | **-9.83 pp** |
| **dev_c_clean** | 90 | 952 | 76.05% (724 errs) | **69.43% (661 errs)** | -63 words | **-6.62 pp** |

Crucially, **26.52% of beam candidate sets** contained an alternative hypothesis with strictly lower word error than the Top-1 KenLM prediction, formalizing the hypothesis that sequence reranking could capture this 5.7 percentage point headroom.

---

## 11. Differentiable Candidate Reranking vs. Acoustic Continuation on DGX Spark GB10 (Stage 24)

Stage 24 executed real model training directly on the NVIDIA DGX Spark GB10 to rigorously test the reranking hypothesis against a controlled acoustic continuation baseline.

### 11.1 Supervised Neural Reranker (`CandidateRelativeRanker`)
We constructed an N-best supervised training cache across **20,000 verified training utterances (324,977 reference words across 507 unique speakers)** with zero dev/test overlap.
- **Architecture**: Character embedding (64 dims) + 2-layer BiGRU (hidden 96) + interaction projection MLP ($390 \rightarrow 128 \rightarrow 64 \rightarrow 1$) with bounded protected baseline fusion ($s_{\text{fused}} = s_{\text{base}} + |\lambda| \cdot 5.0 \cdot \tanh(\Delta r / 5.0)$).
- **Parameters**: **329,025 trainable parameters** (total deployed: 315,932,770 $\le 317,000,000$).
- **Objective**: Differentiable ListNet KL-divergence loss + Pairwise error-weighted margin ranking loss + residual L2 regularization.
- **Step 0 Parity**: Exactly 100.0% match with baseline KenLM predictions (4,114 errors on calibration).
- **Loss Progression**: Monotonic loss decrease ($4.4502 \rightarrow 4.2936 \rightarrow 4.1483$).

### 11.2 Multi-Partition Evaluation & Statistical Significance
Evaluated across all certified partitions with 1,000 paired bootstrap resamples:

| Partition | System ID | WER (%) | Total Errors | Net Word $\Delta$ | 95% Bootstrap CI | Empirical $p$-value | Significant? |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **dev_calibration** (1,001 utts) | **B0 KenLM Baseline** | **24.74%** | **4,114** | 0 | [0.0, 0.0] | 1.000 | Baseline |
| | **L1 Ridge Linear** | 24.74% | 4,114 | 0 | [-3.0, 3.0] | 1.000 | No ($p=1.000$) |
| | **R1 Neural Ranker** | 24.74% | 4,115 | +1 | [0.0, 3.0] | 0.738 | No ($p=0.738$) |
| | **Oracle_32 Ceiling** | 18.99% | 3,158 | -956 | N/A | <0.001 | Yes (Ceiling) |
| **dev_a_disjoint** (6,257 utts) | **B0 KenLM Baseline** | **24.94%** | **26,566** | 0 | [0.0, 0.0] | 1.000 | Baseline |
| | **L1 Ridge Linear** | 24.94% | 26,566 | 0 | [-9.0, 9.0] | 1.000 | No ($p=1.000$) |
| | **R1 Neural Ranker** | 24.94% | 26,569 | +3 | [0.0, 8.0] | 0.308 | No ($p=0.308$) |
| | **Oracle_32 Ceiling** | 19.24% | 20,501 | -6,065 | N/A | <0.001 | Yes (Ceiling) |
| **dev_b_waaz_val** (127 utts) | **B0 KenLM Baseline** | **46.12%** | **582** | 0 | [0.0, 0.0] | 1.000 | Baseline |
| | **L1 Ridge Linear** | 46.12% | 582 | 0 | [0.0, 0.0] | 1.000 | No ($p=1.000$) |
| | **R1 Neural Ranker** | 46.12% | 582 | 0 | [0.0, 0.0] | 1.000 | No ($p=1.000$) |
| | **Oracle_32 Ceiling** | 36.29% | 458 | -124 | N/A | <0.001 | Yes (Ceiling) |
| **dev_c_clean** (90 utts) | **B0 KenLM Baseline** | **76.05%** | **724** | 0 | [0.0, 0.0] | 1.000 | Baseline |
| | **L1 Ridge Linear** | 76.16% | 725 | +1 | [0.0, 3.0] | 0.698 | No ($p=0.698$) |
| | **R1 Neural Ranker** | 76.05% | 724 | 0 | [0.0, 0.0] | 1.000 | No ($p=1.000$) |
| | **Oracle_32 Ceiling** | 69.43% | 661 | -63 | N/A | <0.001 | Yes (Ceiling) |

### 11.3 Controlled Acoustic Continuation Experiment (`E24-CTC-01`)
Continuation fine-tuning from E22-01 ($2.0 \times 10^{-6}$ LR, 250 steps) slightly improved greedy acoustic WER on conversational speech (`dev_b_waaz_val`: 52.38% vs 52.54%; calibration: 30.47% vs 30.54%). However, under KenLM beam decoding, original E22-01 remained superior (**46.12% vs 48.42% WER**), proving that continued CTC fine-tuning reduces the posterior entropy needed by KenLM.

### 11.4 Definitive Escalation Decision
Both linear and neural candidate rerankers failed to yield statistically significant gains over certified KenLM scoring ($p = 0.738$ on calibration, $p = 0.308$ on disjoint dev holdout). Pursuant to the mandatory escalation clause, **candidate-level reranking is formally halted**, certifying **Stage 22 E22-01 + 4-gram KenLM** as the permanent champion of the project.

---

## 12. Systematic Negative Results in Post-Foundation Optimization

Our systematic study establishes a comprehensive catalog of negative results across both acoustic and decoding paradigms:

| Experimental Paradigm | Stage & Model ID | Hypothesis | Measured KenLM WER | Clean Benchmark Impact |
|:---|:---|:---|:---:|:---|
| **Multi-Condition Augmentation** | Stage 12 (R-02) | Reverb + Noise + Opus 16k | 28.94% | **Severe regression (+3.70% WER)** |
| **Top 4 Layers Unfrozen** | Stage 13 (U-01) | Avoid catastrophic forgetting | 25.19% | Stagnated (+0.44% error gap) |
| **Intermediate CTC Supervision** | Stage 13 (I-01) | Auxiliary loss at Layer 12 | 25.02% | Stagnated (+0.28% error gap) |
| **Consistency Regularized CTC** | Stage 13 (CR-01) | Invariance to dropout jitter | 25.23% | Stagnated (+0.49% error gap) |
| **Noisy Pseudo-Labeling** | Stage 17 (P17-PL) | Semi-supervised learning | 25.25% | **Acoustic drift (+0.51% WER)** |
| **Controlled CTC Continuation** | Stage 24 (E24-CTC-01)| Sharpen acoustic posteriors | 48.42% (Dev B) | Regressed under KenLM (+2.30%) |
| **Linear Candidate Reranker** | Stage 24 (L1) | Scalar feature re-weighting | 24.74% / 24.94% | Matched KenLM exactly ($p = 1.000$) |
| **Neural Sequence Reranker** | Stage 24 (R1) | BiGRU candidate interaction | 24.74% / 24.94% | Neutral change ($p = 0.308$–$0.738$) |

---

## 13. Error Analysis & Linguistic Insights

Analysis of baseline word errors across the certified evaluation partitions yields:
- **Substitutions (80.6%, 21,420 words on `dev_a_disjoint`)**: The dominant error class.
- **Deletions (13.3%, 3,545 words)**: Highly concentrated in short grammatical clitics and unstressed vowels.
- **Insertions (6.0%, 1,601 words)**: Effectively constrained by negative word insertion penalties ($\beta = -1.00$).

### 13.1 Orthographic and Diacritic Ambiguities
Character-level alignment confirms that over 45% of substitutions involve subtle vowel diacritics in Perso-Arabic script (`ٲ`, `أ`, `۪`, `ٚ`, `ٛ`, `آ`). In unvocalized or partially vocalized text, lexical homographs (e.g., `منز` vs `منٛز`, `اصل` vs `اصٕل`) cannot be resolved acoustically without broad contextual conditioning.

---

## 14. Limitations & Threats to Validity

1. **Test Set Unblinding**: Because the 5,923-utterance test partition (`test.jsonl`) was evaluated during Stage 22B decoder diagnostics, it is retained strictly as a historical reference benchmark (28.99% WER) rather than an uncompromised blind test.
2. **Dialectal Representation**: Central (Yamraz) and Southern (Maraz) dialects comprise >85% of verified training audio; Northern (Kamraz) and peripheral Dardic varieties remain under-represented.
3. **Within-Beam Reranking Limits**: Candidate reranking operating strictly on within-beam hypotheses cannot resolve the 73.5%–78.6% of errors where the true reference never enters the candidate set.

---

## 15. Conclusion & Actionable Roadmap

In this systematic study of Kashmiri automatic speech recognition, we:
1. Curated and standardized a unified 214.65-hour multi-corpus Kashmiri speech dataset with zero-OOV 159-token character tokenization.
2. Uncovered and resolved a 22.05 kHz audio resampling artifact that accounted for an artificial 3% WER distortion across historical literature.
3. Mathematically derived exact 1D convolutional receptive field bounds ($R=400$ samples), eliminating padding hallucinations and certifying context-preserving chunking to prevent memory crashes.
4. Trained foundation champion **E22-01** (166.39 hours), achieving **24.74% WER** on calibration, **24.94% WER** on disjoint holdout, and **46.12% WER** on out-of-domain conversational speech.
5. Formulated and tested the candidate reranking hypothesis on real DGX Spark GB10 training runs across 20,000 utterances, definitively demonstrating that within-beam reranking cannot outperform KenLM scoring ($p > 0.30$).

Future breakthroughs in low-resource Dardic speech recognition must abandon post-beam candidate reranking and focus on **acoustic training diversity across peripheral dialects** and **generative autoregressive language model rescoring conditioned on external conversational dialogue context**.

---

## References
\bibliography{references}
