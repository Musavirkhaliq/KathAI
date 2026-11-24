# Kashmiri ASR Master Chronological Stage Registry

**Project**: EduCasheer Kashmiri Automatic Speech Recognition Research
**Audit Status**: Complete Forensic Reconstruction (Stages 1 through 24)
**Total Registered Research Stages**: 33
**Last Certified Date**: November 16, 2026

---

## 1. Executive Master Chronology Table

| Stage ID | Canonical Research Name | Verified Dates | Status | Evidence Confidence | Primary Model / Outcome |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`STAGE-01`** | Project Inception & Raw Speech Corpus Harvest | 2026-11-01 to 2026-11-02 | `COMPLETED` | `VERIFIED` | None |
| **`STAGE-02`** | Kashmiri Unicode Orthography & Script Harmonization | 2026-11-02 to 2026-11-02 | `COMPLETED` | `VERIFIED` | None |
| **`STAGE-03`** | Zero-Speaker Overlap Train/Val/Test Partitioning | 2026-11-02 to 2026-11-02 | `COMPLETED` | `VERIFIED` | None |
| **`STAGE-04`** | Tokenization Architecture: Subword vs Character CTC | 2026-11-02 to 2026-11-03 | `COMPLETED` | `VERIFIED` | Character CTC Tokenizer (159 tokens) |
| **`STAGE-05`** | KenLM Domain-Adapted Language Modeling | 2026-11-03 to 2026-11-04 | `COMPLETED` | `VERIFIED` | lm_word_4gram_domain_d.arpa, lm_word_3gram, lm_char_6gram |
| **`STAGE-06`** | External SOTA ASR Landscape Benchmarking | 2026-11-04 to 2026-11-06 | `COMPLETED` | `VERIFIED` | None |
| **`STAGE-07`** | Romanization & Transliteration Neural Adapters | 2026-11-06 to 2026-11-07 | `COMPLETED` | `VERIFIED` | ByT5 Transliteration Adapter, Neural Romanizer |
| **`STAGE-08`** | Foundation Acoustic Model B-01 (92.9 Hours) | 2026-11-07 to 2026-11-08 | `COMPLETED` | `VERIFIED` | B-01-xlsr300m-character-ctc |
| **`STAGE-09`** | Acoustic Scaling Campaign: B-02A, B-02B, B-02C | 2026-11-09 to 2026-11-12 | `COMPLETED` | `VERIFIED` | B-02A, B-02B, B-02C |
| **`STAGE-10`** | Quranic Speech Alignment & FastVerseMatcher | 2026-11-11 to 2026-11-12 | `COMPLETED` | `VERIFIED` | None |
| **`STAGE-11`** | Diagnostic Benchmark Isolation & Robustness Pilots | 2026-11-14 to 2026-11-14 | `COMPLETED (WITH AUDIT DEFECT IDENTIFIED)` | `CORROBORATED` | Stage 11 pilot models |
| **`STAGE-12`** | Environmental Robustness & Noise Augmentation | 2026-11-14 to 2026-11-14 | `COMPLETED (NEGATIVE RESULT)` | `VERIFIED` | R-01 (Reverb), R-02 (Multi-condition) |
| **`STAGE-13`** | Architectural Exploration: Unfreezing & Auxiliary Losses | 2026-11-14 to 2026-11-14 | `COMPLETED (NEGATIVE RESULT)` | `VERIFIED` | U-01, U-02, U-03, CR-01, I-01 |
| **`STAGE-14`** | Minimum Word Error Rate (MWER) & Weight Averaging Pilots | 2026-11-14 to 2026-11-14 | `COMPLETED (NUMERICAL ANOMALY RESOLVED IN STAGE 15)` | `VERIFIED` | MWER-01, WA-01, WA-02 |
| **`STAGE-15`** | Scaled MWER Verification Series (E-01..E-04) | 2026-11-14 to 2026-11-14 | `COMPLETED (ILLUSION IDENTIFIED AND RECONCILED)` | `CORROBORATED` | E-01 (control), E-02, E-03, E-04 |
| **`STAGE-16`** | Advanced Sequence & Decoder Merging | 2026-11-14 to 2026-11-15 | `COMPLETED` | `VERIFIED` | S16-01, S16-02, S16-05 (Merged) |
| **`STAGE-17`** | Noisy Speech Learning & Pseudo-Labeling (P17-PL) | 2026-11-15 to 2026-11-15 | `COMPLETED (NEGATIVE RESULT)` | `VERIFIED` | P17-PL |
| **`STAGE-17B`** | Optimal Transport CTC (OTC) Fine-Tuning | 2026-11-15 to 2026-11-15 | `COMPLETED` | `VERIFIED` | S17B-EXP-A, S17B-EXP-B, S17B-EXP-C |
| **`STAGE-18A`** | Kashmiri ASR Error Atlas & Confusion Matrix | 2026-11-15 to 2026-11-15 | `COMPLETED` | `VERIFIED` | None (Diagnostic Engine & Web Dashboard) |
| **`STAGE-18B`** | Targeted Decoder Optimization & Minimum Bayes Risk (MBR) | 2026-11-15 to 2026-11-15 | `COMPLETED` | `VERIFIED` | MBR Decoder |
| **`STAGE-18C`** | Acoustic-Aware Substitution Rescoring | 2026-11-15 to 2026-11-15 | `COMPLETED` | `VERIFIED` | Adaptive LM Rescorer, Exact CTC Rescorer |
| **`STAGE-19`** | Phonetic Confusion Recovery & Distillation | 2026-11-15 to 2026-11-15 | `COMPLETED (ANOMALY: CHECKPOINTS NOT SAVED)` | `VERIFIED` | Stage 19 KD Model |
| **`STAGE-20`** | CARE-ASR: Contrastive Acoustic Representation Enhancement | 2026-11-15 to 2026-11-15 | `INVALIDATED (SAVED CHECKPOINTS MISSING)` | `INVALID` | CARE-ASR Experimental |
| **`STAGE-20R`** | Forensic Integrity Audit & Root-Cause Discovery | 2026-11-15 to 2026-11-15 | `COMPLETED (CRITICAL MILESTONE)` | `VERIFIED` | None (Forensic Audit Engine) |
| **`STAGE-20S`** | Audio Integrity Recovery & Canonical Benchmark Re-Evaluation | 2026-11-15 to 2026-11-15 | `COMPLETED (CANONICAL GROUND TRUTH ESTABLISHED)` | `CORROBORATED` | None (Unified Evaluation Benchmark) |
| **`STAGE-20T`** | Sequence-Aware Candidate Extraction & Pruning Defect | 2026-11-15 to 2026-11-15 | `COMPLETED (DEFECT REPAIRED IN STAGE 21A)` | `VERIFIED` | Candidate Extraction Pipeline |
| **`STAGE-21A`** | Candidate Parity Repair & Search Space Oracle Ceiling | 2026-11-15 to 2026-11-15 | `COMPLETED` | `VERIFIED` | Parity Verification Pipeline |
| **`STAGE-21B`** | Decoder Hyperparameter Calibration & Pilot Neural Ranking | 2026-11-15 to 2026-11-15 | `COMPLETED` | `CORROBORATED` | Pilot CandidateRelativeRanker (312k params) |
| **`STAGE-22A`** | Conservative Tier A Data Expansion & Champion Model E22-01 | 2026-11-15 to 2026-11-15 | `COMPLETED (NEW SOTA PROJECT CHAMPION)` | `VERIFIED` | E22-01 (Champion), E22-02, E22-03, E22-04 |
| **`STAGE-22B`** | Stage 22B Decoder Grid Search & Independent Test Validation | 2026-11-15 to 2026-11-15 | `COMPLETED` | `VERIFIED` | Calibrated Decoder Configuration |
| **`STAGE-22C`** | Inference Certification, Exact CTC Slicing & Chunking | 2026-11-15 to 2026-11-15 | `COMPLETED (PRODUCTION CERTIFIED)` | `VERIFIED` | None (Inference Certification Harness) |
| **`STAGE-23`** | Acoustic-LM Interaction & Search Space Topology | 2026-11-15 to 2026-11-15 | `COMPLETED` | `VERIFIED` | Diagnostic Analysis Engine |
| **`STAGE-24`** | Actual Neural Candidate Reranker Training & Acoustic Continuation | 2026-11-15 to 2026-11-16 | `COMPLETED (DECISIVE NEGATIVE RESULT & PERMANENT CERTIFICATION)` | `VERIFIED` | CandidateRelativeRanker (329k params), E24-CTC-01 |

---

## 2. Exhaustive Stage-by-Stage Forensic Analysis

### `STAGE-01`: Project Inception & Raw Speech Corpus Harvest

- **Original Stage Identifier**: `Workstream 1`
- **Timeline**: 2026-11-01 – 2026-11-02 | **Status**: `COMPLETED` | **Confidence**: `VERIFIED`
- **Research Objective**: Harvest, aggregate, and catalog raw Kashmiri speech and text archives from fragmented online and institutional sources.
- **Scientific Hypothesis**: Aggregating community, academic, and open-source datasets will yield sufficient data (>100h) for end-to-end ASR pretraining.
- **Predecessor Stage**: `None (Genesis)`
- **Input Data & Artifacts**: Google Drive uncompressed shards (Data/), OpenSLR-122, GAASH-Lab Bolbosh, AI4Bharat IndicVoices
- **Source Code & Pipelines**: `kashmiri_dataset/scripts/audit_dataset.py, build_manifest.py`
- **Models Loaded / Utilized**: `None`
- **Models / Checkpoints Trained**: `None`
- **Major Findings & Verified Results**: Identified 214.65 hours of speech, but found severe orthographic chaos, non-standard diacritics, and inconsistent audio sample rates (16kHz, 22.05kHz, 44.1kHz).
- **Failed Approaches & Anomalies**: Direct concatenation of raw manifests failed due to incompatible metadata schemas and non-standard Arabic letter encodings.
- **Relationship to Subsequent Research**: Supplied raw inputs for Stage 2 normalizer and Stage 6 tokenizer.

### `STAGE-02`: Kashmiri Unicode Orthography & Script Harmonization

- **Original Stage Identifier**: `Workstream 2`
- **Timeline**: 2026-11-02 – 2026-11-02 | **Status**: `COMPLETED` | **Confidence**: `VERIFIED`
- **Research Objective**: Standardize Kashmiri Perso-Arabic text representations and design deterministic normalization rules.
- **Scientific Hypothesis**: Mapping Arabic glyph variants (Kaf, Yeh, Heh) to canonical Kashmiri characters eliminates spurious token explosion.
- **Predecessor Stage**: `STAGE-01`
- **Input Data & Artifacts**: Raw transcripts from Bolbosh, IndicVoices, OpenSLR-122, Wikipedia
- **Source Code & Pipelines**: `kashmiri_dataset/scripts/normalize_transcripts.py, deep_data_audit.py`
- **Models Loaded / Utilized**: `None`
- **Models / Checkpoints Trained**: `None`
- **Major Findings & Verified Results**: Created deterministic normalization mapping (Keheh U+06A9, Heh Goal U+06C1, Farsi Yeh U+06CC), eliminating non-contrastive Arabic diacritics while preserving phonemic central vowels.
- **Failed Approaches & Anomalies**: Complete diacritic stripping destroyed distinct Kashmiri vowel contrasts (e.g. distinguishing central vowels).
- **Relationship to Subsequent Research**: Underpins all transcript processing across all acoustic models (B-01 through E22-01).

### `STAGE-03`: Zero-Speaker Overlap Train/Val/Test Partitioning

- **Original Stage Identifier**: `Workstream 3`
- **Timeline**: 2026-11-02 – 2026-11-02 | **Status**: `COMPLETED` | **Confidence**: `VERIFIED`
- **Research Objective**: Partition the master speech corpus into leak-proof train, validation, and sealed test splits.
- **Scientific Hypothesis**: Strict speaker disjointness prevents acoustic memorization and yields trustworthy generalization error estimates.
- **Predecessor Stage**: `STAGE-02`
- **Input Data & Artifacts**: Master normalized manifest
- **Source Code & Pipelines**: `kashmiri_dataset/scripts/create_splits.py, audit_and_build_eval_splits.py`
- **Models Loaded / Utilized**: `None`
- **Models / Checkpoints Trained**: `None`
- **Major Findings & Verified Results**: Constructed canonical val.jsonl (6,728 samples / 104,685 words) and sealed test.jsonl with 0.000% speaker overlap against training splits.
- **Failed Approaches & Anomalies**: Random uniform splitting resulted in 14.2% speaker contamination.
- **Relationship to Subsequent Research**: Defines the core evaluation protocol for all acoustic baseline and scaling experiments.

### `STAGE-04`: Tokenization Architecture: Subword vs Character CTC

- **Original Stage Identifier**: `Workstream 4`
- **Timeline**: 2026-11-02 – 2026-11-03 | **Status**: `COMPLETED` | **Confidence**: `VERIFIED`
- **Research Objective**: Determine the optimal tokenization paradigm for Kashmiri ASR.
- **Scientific Hypothesis**: Character-level CTC modeling avoids subword out-of-vocabulary penalties and fragmentations on morphologically rich Kashmiri.
- **Predecessor Stage**: `STAGE-03`
- **Input Data & Artifacts**: Kashmiri text corpora (Wikipedia, BPCC, Aya, Transcripts)
- **Source Code & Pipelines**: `kashmiri_dataset/scripts/train_and_evaluate_tokenizers.py, reading/MASTER_TOKENIZATION_HANDBOOK.md`
- **Models Loaded / Utilized**: `SentencePiece (1k, 2k, 4k, 8k, 16k, 32k)`
- **Models / Checkpoints Trained**: `Character CTC Tokenizer (159 tokens)`
- **Major Findings & Verified Results**: SentencePiece subword tokenizers exhibited high out-of-vocabulary and subword fragmentation rates on inflected forms. The 159-token Character CTC Tokenizer achieved 0.000% OOV across the entire 1.2M-sentence corpus.
- **Failed Approaches & Anomalies**: Subword tokenization (BPE) yielded excessive sequence lengths and high CTC alignment loss.
- **Relationship to Subsequent Research**: Selected as the universal tokenization standard for all Wav2Vec2/XLS-R models.

### `STAGE-05`: KenLM Domain-Adapted Language Modeling

- **Original Stage Identifier**: `Workstream 5`
- **Timeline**: 2026-11-03 – 2026-11-04 | **Status**: `COMPLETED` | **Confidence**: `VERIFIED`
- **Research Objective**: Train statistical n-gram language models for shallow fusion CTC beam search decoding.
- **Scientific Hypothesis**: Domain-adapted 4-gram LM rescoring corrects acoustic substitution errors in CTC posterior matrices.
- **Predecessor Stage**: `STAGE-04`
- **Input Data & Artifacts**: 1.2M Kashmiri sentences across secular, literary, and religious texts
- **Source Code & Pipelines**: `kashmiri_dataset/scripts/train_kashmiri_language_models.py`
- **Models Loaded / Utilized**: `KenLM, SRILM`
- **Models / Checkpoints Trained**: `lm_word_4gram_domain_d.arpa, lm_word_3gram, lm_char_6gram`
- **Major Findings & Verified Results**: Word-level 4-gram with modified Kneser-Ney smoothing provided perplexity 142.3 and supported beam search shallow fusion via pyctcdecode.
- **Failed Approaches & Anomalies**: Character 6-gram LM had insufficient long-range lexical context to resolve word boundary ambiguities.
- **Relationship to Subsequent Research**: Used in all hybrid CTC+KenLM decoding pipelines from B-01 to E22-01.

### `STAGE-06`: External SOTA ASR Landscape Benchmarking

- **Original Stage Identifier**: `Workstream 6`
- **Timeline**: 2026-11-04 – 2026-11-06 | **Status**: `COMPLETED` | **Confidence**: `VERIFIED`
- **Research Objective**: Evaluate zero-shot and off-the-shelf multilingual foundation models on held-out Kashmiri speech.
- **Scientific Hypothesis**: Global foundation models (Whisper, IndicConformer) lack sufficient native Kashmiri pretraining.
- **Predecessor Stage**: `STAGE-05`
- **Input Data & Artifacts**: Held-out Kashmiri evaluation benchmark (6,728 samples)
- **Source Code & Pipelines**: `kashmiri_dataset/experiments/external_asr_benchmark/, kashmiri_dataset/scripts/run_whisper_benchmark.py`
- **Models Loaded / Utilized**: `Whisper Large-v3, Whisper Medium, IndicConformer 600M, Bodhan IndicTranscribe, Khushi XLS-R, Muzaffar XLS-R`
- **Models / Checkpoints Trained**: `None`
- **Major Findings & Verified Results**: Whisper Large-v3 exhibited catastrophic hallucination (84.12% WER). IndicConformer achieved 62.14% WER. Community fine-tunes achieved 49.80%-52.40% WER due to incomplete vocabularies.
- **Failed Approaches & Anomalies**: Zero-shot transfer from Hindi/Urdu models failed to resolve Kashmiri central vowel phonemes.
- **Relationship to Subsequent Research**: Established the empirical necessity of full native acoustic training.

### `STAGE-07`: Romanization & Transliteration Neural Adapters

- **Original Stage Identifier**: `Workstream 7`
- **Timeline**: 2026-11-06 – 2026-11-07 | **Status**: `COMPLETED` | **Confidence**: `VERIFIED`
- **Research Objective**: Develop dual Perso-Arabic to Roman/IAST transliteration systems for search and accessibility.
- **Scientific Hypothesis**: A character seq2seq neural adapter accurately converts Perso-Arabic transcripts to readable Roman Kashmiri.
- **Predecessor Stage**: `STAGE-06`
- **Input Data & Artifacts**: RASA transliteration dataset (64,800 pairs)
- **Source Code & Pipelines**: `kashmiri_dataset/scripts/train_byt5_transliteration_adapter.py, train_neural_romanizer.py`
- **Models Loaded / Utilized**: `google/byt5-small`
- **Models / Checkpoints Trained**: `ByT5 Transliteration Adapter, Neural Romanizer`
- **Major Findings & Verified Results**: ByT5 character seq2seq achieved 94.2% exact word transliteration accuracy and was integrated into web_asr_app.py.
- **Failed Approaches & Anomalies**: Rule-based regex transliteration failed on vowel diacritic omissions in informal text.
- **Relationship to Subsequent Research**: Provides user-facing dual-script output in production inference.

### `STAGE-08`: Foundation Acoustic Model B-01 (92.9 Hours)

- **Original Stage Identifier**: `Workstream 8 (B-01)`
- **Timeline**: 2026-11-07 – 2026-11-08 | **Status**: `COMPLETED` | **Confidence**: `VERIFIED`
- **Research Objective**: Train the initial native Kashmiri foundation acoustic model on 92.9 hours of verified speech.
- **Scientific Hypothesis**: Fine-tuning wav2vec2-xls-r-300m with character CTC will achieve <35% WER on native Kashmiri.
- **Predecessor Stage**: `STAGE-04`
- **Input Data & Artifacts**: train.jsonl (53,460 utterances, 92.94 hrs, Bolbosh + IndicVoices + OpenSLR)
- **Source Code & Pipelines**: `kashmiri_dataset/scripts/train_b01_xlsr.py, train_b01_full_schedule.py`
- **Models Loaded / Utilized**: `facebook/wav2vec2-xls-r-300m`
- **Models / Checkpoints Trained**: `B-01-xlsr300m-character-ctc`
- **Major Findings & Verified Results**: Achieved 33.86% Greedy WER / 31.80% KenLM WER on val.jsonl. Outperformed all external and commercial baselines by >16% absolute WER.
- **Failed Approaches & Anomalies**: Initial full-precision FP32 training hit VRAM limits; BF16 AMP solved this without numeric divergence.
- **Relationship to Subsequent Research**: Served as the architectural blueprint for the Stage 9 acoustic scaling series.

### `STAGE-09`: Acoustic Scaling Campaign: B-02A, B-02B, B-02C

- **Original Stage Identifier**: `Workstream 8 Stage 9 (B-02)`
- **Timeline**: 2026-11-09 – 2026-11-12 | **Status**: `COMPLETED` | **Confidence**: `VERIFIED`
- **Research Objective**: Empirically evaluate acoustic data scaling by training models at 104.9h, 123.5h, and 165.3h.
- **Scientific Hypothesis**: Acoustic scaling follows a log-linear power law, reducing word error rate monotonically as corpus size increases.
- **Predecessor Stage**: `STAGE-08`
- **Input Data & Artifacts**: Progressive manifests: B-02A (64.5k utts), B-02B (80.1k utts), B-02C (96.7k utts)
- **Source Code & Pipelines**: `kashmiri_dataset/scripts/run_stage9_full_pipeline.py, consolidate_stage9_results.py`
- **Models Loaded / Utilized**: `facebook/wav2vec2-xls-r-300m`
- **Models / Checkpoints Trained**: `B-02A, B-02B, B-02C`
- **Major Findings & Verified Results**: B-02A achieved 29.42% WER (-13.1% rel vs B-01). B-02B achieved 29.39% WER (+1.5% religious recall). B-02C achieved 27.84% Greedy WER / 25.24% KenLM WER (-17.78% rel vs B-01, p < 0.0001).
- **Failed Approaches & Anomalies**: Scaling alone began to show diminishing returns past 150 hours without acoustic domain diversification.
- **Relationship to Subsequent Research**: B-02C became the canonical baseline and parent model for all subsequent optimization stages (Stages 11-21).

### `STAGE-10`: Quranic Speech Alignment & FastVerseMatcher

- **Original Stage Identifier**: `Workstream 8 Stage 10`
- **Timeline**: 2026-11-11 – 2026-11-12 | **Status**: `COMPLETED` | **Confidence**: `VERIFIED`
- **Research Objective**: Align and audit spoken Kashmiri Quran translations across 114 Surahs.
- **Scientific Hypothesis**: Verse-anchored character trigram alignment filters out Arabic recitations and misaligned audio.
- **Predecessor Stage**: `STAGE-09`
- **Input Data & Artifacts**: Dr. War, King Fahd, and Mirwaiz spoken Quran archives
- **Source Code & Pipelines**: `kashmiri_dataset/scripts/audit_all_surahs_coverage.py, run_verse_anchor_statistical_experiment.py`
- **Models Loaded / Utilized**: `B-02B acoustic model`
- **Models / Checkpoints Trained**: `None`
- **Major Findings & Verified Results**: Audited 114 Surahs. Identified 437 pure Arabic recitation segments and quarantined them; verified 3,678 clean Kashmiri spoken commentary verses.
- **Failed Approaches & Anomalies**: Unanchored heuristic string matching caused cross-verse drift.
- **Relationship to Subsequent Research**: Provided high-fidelity religious domain speech for B-02B and Stage 22 data expansion.

### `STAGE-11`: Diagnostic Benchmark Isolation & Robustness Pilots

- **Original Stage Identifier**: `Stage 11`
- **Timeline**: 2026-11-14 – 2026-11-14 | **Status**: `COMPLETED (WITH AUDIT DEFECT IDENTIFIED)` | **Confidence**: `CORROBORATED`
- **Research Objective**: Isolate dev_assessment (1,001 utts) and dev_calibration (1,000 utts) and test initial perturbations.
- **Scientific Hypothesis**: A smaller, highly diverse diagnostic benchmark enables rapid testing of acoustic and decoder hypotheses.
- **Predecessor Stage**: `STAGE-09`
- **Input Data & Artifacts**: dev_assessment.jsonl, dev_calibration.jsonl
- **Source Code & Pipelines**: `kashmiri_dataset/experiments/stage11_robustness_optimization/`
- **Models Loaded / Utilized**: `B-02C`
- **Models / Checkpoints Trained**: `Stage 11 pilot models`
- **Major Findings & Verified Results**: Established dev_assessment as primary diagnostic set. However, 431 Bolbosh audio files were included at their native 22.05 kHz sample rate, introducing an unresampled evaluation latency defect.
- **Failed Approaches & Anomalies**: Evaluation scripts did not enforce strict audio resampling flags, masking the sample rate mismatch.
- **Relationship to Subsequent Research**: Led directly to Stage 20R forensic investigation.

### `STAGE-12`: Environmental Robustness & Noise Augmentation

- **Original Stage Identifier**: `Stage 12`
- **Timeline**: 2026-11-14 – 2026-11-14 | **Status**: `COMPLETED (NEGATIVE RESULT)` | **Confidence**: `VERIFIED`
- **Research Objective**: Train models under synthetic room impulse response (RIR) reverberation and Opus compression.
- **Scientific Hypothesis**: Synthetic multi-condition augmentation improves generalization to field recordings.
- **Predecessor Stage**: `STAGE-11`
- **Input Data & Artifacts**: train.jsonl + synthetic noise/RIR
- **Source Code & Pipelines**: `kashmiri_dataset/experiments/stage12_robustness_training/`
- **Models Loaded / Utilized**: `B-02C`
- **Models / Checkpoints Trained**: `R-01 (Reverb), R-02 (Multi-condition)`
- **Major Findings & Verified Results**: KenLM WER degraded from 25.24% to 28.94% (+3.70% error penalty) on clean speech. Synthetic noise corrupted delicate central vowel formants.
- **Failed Approaches & Anomalies**: Synthetic noise injection without clean-anchor regularization severely hurt acoustic representations.
- **Relationship to Subsequent Research**: Proved that synthetic noise is harmful for low-resource phonetically dense vowels.

### `STAGE-13`: Architectural Exploration: Unfreezing & Auxiliary Losses

- **Original Stage Identifier**: `Stage 13`
- **Timeline**: 2026-11-14 – 2026-11-14 | **Status**: `COMPLETED (NEGATIVE RESULT)` | **Confidence**: `VERIFIED`
- **Research Objective**: Test layer unfreezing (U-01..03), Consistency Regularization (CR-01), and Intermediate CTC (I-01).
- **Scientific Hypothesis**: Auxiliary intermediate losses and selective layer unfreezing improve representation stability.
- **Predecessor Stage**: `STAGE-09`
- **Input Data & Artifacts**: train.jsonl
- **Source Code & Pipelines**: `kashmiri_dataset/experiments/stage13_architectural_optimization/`
- **Models Loaded / Utilized**: `B-02C`
- **Models / Checkpoints Trained**: `U-01, U-02, U-03, CR-01, I-01`
- **Major Findings & Verified Results**: All models achieved 24.91%-25.23% KenLM WER on dev_assessment, failing to beat fully converged B-02C (24.47%). Full-model end-to-end training proved superior to selective tuning.
- **Failed Approaches & Anomalies**: Selective layer unfreezing restricted parameter adaptation capacity.
- **Relationship to Subsequent Research**: Established that architectural auxiliary losses provide no advantage over full-network pretraining.

### `STAGE-14`: Minimum Word Error Rate (MWER) & Weight Averaging Pilots

- **Original Stage Identifier**: `Stage 14`
- **Timeline**: 2026-11-14 – 2026-11-14 | **Status**: `COMPLETED (NUMERICAL ANOMALY RESOLVED IN STAGE 15)` | **Confidence**: `VERIFIED`
- **Research Objective**: Evaluate sequence-level MWER loss and weight averaging across checkpoint trajectories.
- **Scientific Hypothesis**: Directly optimizing expected edit distance aligns model outputs with Word Error Rate.
- **Predecessor Stage**: `STAGE-09`
- **Input Data & Artifacts**: dev_calibration.jsonl
- **Source Code & Pipelines**: `kashmiri_dataset/experiments/stage14_mwer_checkpoint_optimization/`
- **Models Loaded / Utilized**: `B-02C`
- **Models / Checkpoints Trained**: `MWER-01, WA-01, WA-02`
- **Major Findings & Verified Results**: Reported promising gains, but cosine similarity calculations suffered from BF16 floating point round-off error, violating the Cauchy-Schwarz bound.
- **Failed Approaches & Anomalies**: FP32/BF16 layer-wise norm accumulators produced cosine similarity > 1.0 due to mantissa truncation.
- **Relationship to Subsequent Research**: Triggered Stage 15 rigorous FP64 mathematical reformulation.

### `STAGE-15`: Scaled MWER Verification Series (E-01..E-04)

- **Original Stage Identifier**: `Stage 15`
- **Timeline**: 2026-11-14 – 2026-11-14 | **Status**: `COMPLETED (ILLUSION IDENTIFIED AND RECONCILED)` | **Confidence**: `CORROBORATED`
- **Research Objective**: Scale MWER sequence training across 4 loss weights (lambda = 0.0, 0.05, 0.10, 0.20) with FP64 geometry.
- **Scientific Hypothesis**: Scaled MWER sequence optimization yields significant Word Error Rate reductions.
- **Predecessor Stage**: `STAGE-14`
- **Input Data & Artifacts**: train.jsonl, dev_calibration.jsonl
- **Source Code & Pipelines**: `kashmiri_dataset/experiments/stage15_mwer_verification_and_scaling/`
- **Models Loaded / Utilized**: `B-02C`
- **Models / Checkpoints Trained**: `E-01 (control), E-02, E-03, E-04`
- **Major Findings & Verified Results**: Unresampled evaluation initially claimed E-03 achieved 27.39% WER vs 30.42% for B-02C. However, when evaluated under canonical 16kHz resampling (Stage 20R/20S), E-03 and B-02C achieved 4,121 vs 4,122 errors (Delta = 1 word error across 16,845 words; p = 0.963).
- **Failed Approaches & Anomalies**: The perceived superiority of MWER was an artifact of the 22.05kHz unresampled evaluation defect.
- **Relationship to Subsequent Research**: Reconciled under Stage 20S; proved that acoustic scaling dominates sequence fine-tuning.

### `STAGE-16`: Advanced Sequence & Decoder Merging

- **Original Stage Identifier**: `Stage 16`
- **Timeline**: 2026-11-14 – 2026-11-15 | **Status**: `COMPLETED` | **Confidence**: `VERIFIED`
- **Research Objective**: Explore model soup weight merging across sequence-trained checkpoints.
- **Scientific Hypothesis**: Linearly interpolating weights of diverse sequence-optimized models creates a superior ensemble.
- **Predecessor Stage**: `STAGE-15`
- **Input Data & Artifacts**: Checkpoints from Stage 15 (E-01..04)
- **Source Code & Pipelines**: `kashmiri_dataset/experiments/stage16_advanced_sequence_and_decoder/`
- **Models Loaded / Utilized**: `E-01..04, B-02C`
- **Models / Checkpoints Trained**: `S16-01, S16-02, S16-05 (Merged)`
- **Major Findings & Verified Results**: S16-05 achieved 24.47% KenLM WER (4,122 errors), identical to parent B-02C. Model merging in weight space did not break through the acoustic error floor.
- **Failed Approaches & Anomalies**: Weight interpolation provided no measurable gain over single best checkpoint.
- **Relationship to Subsequent Research**: Confirmed that acoustic representation capacity was the true bottleneck.

### `STAGE-17`: Noisy Speech Learning & Pseudo-Labeling (P17-PL)

- **Original Stage Identifier**: `Stage 17`
- **Timeline**: 2026-11-15 – 2026-11-15 | **Status**: `COMPLETED (NEGATIVE RESULT)` | **Confidence**: `VERIFIED`
- **Research Objective**: Incorporate uncurated Tier B audio using self-training and pseudo-labeling.
- **Scientific Hypothesis**: Pseudo-labeling high-confidence uncurated speech expands acoustic diversity.
- **Predecessor Stage**: `STAGE-16`
- **Input Data & Artifacts**: Tier B uncurated audio (45 hrs)
- **Source Code & Pipelines**: `kashmiri_dataset/experiments/stage17_noisy_speech_learning/`
- **Models Loaded / Utilized**: `B-02C`
- **Models / Checkpoints Trained**: `P17-PL`
- **Major Findings & Verified Results**: KenLM WER degraded to 25.25% (4,253 errors, +132 word errors). Confirmation bias caused the acoustic model to learn its own hallucinated phonemes.
- **Failed Approaches & Anomalies**: Semi-supervised self-training without ground truth verification degraded acoustic representations.
- **Relationship to Subsequent Research**: Proved that unverified pseudo-labeling is harmful for Kashmiri ASR.

### `STAGE-17B`: Optimal Transport CTC (OTC) Fine-Tuning

- **Original Stage Identifier**: `Stage 17B`
- **Timeline**: 2026-11-15 – 2026-11-15 | **Status**: `COMPLETED` | **Confidence**: `VERIFIED`
- **Research Objective**: Recover from Stage 17 regression using Wasserstein-based Optimal Transport CTC alignment.
- **Scientific Hypothesis**: Optimal transport relaxes rigid frame-to-token alignments, recovering from noisy training data.
- **Predecessor Stage**: `STAGE-17`
- **Input Data & Artifacts**: train.jsonl
- **Source Code & Pipelines**: `kashmiri_dataset/experiments/stage17b_regression_recovery_otc/`
- **Models Loaded / Utilized**: `B-02C`
- **Models / Checkpoints Trained**: `S17B-EXP-A, S17B-EXP-B, S17B-EXP-C`
- **Major Findings & Verified Results**: S17B-EXP-A achieved 24.57% KenLM WER (4,138 errors), recovering part of the Stage 17 regression but failing to outperform clean baseline B-02C (24.47%).
- **Failed Approaches & Anomalies**: OTC could not undo corrupted acoustic priors without fresh data.
- **Relationship to Subsequent Research**: Directed team attention toward decoder-side intelligence and error forensics.

### `STAGE-18A`: Kashmiri ASR Error Atlas & Confusion Matrix

- **Original Stage Identifier**: `Stage 18A`
- **Timeline**: 2026-11-15 – 2026-11-15 | **Status**: `COMPLETED` | **Confidence**: `VERIFIED`
- **Research Objective**: Perform exhaustive error categorization across substitutions, deletions, and insertions on dev_assessment.
- **Scientific Hypothesis**: Quantifying systematic phonetic confusion patterns reveals targeted decoder interventions.
- **Predecessor Stage**: `STAGE-17B`
- **Input Data & Artifacts**: Predictions on dev_assessment.jsonl
- **Source Code & Pipelines**: `kashmiri_dataset/experiments/stage18a_kashmiri_asr_error_atlas/`
- **Models Loaded / Utilized**: `B-02C`
- **Models / Checkpoints Trained**: `None (Diagnostic Engine & Web Dashboard)`
- **Major Findings & Verified Results**: Cataloged 3,709 substitutions, 669 deletions, and 236 insertions. Discovered that central vowel confusion (Tse U+0698 vs Jim U+062C, Palatalized Ya) accounted for 48.7% of all word substitutions.
- **Failed Approaches & Anomalies**: Initial error counts were inflated by unresampled Bolbosh audio files (corrected in Stage 20S).
- **Relationship to Subsequent Research**: Guided targeted substitution rescoring in Stages 18B and 18C.

### `STAGE-18B`: Targeted Decoder Optimization & Minimum Bayes Risk (MBR)

- **Original Stage Identifier**: `Stage 18B`
- **Timeline**: 2026-11-15 – 2026-11-15 | **Status**: `COMPLETED` | **Confidence**: `VERIFIED`
- **Research Objective**: Test word-level Minimum Bayes Risk decoding over N-best beam candidates.
- **Scientific Hypothesis**: Selecting candidates by minimizing expected risk under posterior probability improves WER.
- **Predecessor Stage**: `STAGE-18A`
- **Input Data & Artifacts**: N-best beam candidate hypotheses
- **Source Code & Pipelines**: `kashmiri_dataset/experiments/stage18b_targeted_decoder_optimization/`
- **Models Loaded / Utilized**: `B-02C + KenLM`
- **Models / Checkpoints Trained**: `MBR Decoder`
- **Major Findings & Verified Results**: MBR decoding with temperature T=1.0 achieved 24.43% WER (4,115 errors, -6 errors vs baseline). Demonstrates modest edit distance benefit but limited by beam diversity.
- **Failed Approaches & Anomalies**: Low beam diversity (N <= 8) caused MBR to degenerate to Top-1 selection.
- **Relationship to Subsequent Research**: Highlighted the need for larger beam candidate spaces (tested in Stage 21A/23).

### `STAGE-18C`: Acoustic-Aware Substitution Rescoring

- **Original Stage Identifier**: `Stage 18C`
- **Timeline**: 2026-11-15 – 2026-11-15 | **Status**: `COMPLETED` | **Confidence**: `VERIFIED`
- **Research Objective**: Dynamically adjust language model weight alpha based on acoustic frame entropy.
- **Scientific Hypothesis**: Increasing LM weight during uncertain acoustic frames suppresses phonetic substitutions.
- **Predecessor Stage**: `STAGE-18B`
- **Input Data & Artifacts**: Frame posterior distributions and phoneme confusion matrices
- **Source Code & Pipelines**: `kashmiri_dataset/experiments/stage18c_acoustic_aware_substitution_rescoring/`
- **Models Loaded / Utilized**: `B-02C`
- **Models / Checkpoints Trained**: `Adaptive LM Rescorer, Exact CTC Rescorer`
- **Major Findings & Verified Results**: Adaptive alpha adjustment provided marginal gains (-4 errors), showing that global calibrated static parameters perform nearly as well as frame-level dynamic scaling.
- **Failed Approaches & Anomalies**: Excessive dynamic scaling caused acoustic over-correction on dialectal variations.
- **Relationship to Subsequent Research**: Reinforced the importance of global grid calibration.

### `STAGE-19`: Phonetic Confusion Recovery & Distillation

- **Original Stage Identifier**: `Stage 19`
- **Timeline**: 2026-11-15 – 2026-11-15 | **Status**: `COMPLETED (ANOMALY: CHECKPOINTS NOT SAVED)` | **Confidence**: `VERIFIED`
- **Research Objective**: Fine-tune acoustic model with phoneme confusion knowledge distillation loss.
- **Scientific Hypothesis**: Penalizing confused vowel pairs during training sharpens acoustic decision boundaries.
- **Predecessor Stage**: `STAGE-18A`
- **Input Data & Artifacts**: train.jsonl
- **Source Code & Pipelines**: `kashmiri_dataset/experiments/stage19_phonetic_confusion_recovery/`
- **Models Loaded / Utilized**: `B-02C`
- **Models / Checkpoints Trained**: `Stage 19 KD Model`
- **Major Findings & Verified Results**: The training loop hit Step 0 serialization: calib_wer did not drop below initial baseline, so no updated weights were serialized to disk.
- **Failed Approaches & Anomalies**: Training exited without saving new checkpoints because stopping criteria required immediate monotonic validation gain.
- **Relationship to Subsequent Research**: Exposed checkpoint saving defects audited in Stage 20R.

### `STAGE-20`: CARE-ASR: Contrastive Acoustic Representation Enhancement

- **Original Stage Identifier**: `Stage 20`
- **Timeline**: 2026-11-15 – 2026-11-15 | **Status**: `INVALIDATED (SAVED CHECKPOINTS MISSING)` | **Confidence**: `INVALID`
- **Research Objective**: Introduce contrastive loss over acoustic feature representations to disambiguate central vowels.
- **Scientific Hypothesis**: Contrastive feature learning pulls identical phonemic tokens closer in latent space.
- **Predecessor Stage**: `STAGE-19`
- **Input Data & Artifacts**: train.jsonl
- **Source Code & Pipelines**: `kashmiri_dataset/experiments/stage20_care_asr/`
- **Models Loaded / Utilized**: `B-02C`
- **Models / Checkpoints Trained**: `CARE-ASR Experimental`
- **Major Findings & Verified Results**: Experiment logs claimed 24.53% WER (4,132 errors), but audit revealed that the reported evaluation was performed on in-memory representations without saving checkpoints to disk.
- **Failed Approaches & Anomalies**: In-memory evaluation was reported as a verified checkpoint result.
- **Relationship to Subsequent Research**: Direct catalyst for Stage 20R forensic integrity audit.

### `STAGE-20R`: Forensic Integrity Audit & Root-Cause Discovery

- **Original Stage Identifier**: `Stage 20R`
- **Timeline**: 2026-11-15 – 2026-11-15 | **Status**: `COMPLETED (CRITICAL MILESTONE)` | **Confidence**: `VERIFIED`
- **Research Objective**: Conduct comprehensive forensic audit across all historical checkpoints, pipelines, and evaluation scripts.
- **Scientific Hypothesis**: Discrepancies in historical metrics stem from pipeline bugs rather than architectural differences.
- **Predecessor Stage**: `STAGE-20`
- **Input Data & Artifacts**: All historical logs, scripts, and evaluation audio files
- **Source Code & Pipelines**: `kashmiri_dataset/experiments/stage20r_integrity_audit/`
- **Models Loaded / Utilized**: `All 30+ historical checkpoints`
- **Models / Checkpoints Trained**: `None (Forensic Audit Engine)`
- **Major Findings & Verified Results**: Discovered the 22.05 kHz Resampling Defect: 431 Bolbosh audio files at 22,050 Hz were passed into 16 kHz feature extractors unresampled, warping duration by 1.378x and pitch by 5.53 semitones. Resampling eliminated 493 phantom word errors on frozen weights. Also uncovered the Stage 20 Step-0 serialization bug.
- **Failed Approaches & Anomalies**: N/A (Forensic discovery)
- **Relationship to Subsequent Research**: Transformed project methodology and established the Stage 20S Unified Evaluation Protocol.

### `STAGE-20S`: Audio Integrity Recovery & Canonical Benchmark Re-Evaluation

- **Original Stage Identifier**: `Stage 20S`
- **Timeline**: 2026-11-15 – 2026-11-15 | **Status**: `COMPLETED (CANONICAL GROUND TRUTH ESTABLISHED)` | **Confidence**: `CORROBORATED`
- **Research Objective**: Re-evaluate all 30+ historical checkpoints under strict 16 kHz Kaiser sinc resampling.
- **Scientific Hypothesis**: Re-evaluating under canonical conditions reveals the true, unconfounded performance of all models.
- **Predecessor Stage**: `STAGE-20R`
- **Input Data & Artifacts**: dev_assessment.jsonl (16 kHz Kaiser resampled)
- **Source Code & Pipelines**: `kashmiri_dataset/experiments/stage20s_audio_integrity_recovery/`
- **Models Loaded / Utilized**: `All 30+ historical checkpoints (B-01, B-02A/B/C, E-01..04, S16, S17, WA-01..02)`
- **Models / Checkpoints Trained**: `None (Unified Evaluation Benchmark)`
- **Major Findings & Verified Results**: Established that foundation B-02C (4,122 errors) and sequence-continuation E-03 (4,121 errors) are statistically indistinguishable (p = 0.963). Disproved the hypothesis that MWER was a major breakthrough; verified that 100% of the historical error gap was caused by the unresampled audio bug.
- **Failed Approaches & Anomalies**: N/A
- **Relationship to Subsequent Research**: Established immutable baseline for Stages 21 through 24.

### `STAGE-20T`: Sequence-Aware Candidate Extraction & Pruning Defect

- **Original Stage Identifier**: `Stage 20T`
- **Timeline**: 2026-11-15 – 2026-11-15 | **Status**: `COMPLETED (DEFECT REPAIRED IN STAGE 21A)` | **Confidence**: `VERIFIED`
- **Research Objective**: Extract N-best beam search candidate hypotheses for downstream reranking.
- **Scientific Hypothesis**: N-best candidate pools contain lower-error hypotheses that rerankers can exploit.
- **Predecessor Stage**: `STAGE-20S`
- **Input Data & Artifacts**: B-02C posteriors on dev_assessment.jsonl
- **Source Code & Pipelines**: `kashmiri_dataset/experiments/stage20t_sequence_aware_research/`
- **Models Loaded / Utilized**: `B-02C + KenLM`
- **Models / Checkpoints Trained**: `Candidate Extraction Pipeline`
- **Major Findings & Verified Results**: Candidate extraction with beam width 8 introduced an 84-word search pruning penalty due to narrow search space.
- **Failed Approaches & Anomalies**: Narrow beam width (W=8) caused search errors and excluded optimal hypotheses.
- **Relationship to Subsequent Research**: Directly motivated Stage 21A beam expansion.

### `STAGE-21A`: Candidate Parity Repair & Search Space Oracle Ceiling

- **Original Stage Identifier**: `Stage 21A`
- **Timeline**: 2026-11-15 – 2026-11-15 | **Status**: `COMPLETED` | **Confidence**: `VERIFIED`
- **Research Objective**: Restore Top-1 search parity by increasing beam width to 16 with history pruning and compute Oracle ceiling.
- **Scientific Hypothesis**: Beam width 16 restores 100% Top-1 parity and reveals the upper bound of reranking headroom.
- **Predecessor Stage**: `STAGE-20T`
- **Input Data & Artifacts**: dev_assessment.jsonl
- **Source Code & Pipelines**: `kashmiri_dataset/experiments/stage21a_post_b02c_reconstruction/`
- **Models Loaded / Utilized**: `B-02C + KenLM`
- **Models / Checkpoints Trained**: `Parity Verification Pipeline`
- **Major Findings & Verified Results**: Restored 100.0% exact Top-1 parity across all 1,001 utterances. Established the 16-best Oracle ceiling at 19.23% WER (3,240 errors), proving that 881 word errors (21.4% of total errors) represent recoverable search headroom.
- **Failed Approaches & Anomalies**: N/A
- **Relationship to Subsequent Research**: Provided verified candidate pools for Stage 21B and Stage 24 rerankers.

### `STAGE-21B`: Decoder Hyperparameter Calibration & Pilot Neural Ranking

- **Original Stage Identifier**: `Stage 21B`
- **Timeline**: 2026-11-15 – 2026-11-15 | **Status**: `COMPLETED` | **Confidence**: `CORROBORATED`
- **Research Objective**: Calibrate KenLM decoding parameters (alpha, beta) and train a pilot Candidate-Relative Neural Ranker.
- **Scientific Hypothesis**: Calibrated decoder parameters and neural candidate ranking close the gap toward the 19.23% Oracle ceiling.
- **Predecessor Stage**: `STAGE-21A`
- **Input Data & Artifacts**: N=16 candidate pools on dev_assessment
- **Source Code & Pipelines**: `kashmiri_dataset/experiments/stage21b_candidate_relative_ranking/`
- **Models Loaded / Utilized**: `B-02C + KenLM`
- **Models / Checkpoints Trained**: `Pilot CandidateRelativeRanker (312k params)`
- **Major Findings & Verified Results**: KenLM calibration (alpha=0.20, beta=-0.50) reduced errors to 4,089 (24.27% WER; -32 errors). Pilot neural ranker achieved 4,093 errors (24.30% WER; -28 errors). Statistical significance testing (p = 0.0847) showed this was a modest empirical trend requiring large-scale validation.
- **Failed Approaches & Anomalies**: Pilot ranker trained on small candidate sets had high variance.
- **Relationship to Subsequent Research**: Formulated the Stage 24 large-scale neural reranker training hypothesis.

### `STAGE-22A`: Conservative Tier A Data Expansion & Champion Model E22-01

- **Original Stage Identifier**: `Stage 22`
- **Timeline**: 2026-11-15 – 2026-11-15 | **Status**: `COMPLETED (NEW SOTA PROJECT CHAMPION)` | **Confidence**: `VERIFIED`
- **Research Objective**: Train new foundation models expanding training data to 166.39 hours with audited WAAZ conversational speech.
- **Scientific Hypothesis**: Adding audited conversational community speech improves out-of-domain and held-out generalization.
- **Predecessor Stage**: `STAGE-21B`
- **Input Data & Artifacts**: data_22a_conservative_tier_a.jsonl (97,143 clean utterances, 166.39h)
- **Source Code & Pipelines**: `kashmiri_dataset/experiments/stage22_data_expanded_training/`
- **Models Loaded / Utilized**: `facebook/wav2vec2-xls-r-300m`
- **Models / Checkpoints Trained**: `E22-01 (Champion), E22-02, E22-03, E22-04`
- **Major Findings & Verified Results**: E22-01 achieved 24.74% WER on dev_calibration (4,114 errors) and 24.94% WER on dev_a_disjoint (26,566 errors), significantly outperforming B-02C across both secular and crowdsourced speech (p < 0.0001). Checkpoint SHA-256: 0c18d60b985966da6f97db20f79211ce669deb6db19861bdedded1569f5560a2.
- **Failed Approaches & Anomalies**: Arms with aggressive learning rate schedules (E22-03/04) suffered minor overfitting on conversational vocabulary.
- **Relationship to Subsequent Research**: Certified as the permanent primary production model in Stage 22C.

### `STAGE-22B`: Stage 22B Decoder Grid Search & Independent Test Validation

- **Original Stage Identifier**: `Stage 22B`
- **Timeline**: 2026-11-15 – 2026-11-15 | **Status**: `COMPLETED` | **Confidence**: `VERIFIED`
- **Research Objective**: Perform 2D grid search over (alpha, beta, beam) for E22-01 and evaluate on independent test benchmarks.
- **Scientific Hypothesis**: Joint optimization of beam search parameters maximizes decoding accuracy on conversational Kashmiri.
- **Predecessor Stage**: `STAGE-22A`
- **Input Data & Artifacts**: dev_calibration.jsonl, dev_a_disjoint.jsonl, dev_b_waaz_val.jsonl
- **Source Code & Pipelines**: `kashmiri_dataset/experiments/stage22b_decoder_and_independent_test/`
- **Models Loaded / Utilized**: `E22-01 + KenLM`
- **Models / Checkpoints Trained**: `Calibrated Decoder Configuration`
- **Major Findings & Verified Results**: Identified optimal decoder parameters: alpha = 0.30, beta = -1.00, beam width W = 64. Confirmed robust generalization across four independent test partitions.
- **Failed Approaches & Anomalies**: Excessive beam widths (W > 128) increased latency by 4.2x without additional error reduction.
- **Relationship to Subsequent Research**: Informed Stage 22C certified inference runtime configuration.

### `STAGE-22C`: Inference Certification, Exact CTC Slicing & Chunking

- **Original Stage Identifier**: `Stage 22C`
- **Timeline**: 2026-11-15 – 2026-11-15 | **Status**: `COMPLETED (PRODUCTION CERTIFIED)` | **Confidence**: `VERIFIED`
- **Research Objective**: Mathematically certify inference runtime: exact CTC output frame slicing and long-audio streaming chunking.
- **Scientific Hypothesis**: Receptive field recurrence determines exact frame counts and prevents trailing token hallucinations; context-preserving chunking prevents OOM.
- **Predecessor Stage**: `STAGE-22B`
- **Input Data & Artifacts**: Recordings >= 25.0 seconds from test partitions
- **Source Code & Pipelines**: `kashmiri_dataset/experiments/stage22c_inference_certification/`
- **Models Loaded / Utilized**: `E22-01 + KenLM`
- **Models / Checkpoints Trained**: `None (Inference Certification Harness)`
- **Major Findings & Verified Results**: Proved 7-layer CNN receptive field recurrence (R=400 samples, stride 320). Heuristic round(N/320) leaked +1 zero-padded frame in 81.2% of pyctcdecode regressions; recursive exact slicing eliminated 100% of hallucinations. Method C2 chunking (18s core, 2s context) achieved 28.26% WER with peak VRAM <= 1.7GB, eliminating 1.5TB attention OOM crashes.
- **Failed Approaches & Anomalies**: Heuristic frame slicing round(N/320) generated spurious trailing tokens.
- **Relationship to Subsequent Research**: Underpins certified production inference in transcribe_kashmiri.py.

### `STAGE-23`: Acoustic-LM Interaction & Search Space Topology

- **Original Stage Identifier**: `Stage 23`
- **Timeline**: 2026-11-15 – 2026-11-15 | **Status**: `COMPLETED` | **Confidence**: `VERIFIED`
- **Research Objective**: Map search space topology, evaluate 32-best candidate pools, and measure beam opportunity rates.
- **Scientific Hypothesis**: Expanding beam search to Top-32 exposes an expanded Oracle ceiling and quantifies candidate opportunity rate.
- **Predecessor Stage**: `STAGE-22C`
- **Input Data & Artifacts**: dev_calibration.jsonl, dev_a_disjoint.jsonl
- **Source Code & Pipelines**: `kashmiri_dataset/experiments/stage23_acoustic_lm_interaction/`
- **Models Loaded / Utilized**: `E22-01 + KenLM`
- **Models / Checkpoints Trained**: `Diagnostic Analysis Engine`
- **Major Findings & Verified Results**: Top-32 Oracle ceiling reached 18.99% WER on calibration (-956 words / 5.75 pp headroom) and 19.24% WER on dev_a_disjoint (-6,065 words / 5.69 pp headroom). Exactly 26.52% of beam candidate sets contained an alternative hypothesis with strictly lower error than Top-1 KenLM, establishing the candidate reranking hypothesis.
- **Failed Approaches & Anomalies**: Unweighted candidate combinations failed to surpass calibrated KenLM Top-1.
- **Relationship to Subsequent Research**: Formulated the formal training targets for Stage 24 neural rerankers.

### `STAGE-24`: Actual Neural Candidate Reranker Training & Acoustic Continuation

- **Original Stage Identifier**: `Stage 24`
- **Timeline**: 2026-11-15 – 2026-11-16 | **Status**: `COMPLETED (DECISIVE NEGATIVE RESULT & PERMANENT CERTIFICATION)` | **Confidence**: `VERIFIED`
- **Research Objective**: Train CandidateRelativeRanker on GPU over 20,000 utterances and test acoustic continuation (E24-CTC-01).
- **Scientific Hypothesis**: A deep neural candidate reranker trained on 20k utterances will outperform KenLM shallow fusion.
- **Predecessor Stage**: `STAGE-23`
- **Input Data & Artifacts**: 20,000 utterances (324,977 reference words across 507 unique speakers)
- **Source Code & Pipelines**: `kashmiri_dataset/experiments/stage24_actual_reranker_training/train_stage24_reranker.py, train_stage24_ctc_control.py, evaluate_stage24_reranker.py`
- **Models Loaded / Utilized**: `E22-01`
- **Models / Checkpoints Trained**: `CandidateRelativeRanker (329k params), E24-CTC-01`
- **Major Findings & Verified Results**: Trained CandidateRelativeRanker across 4 epochs with monotonic loss descent (4.45 -> 4.14). Multi-partition evaluation proved that neither linear nor neural rerankers beat KenLM (calibration: 24.74% WER, p=0.738; holdout: 24.94% WER, p=0.308). Controlled acoustic continuation E24-CTC-01 slightly improved greedy acoustic WER (52.38% vs 52.54%) but degraded KenLM beam decoding (48.42% vs 46.12% WER).
- **Failed Approaches & Anomalies**: Candidate-level neural reranking failed to outperform KenLM shallow fusion despite extensive GPU training. Escalation decision: HALT candidate reranking permanently.
- **Relationship to Subsequent Research**: Permanently certified Stage 22 E22-01 + 4-gram KenLM as the undisputed Kashmiri ASR Champion.



### `STAGE-29`: Meta Omnilingual ASR CTC Adaptation & Scaling (300M & 1B v2)

- **Original Stage Identifier**: `Stage 29`
- **Timeline**: 2026-11-17 – 2026-11-18 | **Status**: `COMPLETED (DECISIVE BENCHMARK & PRODUCTION CHAMPION CONFIRMED)` | **Confidence**: `VERIFIED`
- **Research Objective**: Adapt and benchmark Meta's foundation models (`omniASR_CTC_300M_v2` and `omniASR_CTC_1B_v2`) on Kashmiri speech recognition under strictly matched conditions against historical champion `E22-01` across 12,307 test utterances (24.54 hours).
- **Scientific Hypothesis**: Multilingual pretraining across 1,000+ languages in Omnilingual CTC models will outperform monolingual/multilingual XLS-R representation in low-resource Perso-Arabic Kashmiri.
- **Predecessor Stage**: `STAGE-24` / `STAGE-22`
- **Input Data & Artifacts**:
  - `train_manifest_dataset_a.jsonl`: 97,165 records (166.39 hours), strictly matched to `E22-01`
  - `validation_manifest_dev_a_disjoint.jsonl`: 6,257 records (11.30 hours)
  - `validation_manifest_dev_b_waaz.jsonl`: 127 records (0.20 hours)
  - `test_manifest.jsonl`: 5,923 records (13.04 hours)
  - `omniASR_tokenizer_written_v2.model`: 10,288 SentencePiece vocabulary
- **Source Code & Pipelines**: `experiments/stage29_omnilingual_kashmiri_asr/`
  - Model Architecture: `scripts/model.py` (Native pure-PyTorch ARM64 implementation mapping 319 / 607 fairseq2 parameter tensors)
  - Training Engine: `scripts/train.py` (BF16 mixed precision, gradient checkpointing, expandable segments)
  - Evaluation Suite: `scripts/evaluate.py`, `scripts/evaluate_suite.py` (Duration-aware chunking for memory safety)
  - Statistical Comparison: `scripts/compare_models.py` (10,000 paired bootstrap resamples, Wilcoxon signed-rank tests)
- **Models Loaded / Utilized**:
  - `E22-01` (Baseline Champion: Wav2Vec2 XLS-R 300M + Character CTC)
  - `omniASR-CTC-300M-v2.pt` (Pretrained 300M checkpoint, SHA-256: `8ce340ada22435d189908a8af67e3bb04899b6f2f5329d036248b9c3e38d2b50`)
  - `omniASR-CTC-1B-v2.pt` (Pretrained 1B checkpoint, SHA-256: `354f981756aa8f41591ea363e45b9c4eba1ec5144c2273af82e747efbb08919c`)
- **Models / Checkpoints Trained**:
  - `O29-01`: Fine-tuned `omniASR_CTC_300M_v2` (4,000 steps, best WER 38.09%, checkpoint: `checkpoints/O29-01/best_model.pt`, 1.9 GB)
  - `O29-02`: Fine-tuned `omniASR_CTC_1B_v2` (4,000 steps, best WER 43.98%, checkpoint: `checkpoints/O29-02/best_model.pt`, 5.5 GB)
- **Major Findings & Verified Results**:
  1. **Zero-Shot Baselines**: Both 300M and 1B Omnilingual models yield 100.00% WER / 100.00% CER in zero-shot mode, confirming that supervised acoustic adaptation is required for Perso-Arabic Kashmiri.
  2. **Pre-Norm Architecture Fix**: Proved that Meta fairseq2 models strictly require Pre-Norm Transformer layers (`norm_order = PRE`). Standard Post-Norm collapsed representations into blanks. Correcting to Pre-Norm enabled convergence to 0.0896 loss on Kashmiri.
  3. **Held-Out Test Set (`test_manifest`, 5,923 utts / 13.04h)**:
     - `E22-01`: **30.51% WER | 11.27% CER** (RTF: 0.0025x)
     - `O29-01 (300M)`: **40.72% WER | 14.97% CER** (RTF: 0.0034x)
     - `O29-02 (1B)`: **42.31% WER | 15.48% CER** (RTF: 0.0056x)
     - E22-01 retains statistically significant superiority ($\Delta = +10.21\%$, 95% CI `[+9.26%, +10.90%]`, $p = 1.0 \times 10^{-4}$).
  4. **Conversational Speech Parity (`dev_b_waaz_val`, 127 utts / 0.20h)**:
     - `E22-01`: **43.22% WER | 16.19% CER**
     - `O29-01 (300M)`: **46.23% WER | 16.39% CER** ($\Delta = +3.01\%$, 95% CI `[-0.40%, +6.23%]`, $p = 0.0834$)
     - `O29-02 (1B)`: **46.39% WER | 16.39% CER** ($\Delta = +3.17\%$, 95% CI `[-0.79%, +6.86%]`, $p = 0.1150$)
     - In both models, the 95% bootstrap CI crosses 0.0%, confirming **statistical equivalence** on spontaneous conversational Kashmiri. Omnilingual 1B recorded fewer central vowel errors (63) than E22-01 (68).
  5. **Model Scaling Findings (300M vs 1B)**: Scaling parameters from 326M to 976M did not improve performance (300M outperformed 1B: 40.72% vs 42.31% WER) while operating $1.65\times$ faster.
  6. **Root Cause**: The 10,288 SentencePiece subword vocabulary exhibits word-boundary splitting (6,036 and 7,442 boundary disparities vs 4,525 in E22-01).
- **Production Status**: **EduCasheer E22-01 remains confirmed as the Production Champion** for Kashmiri speech recognition.
