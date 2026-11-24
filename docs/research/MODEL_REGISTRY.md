# Kashmiri AI Central Model & Checkpoint Registry

**Project**: EduCasheer Kashmiri Automatic Speech Recognition Research
**Hardware Platform**: NVIDIA DGX Spark (NVIDIA GB10 GPU, 128GB Unified Memory)
**Last Updated**: November 16, 2026

---

## 1. Executive Summary: What Model to Use Today

| Operational Role | Champion Model | Best Checkpoint Path | Verified Accuracy | Recommended Deployment |
|:---|:---|:---|:---:|:---|
| 🏆 **Primary Production Champion** | **Stage 22 E22-01** | `kashmiri_dataset/experiments/stage22_data_expanded_training/checkpoints/E22-01/best_model` | **24.74% WER (Calib) / 24.94% (Holdout)** | **Default ASR Engine.** Use for general conversation, read speech, and community recordings. |
| 🏛️ **Foundation Research Baseline** | **Stage 9 B-02C** | `kashmiri_dataset/experiments/native_asr_training/experiments/stage9_b02_acoustic_comparison/B-02C/checkpoints/best_model` | **25.24% WER (val.jsonl) / 24.47% (Dev Assess)** | Historical research reference. Used for scientific comparison against prior publications. |
| 📖 **Religious & Classical Kashmiri** | **Stage 9 B-02B** | `kashmiri_dataset/experiments/native_asr_training/experiments/stage9_b02_acoustic_comparison/B-02B/checkpoints/best_model` | **26.41% WER (val.jsonl)** | Specialized engine for Qur'anic commentaries and Arabic-Kashmiri code-mixing. |
| 🔤 **Dual Romanization Service** | **ByT5 Adapter** | `kashmiri_dataset/experiments/transliteration_evaluation/byt5_adapter/best_checkpoint` | **94.2% Exact Match** | Converts Perso-Arabic transcripts into phonetic Roman/IAST for search and indexing. |

---

## 2. Checkpoint Details & Cryptographic Provenance

### Stage 22 E22-01 (Certified Project Champion)

- **Research Stage**: `Stage 22A`
- **Architecture**: `facebook/wav2vec2-xls-r-300m + Character CTC` (315,603,745 parameters, vocab: 159)
- **Training Dataset**: 166.39 hours (97,143 clean utterances, conservative Tier A + WAAZ)
- **Local Checkpoint Directory**: `kashmiri_dataset/experiments/stage22_data_expanded_training/checkpoints/E22-01/best_model`
- **Weights File**: `model.safetensors` (1,262,467,532 bytes)
- **SHA-256 Checksum**: `0c18d60b985966da6f97db20f79211ce669deb6db19861bdedded1569f5560a2`
- **Verified Benchmark Metrics**: 24.74% WER (dev_calib), 24.94% WER (dev_a_disjoint), 46.12% WER (dev_b_waaz)
- **Deployment Status**: `OFFICIAL PRODUCTION CHAMPION`
- **Reproduction Command**: `python3 kashmiri_dataset/experiments/stage22c_inference_certification/inference/reproduce_certified_stage22c_benchmarks.py`

### Foundation Model B-02C (Acoustic Baseline)

- **Research Stage**: `Stage 9`
- **Architecture**: `facebook/wav2vec2-xls-r-300m + Character CTC` (315,603,745 parameters, vocab: 159)
- **Training Dataset**: 165.32 hours (96,664 utterances, Bolbosh + IndicVoices-R + OpenSLR)
- **Local Checkpoint Directory**: `kashmiri_dataset/experiments/native_asr_training/experiments/stage9_b02_acoustic_comparison/B-02C/checkpoints/best_model`
- **Weights File**: `model.safetensors` (1,262,467,532 bytes)
- **SHA-256 Checksum**: `Verified in provenance registry`
- **Verified Benchmark Metrics**: 27.84% Greedy WER / 25.24% KenLM WER on val.jsonl; 24.47% on dev_assessment
- **Deployment Status**: `FOUNDATION BASELINE (RETAINED FOR BENCHMARK COMPATIBILITY)`
- **Reproduction Command**: `python3 test_batch_transcribe.py`

### Religious Domain Model B-02B

- **Research Stage**: `Stage 9`
- **Architecture**: `facebook/wav2vec2-xls-r-300m + Character CTC` (315,603,745 parameters, vocab: 159)
- **Training Dataset**: 123.45 hours (80,106 utterances, includes Mirwaiz spoken Quran)
- **Local Checkpoint Directory**: `kashmiri_dataset/experiments/native_asr_training/experiments/stage9_b02_acoustic_comparison/B-02B/checkpoints/best_model`
- **Weights File**: `model.safetensors` (1,262,467,532 bytes)
- **SHA-256 Checksum**: `Verified in provenance registry`
- **Verified Benchmark Metrics**: 29.39% Greedy WER / 26.41% KenLM WER on val.jsonl (+1.52% religious recall)
- **Deployment Status**: `DOMAIN SPECIALIZED (RELIGIOUS ASR)`
- **Reproduction Command**: `python3 transcribe_kashmiri.py --model-dir .../B-02B/checkpoints/best_model`

### Proven Baseline Model B-02A

- **Research Stage**: `Stage 9`
- **Architecture**: `facebook/wav2vec2-xls-r-300m + Character CTC` (315,603,745 parameters, vocab: 159)
- **Training Dataset**: 104.93 hours (64,522 utterances)
- **Local Checkpoint Directory**: `kashmiri_dataset/experiments/native_asr_training/experiments/stage9_b02_acoustic_comparison/B-02A/checkpoints/best_model`
- **Weights File**: `model.safetensors` (1,262,467,532 bytes)
- **SHA-256 Checksum**: `Verified in provenance registry`
- **Verified Benchmark Metrics**: 29.42% Greedy WER / 26.30% KenLM WER on val.jsonl
- **Deployment Status**: `HISTORICAL BASELINE`
- **Reproduction Command**: `python3 transcribe_kashmiri.py --model-dir .../B-02A/checkpoints/best_model`

### Genesis Baseline Model B-01

- **Research Stage**: `Stage 8`
- **Architecture**: `facebook/wav2vec2-xls-r-300m + Character CTC` (315,603,745 parameters, vocab: 159)
- **Training Dataset**: 92.94 hours (53,460 utterances)
- **Local Checkpoint Directory**: `kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc/best_model`
- **Weights File**: `model.safetensors` (1,262,467,532 bytes)
- **SHA-256 Checksum**: `Verified in provenance registry`
- **Verified Benchmark Metrics**: 33.86% Greedy WER / 31.80% KenLM WER on val.jsonl
- **Deployment Status**: `GENESIS BASELINE (ARCHIVED)`
- **Reproduction Command**: `python3 transcribe_kashmiri.py --model-dir .../B-01-xlsr300m-character-ctc/best_model`

### ByT5 Transliteration Adapter

- **Research Stage**: `Stage 7`
- **Architecture**: `google/byt5-small Character Seq2Seq` (300,000,000 parameters, vocab: 256)
- **Training Dataset**: 64,800 RASA Roman-Perso-Arabic transliteration pairs
- **Local Checkpoint Directory**: `kashmiri_dataset/experiments/transliteration_evaluation/byt5_adapter/best_checkpoint`
- **Weights File**: `model.safetensors` (1,198,542,104 bytes)
- **SHA-256 Checksum**: `Verified in provenance registry`
- **Verified Benchmark Metrics**: 94.2% exact word transliteration accuracy
- **Deployment Status**: `ACTIVE UTILITY (DUAL ROMANIZATION SERVICE)`
- **Reproduction Command**: `python3 kashmiri_dataset/scripts/train_byt5_transliteration_adapter.py`

### Candidate-Relative Neural Sequence Ranker

- **Research Stage**: `Stage 24`
- **Architecture**: `Char Embedding + 2-layer BiGRU (hidden 96) + Projection MLP` (329,025 parameters, vocab: 159)
- **Training Dataset**: 20,000 utterances (324,977 reference words across 507 unique speakers)
- **Local Checkpoint Directory**: `kashmiri_dataset/experiments/stage24_actual_reranker_training/checkpoints/best_calib_ranker.pt`
- **Weights File**: `best_calib_ranker.pt` (3,982,400 bytes)
- **SHA-256 Checksum**: `Verified in stage24 checkpoint registry`
- **Verified Benchmark Metrics**: 24.74% WER (dev_calib, p=0.738 vs KenLM), 24.94% WER (dev_a_disjoint, p=0.308 vs KenLM)
- **Deployment Status**: `RESEARCH COMPLETE (CONFIRMED EQUIVALENT TO KENLM, PERMANENTLY HALTED)`
- **Reproduction Command**: `python3 kashmiri_dataset/experiments/stage24_actual_reranker_training/evaluate_stage24_reranker.py`



### Meta Omnilingual ASR CTC 300M v2 (O29-01)

- **Research Stage**: `Stage 29`
- **Architecture**: `Meta Omnilingual ASR CTC 300M v2 + SentencePiece CTC` (326,227,888 parameters, vocab: 10,288)
- **Training Dataset**: 166.39 hours (97,165 utterances, Dataset A matched to E22-01)
- **Local Checkpoint Directory**: `experiments/stage29_omnilingual_kashmiri_asr/checkpoints/O29-01/best_model.pt`
- **Weights File**: `best_model.pt` (1,962,561,048 bytes)
- **Verified Benchmark Metrics**:
  - `dev_a_disjoint` (6,257 utts / 11.3h): 38.09% WER | 14.63% CER | 0.0034x RTF
  - `dev_b_waaz_val` (127 utts / 0.2h): 46.23% WER | 16.39% CER | 0.0033x RTF (Statistical parity with E22-01, p=0.0834)
  - `test_manifest` (5,923 utts / 13.04h): 40.72% WER | 14.97% CER | 0.0034x RTF
- **Deployment Status**: `RESEARCH COMPLETE (CONVERSATIONAL ALTERNATIVE / SPEED CHAMPION: 294x REAL-TIME)`
- **Reproduction Command**: `python3 experiments/stage29_omnilingual_kashmiri_asr/scripts/evaluate_suite.py --model-type 300m --checkpoint experiments/stage29_omnilingual_kashmiri_asr/checkpoints/O29-01/best_model.pt --exp-name O29-01`

### Meta Omnilingual ASR CTC 1B v2 (O29-02)

- **Research Stage**: `Stage 29`
- **Architecture**: `Meta Omnilingual ASR CTC 1B v2 + SentencePiece CTC` (975,671,216 parameters, vocab: 10,288)
- **Training Dataset**: 166.39 hours (97,165 utterances, Dataset A matched to E22-01)
- **Local Checkpoint Directory**: `experiments/stage29_omnilingual_kashmiri_asr/checkpoints/O29-02/best_model.pt`
- **Weights File**: `best_model.pt` (5,862,298,728 bytes)
- **Verified Benchmark Metrics**:
  - `dev_a_disjoint` (6,257 utts / 11.3h): 39.42% WER | 14.93% CER | 0.0056x RTF
  - `dev_b_waaz_val` (127 utts / 0.2h): 46.39% WER | 16.39% CER | 0.0058x RTF (Fewer central vowel errors than E22-01: 63 vs 68)
  - `test_manifest` (5,923 utts / 13.04h): 42.31% WER | 15.48% CER | 0.0056x RTF
- **Deployment Status**: `RESEARCH COMPLETE (SCALING AUDIT COMPLETE; SURPASSED BY 300M CANDIDATE)`
- **Reproduction Command**: `python3 experiments/stage29_omnilingual_kashmiri_asr/scripts/evaluate_suite.py --model-type 1b --checkpoint experiments/stage29_omnilingual_kashmiri_asr/checkpoints/O29-02/best_model.pt --exp-name O29-02`
