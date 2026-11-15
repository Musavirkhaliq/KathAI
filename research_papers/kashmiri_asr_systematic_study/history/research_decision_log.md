# Research Decision Log: Critical Strategic Shifts

This document records the major scientific and methodological pivot points across the Kashmiri ASR project, documenting the rationale, the empirical evidence observed, the risk identified, and the resulting architectural decision.

---

## Decision 1: Monolingual Adaptation vs Multilingual Prompting
- **Date**: November 2, 2026 (Stage 1–2)
- **Context**: Zero-shot evaluation of OpenAI Whisper models (Whisper-small and Whisper-large-v3) demonstrated catastrophic hallucination (>84% to 107% WER). Whisper lacked a dedicated Kashmiri `<|ks|>` token, causing cross-lingual looping into Slavic or Devanagari Hindi text.
- **Decision**: Reject black-box zero-shot foundation models as unusable for native Kashmiri speech. Dedicate all compute resources to fine-tuning self-supervised multilingual encoders (`facebook/wav2vec2-xls-r-300m`) with a custom monolingual Kashmiri character vocabulary.
- **Impact**: Established the foundation architecture that powered all subsequent research stages.

## Decision 2: 159-Token Character CTC vs Subword BPE
- **Date**: November 3, 2026 (Stage 3–4)
- **Context**: Subword SentencePiece tokenizers (8k–32k vocabularies) exhibited severe fragmentation on vowel diacritics (`U+0672`, `U+065F`, `U+065A`), leading to excessive sequence lengths, high fertility, and CTC alignment breakdown.
- **Decision**: Construct an audited 159-entry Character CTC Tokenizer encoding every base Perso-Arabic consonant, central vowel, and ligature, achieving 0.00% OOV.
- **Impact**: Completely eliminated the "tokenization tax" and stabilized CTC acoustic alignment.

## Decision 3: Incremental Acoustic Scaling Strategy
- **Date**: November 4–8, 2026 (Stage 8–9)
- **Context**: Evaluating whether to train on all scraped audio simultaneously or execute controlled scaling stages.
- **Decision**: Perform controlled step-wise data expansion: B-01 (92.9h) $\rightarrow$ B-02A (104.9h) $\rightarrow$ B-02B (123.5h) $\rightarrow$ B-02C (165.3h).
- **Impact**: Provided empirical proof of power-law acoustic scaling in Kashmiri ASR, lowering validation WER from 33.86% to 27.84% ($p < 0.0001$).

## Decision 4: Invalidation of Synthetic Multi-Condition Augmentation
- **Date**: November 14, 2026 (Stage 12)
- **Context**: Model R-02 trained with aggressive synthetic reverberation, additive noise, and Opus bit-rate throttling regressed from 25.24% to 28.94% KenLM WER on clean speech.
- **Decision**: Halt synthetic noise augmentation during primary model training. Preserve clean acoustic representations.
- **Impact**: Prevented degradation of fine phonetic distinctions in low-resource speech.

## Decision 5: Halting Layer-Unfreezing and Auxiliary Loss Exploration
- **Date**: November 14, 2026 (Stage 13)
- **Context**: Progressive layer unfreezing (U-01..03), Consistency Regularization (CR-01), and Intermediate CTC (I-01) all yielded 24.91%–25.23% WER, failing to beat full-corpus B-02C (24.47%).
- **Decision**: Freeze encoder unfreezing experiments and conclude that architectural auxiliary losses cannot substitute for full-corpus acoustic convergence.
- **Impact**: Re-focused compute on sequence optimization and decoding rather than encoder surgery.

## Decision 6: The Sample-Rate Forensic Pivot
- **Date**: November 15, 2026 (Stage 20R/20S)
- **Context**: Stage 20 reported a 482-error improvement with zero weight changes at step 0. Audit revealed that legacy evaluation scripts passed 22.05 kHz Bolbosh audio unresampled into 16 kHz models.
- **Decision**: Formulate a mandatory canonical audio contract: all audio must pass through a Kaiser windowed sinc interpolator (`canonical_audio.py`) before inference. Re-evaluate all historical checkpoints.
- **Impact**: Resolved historical inconsistencies; eliminated 493 phantom word errors; established that B-02C and E-03 are statistically identical ($p = 0.963$).

## Decision 7: Decoder Parity Enforcement & Beam Width Contract
- **Date**: November 15, 2026 (Stage 21A)
- **Context**: Stage 20T candidate extraction used beam width 8, introducing 84 search pruning errors.
- **Decision**: Mandate `beam_width=16` with history pruning as the strict baseline contract for all candidate generation and rescoring pipelines.
- **Impact**: Restored 100.0% Top-1 parity with canonical benchmarks across all 1,001 assessment samples.

## Decision 8: Scientific Scoping of the Manuscript
- **Date**: November 15, 2026 (Paper Planning)
- **Context**: Considering whether to claim a "novel sequence reranking breakthrough" based on Stage 21B results (24.30% vs 24.46% WER).
- **Decision**: Rigorous statistical testing ($p = 0.0847$, 95% CI $[-0.402, +0.072]$) confirmed that the 28–32 error gain is marginally significant. Frame the paper honestly around **heterogeneous data consolidation, evaluation integrity, and systematic empirical benchmarking**, highlighting negative results and preprocessing integrity as core scientific contributions.
- **Impact**: Protects scientific validity, ensures high publication credibility, and prevents over-claiming.

## Decision 9: Data Expansion and Acoustic Champion E22-01
- **Date**: November 15, 2026 (Stage 22A)
- **Context**: Historical acoustic models plateaued at 165.3 hours. Community data collection via WAAZ yielded high-confidence speech that expanded the verified Tier A dataset to 97,143 utterances (166.39 hours).
- **Decision**: Train a full data-expanded acoustic model (`E22-01`) on XLS-R 300M with character CTC.
- **Impact**: Established a new foundation acoustic champion (24.74% on calibration, 24.94% on disjoint dev, 46.12% on out-of-domain conversational speech, 28.99% on historical test reference).

## Decision 10: Mathematical Certification of Inference and Long-Audio Chunking
- **Date**: November 15, 2026 (Stage 22C)
- **Context**: Unblinded test evaluation in Stage 22B uncovered trailing token hallucinations in pyctcdecode and an operating system OOM crash on a 546-second sermon audio generating a 1.5 TB attention matrix.
- **Decision**: 
  1. Mathematically derive exact convolutional feature extractor lengths ($R=400$, stride 320) and replace approximate `round(N/320)` with `compute_exact_ctc_length(N)` to eliminate trailing padding hallucinations.
  2. Mandate Method C2 Context-Preserving Central Frames (18s core, 2s context) for audio $> 25.0$s to cap peak VRAM $\le 1.7$ GB.
- **Impact**: Restored 100.0% deterministic bitwise parity, eliminated trailing hallucinations, and completely prevented memory crashes during batched inference.

## Decision 11: Characterizing the Beam Search Headroom Hypothesis
- **Date**: November 15, 2026 (Stage 23)
- **Context**: Expanding KenLM beam width to 64 uncovered a 5.7 percentage point oracle headroom (18.99% vs 24.74% on calibration; 19.24% vs 24.94% on disjoint dev), with 26.52% of beams containing a superior hypothesis.
- **Decision**: Frame hypothesis reranking as an empirical research question: can a differentiable neural sequence ranker trained on supervised N-best beams learn to unlock this headroom without altering the frozen acoustic backbone?
- **Impact**: Designed a rigorous training protocol requiring 20,000 training-only utterances, Step 0 baseline parity, bounded protected fusion, and paired bootstrap significance testing.

## Decision 12: Empirical Rejection of Hypothesis Reranking and Final Benchmark Certification
- **Date**: November 16, 2026 (Stage 24)
- **Context**: Executed actual GPU training of the Candidate-Relative Neural Sequence Ranker (`CandidateRelativeRanker`, 329,025 params) and controlled acoustic continuation (`E24-CTC-01`). Despite monotonic loss reduction ($4.45 \rightarrow 4.15$), the reranker produced no statistically significant gain over KenLM ($p = 0.738$ on calibration, $p = 0.308$ on disjoint dev). Acoustic continuation slightly improved greedy WER (52.38% vs 52.54%) but degraded KenLM beam decoding (48.42% vs 46.12%).
- **Decision**: Enforce the mandatory escalation policy: **HALT post-decoder candidate reranking permanently.** Do not create further speculative post-processing stages. Certify Stage 22 E22-01 + 4-gram KenLM as the permanent champion of the project.
- **Impact**: Preserves scientific integrity by establishing an empirical negative result that prevents wasted effort on post-beam reranking, definitively directing future research toward acoustic mixture diversity and autoregressive language modeling.
