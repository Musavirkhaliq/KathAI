# EDUCASHEER KASHMIRI ASR: B-02A FOLLOW-UP HELD-OUT EVALUATION PROTOCOL

**Document Version:** 1.0 (Pre-Registered & Frozen Prior to Execution)  
**Date:** November 12, 2026  
**Status:** AUTHORIZED, FROZEN, AND LOCKED  
**Evaluation Scope:** One Controlled Follow-Up Evaluation on the Stage 6 Held-Out Test Set  
**Primary Endpoint:** Corpus-Level Normalized Word Error Rate (Norm WER) on Held-Out Test Set (5,172 Utterances / ~9.30 Audio Hours)  

---

## 1. Context & Scientific Authorization

This protocol governs an authorized, single-run follow-up evaluation of the frozen **B-02A (Epoch 19)** acoustic checkpoint on the frozen EduCasheer Kashmiri Stage 6 held-out test benchmark. 

### Mandatory Safeguards & Scientific Disclaimers
1. **Follow-Up Analysis on Previously Evaluated Test Set:** This evaluation is explicitly designated as a follow-up analysis on a previously evaluated test set. It does **not** replace or overwrite historical Stage 6 benchmarks, nor does it claim independent, uncontaminated proof of State-of-the-Art (SOTA) generalization.
2. **Strict Test Set Isolation:** The test set (`test.jsonl`) did not contribute any audio, features, or text to the training of acoustic model B-01, acoustic model B-02A, or language model KenLM.
3. **Development-Only Checkpoint Selection:** All model checkpoints were chosen strictly based on validation performance on `val.jsonl` (6,728 utterances / 11.05 audio hours):
   - **B-01:** Checkpoint Epoch 13 (`best_model`) selected based on dev validation loss and WER.
   - **B-02A:** Checkpoint Epoch 19 (`checkpoints/epoch_19`) selected based on minimum dev validation WER (29.37% Norm WER).
4. **Frozen Hyperparameters:** Language model shallow fusion weights ($\alpha = 0.25, \beta = 0.00$) and beam width ($W = 16$) were selected during Stage 7 exclusively on the development validation logit cache. No parameter tuning against test performance has occurred or will be permitted.
5. **Preservation of Historical Artifacts:** The historical Stage 6 baseline (32.34% Norm WER for B-01 under unmasked batched greedy CTC) and all historical Stage 6 reports, predictions, and logs remain completely unchanged.
6. **Strict Stopping Condition:** This evaluation is executed exactly once. No retraining, retuning, or repeated evaluation of subsequent checkpoints (e.g., B-02B, B-02C) on this held-out test set is authorized.

---

## 2. Checkpoint & Artifact Cryptographic Verification

All acoustic model weights, tokenizers, configurations, and language models are cryptographically verified prior to ingesting test audio:

| System / Component | Artifact | File Path | File Size (Bytes) | SHA-256 Checksum |
|---|---|---|---:|---|
| **B-01 (Baseline)** | Model Weights | `.../B-01-xlsr300m-character-ctc/best_model/model.safetensors` | 1,262,467,532 | `793b94bae5a09324212e3789575027a526d8e6dc61f7646ebeb65303eb990635` |
| **B-01 (Baseline)** | Vocabulary | `.../B-01-xlsr300m-character-ctc/best_model/vocab.json` | 1,924 | `fc295c91fede5cdbdc52719ab05ee4f01ccff8af32d45e7b0d8a3a387724b330` |
| **B-01 (Baseline)** | Config | `.../B-01-xlsr300m-character-ctc/best_model/config.json` | 2,032 | `5aa814397a739c26e58303223b70d5503bf5e564fa59136fc74bbc6594df43f7` |
| **B-02A (Epoch 19)** | Model Weights | `.../B-02A/checkpoints/epoch_19/model.safetensors` | 1,262,467,532 | `4676bb8c20a3d797b1b31ec19e02f515a6d52543d5897b28fb4dc54c136089f7` |
| **B-02A (Epoch 19)** | Vocabulary | `.../B-02A/checkpoints/epoch_19/vocab.json` | 1,924 | `fc295c91fede5cdbdc52719ab05ee4f01ccff8af32d45e7b0d8a3a387724b330` |
| **B-02A (Epoch 19)** | Config | `.../B-02A/checkpoints/epoch_19/config.json` | 2,032 | `5aa814397a739c26e58303223b70d5503bf5e564fa59136fc74bbc6594df43f7` |
| **B-02A (Epoch 19)** | Processor Config | `.../B-02A/checkpoints/epoch_19/processor_config.json` | 299 | `8bc3c8d6d97ea61524d5c15d8554b4baf072d2264c0bda1839bc7a822c200d66` |
| **B-02A (Epoch 19)** | Tokenizer Config | `.../B-02A/checkpoints/epoch_19/tokenizer_config.json` | 1,335 | `55421d15356f52af6b052836acf53c35ea5d635b377c324db134e8182534333e` |
| **Stage 7 KenLM** | 4-Gram LM (Domain D) | `.../stage7_ctc_lm_decoding/language_models/lm_word_4gram_domain_d.arpa` | 147,482,084 | `a70c5f79831f668e14fb9aa3ababd5b756c87d09c8cc09e671d5da69e1da61e0` |

---

## 3. Test Population & Pre-Registered Eligibility

The evaluation population is identical to the Stage 6 held-out benchmark:
- **Raw Test Set:** `kashmiri_dataset/experiments/native_asr_training/data_splits/test.jsonl` (5,923 utterances).
- **Inclusion Criteria:**
  1. $0.5\text{ s} \le \text{duration} \le 16.0\text{ s}$
  2. $0 < \text{len}(\text{NFKC}(\text{text})) < \lfloor \text{duration} \times 50 \rfloor$ (CTC acoustic feasibility)
- **Eligible Evaluated Population:** Exactly **5,172 utterances (~9.30 audio hours)**.
- **Set Identity:** Verified 100.00% pairwise sample ID identity with Stage 6 `b01_test_predictions.jsonl`.

### Source Distribution
- `humair025/KashmiriSpeech-IndicVoices`: 2,473 utterances (47.81%)
- `GAASH-Lab/Bolbosh`: 1,905 utterances (36.83%)
- `openslr_slr122`: 321 utterances (6.21%)
- `xkaab_asr_4hrs`: 320 utterances (6.19%)
- `muneebharoon/whisper-kashmiri`: 153 utterances (2.96%)

---

## 4. Matched Inference & Decoding Specifications

Both B-01 and B-02A will be evaluated across three matched decoders under identical conditions:

1. **Decoder 1: Corrected Masked Greedy CTC**
   - Active `attention_mask` passed to Wav2Vec2 encoder.
   - Dynamic batching with exact sequence slicing:
     $$T_{\text{conv}} = \text{model.\_get\_feat\_extract\_output\_lengths}(\text{len}(\text{audio}))$$
   - Logits sliced to $T_{\text{conv}}$ before argmax decoding, completely eliminating padding frame contamination.

2. **Decoder 2: CTC Prefix Beam Search (No LM)**
   - Beam width $W = 16$.
   - Lexicon-free character-level prefix beam search via `pyctcdecode`.
   - Log-probabilities sliced to $T_{\text{conv}}$.

3. **Decoder 3: CTC Prefix Beam Search + Stage 7 Kashmiri KenLM**
   - Beam width $W = 16$.
   - Frozen 4-gram word KenLM (`lm_word_4gram_domain_d.arpa`).
   - Shallow fusion weight $\alpha = 0.25$, word insertion bonus $\beta = 0.00$.
   - Log-probabilities sliced to $T_{\text{conv}}$.

---

## 5. Metrics & Statistical Significance Protocol

### Required Metrics
- **Normalized Word Error Rate (Norm WER)** and **Normalized Character Error Rate (Norm CER)**.
- **Raw Word Error Rate (Raw WER)** and **Raw Character Error Rate (Raw CER)**.
- **Word Error Breakdown:** Substitutions ($S$), Deletions ($D$), Insertions ($I$), and Hits ($H$).
- **Exact Sentence Matches:** Count and percentage of zero-WER utterances.
- **Empty Predictions:** Count and percentage of null hypotheses.
- **Inference Efficiency:** GPU forward latency, decoder-only latency, total evaluation time, RTF, and real-time throughput.
- **Source-Wise Disaggregation:** Individual metrics for all 5 subcorpora.

### Statistical Testing Protocol
- Paired bootstrap test with $B = 10,000$ resamples with replacement over the matched 5,172 utterances.
- 95% bootstrap confidence intervals for B-01, B-02A, and paired differences ($\Delta = \text{B-02A} - \text{B-01}$).
- Two-tailed empirical $p$-values:
  $$p = 2 \times \min\left(\frac{1}{B}\sum_{b=1}^B \mathbb{I}(\Delta_b \le 0), \frac{1}{B}\sum_{b=1}^B \mathbb{I}(\Delta_b \ge 0)\right)$$
- Absolute percentage-point change ($\Delta\text{ pp} = \text{WER}_{\text{B02A}} - \text{WER}_{\text{B01}}$) and Relative Word Error Rate Reduction:
  $$\text{WERR} = \frac{\text{WER}_{\text{B01}} - \text{WER}_{\text{B02A}}}{\text{WER}_{\text{B01}}} \times 100\%$$
