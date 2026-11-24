# Stage 29 Kashmiri ASR Training Report: Meta Omnilingual ASR CTC 1B / 300M v2

**Experiment ID:** Stage 29 Omnilingual ASR Training  
**Date:** November 17–18, 2026  
**Hardware:** NVIDIA DGX Spark (NVIDIA GB10, Compute Capability 12.1, 128 GB Unified Memory, Linux aarch64, CUDA 13.0)  
**Precision:** Native BF16 mixed precision with Gradient Checkpointing and Expandable Segments  

---

## 1. Experimental Training Matrix & Execution Status

| Experiment ID | Model Architecture | Parameters | Training Dataset | Duration (Hours) | LR Schedule | Effective Batch | Total Steps | Status | Best Val WER | Final Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **O29-00A** | omniASR_CTC_300M_v2 | 326M | None (Zero-Shot) | 0.0h | N/A | N/A | 0 | Completed | 100.00% | Established zero-shot reference |
| **O29-00B** | omniASR_CTC_1B_v2 | 976M | None (Zero-Shot) | 0.0h | N/A | N/A | 0 | Completed | 100.00% | Established zero-shot reference |
| **O29-01** | omniASR_CTC_300M_v2 | 326M | Dataset A (Matched) | 166.39h (97,165 utts) | AdamW (1e-4 $\rightarrow$ 1e-6) | 32 (32 $\times$ 1) | 4,000 | **COMPLETED** | **38.09%** | Best checkpoint serialized |
| **O29-02** | omniASR_CTC_1B_v2 | 976M | Dataset A (Matched) | 166.39h (97,165 utts) | AdamW (5e-5 $\rightarrow$ 5e-7) | 32 (16 $\times$ 2) | 4,000 | **COMPLETED** | **43.98%** | Best checkpoint serialized |

---

## 2. Preflight Gate Verification

### Preflight Step A: One-Batch Overfitting Gate
- **Batch Size:** 4 samples
- **Iterations:** 40 optimization steps (AdamW, LR 1e-4)
- **Initial CTC Loss:** 37.18
- **Final CTC Loss:** 4.61 (-87.6% drop)
- **Outcome:** **PASSED**. Acoustic encoder, projection layer, and CTC loss successfully gradient-converge.
- **Critical Fix:** Identified fairseq2 pre-norm requirement (`x = x + attn(norm(x))`). Single-batch loss plummeted from 11.82 to 0.0896 with exact transcription `'اسلام علیکم'`.

### Preflight Step B: Pilot Training & Checkpoint Serialization Gate
- **Steps:** 10 optimization steps
- **Validation Evaluation:** Evaluated across 500 `dev_a_disjoint` and 20 `dev_b_waaz` samples.
- **Checkpoint Serialization:** `best_model.pt` and `latest_checkpoint.pt` verified for both 300M (1.9 GB) and 1B (5.5 GB) architectures.
- **Outcome:** **PASSED**. Multi-sample batching, gradient accumulation, and checkpoint save/load logic verified.

---

## 3. O29-01 Execution Profile (omniASR_CTC_300M_v2)

- **Architecture:** 24 Transformer layers, 1024 model dim, 4096 FFN dim, 16 heads (326M parameters).
- **Base Checkpoint:** [omniASR-CTC-300M-v2.pt](file:///home/musa/Kashmiri%20Project/experiments/stage29_omnilingual_kashmiri_asr/checkpoints/omni_300m_v2/omniASR-CTC-300M-v2.pt) (SHA-256: `8ce340ada22435d189908a8af67e3bb04899b6f2f5329d036248b9c3e38d2b50`).
- **Tokenizer:** [omniASR_tokenizer_written_v2.model](file:///home/musa/Kashmiri%20Project/experiments/stage29_omnilingual_kashmiri_asr/assets/omniASR_tokenizer_written_v2.model) (SentencePiece, 10,288 vocab, SHA-256: `8aa11a1092142ef472537476ef6e76541123e2f0d789b79f3ebd119008240b1e`).
- **Feature Extractor (CNN):** 7 temporal convolution blocks frozen for initial 1,000 steps; unfrozen at step 1,000 for end-to-end adaptation.
- **Batching:** Micro-batch 32 $\times$ 1 accumulation = 32 effective batch size.
- **Training Progression:**
  - Initial Loss (Step 25): **233.99**
  - Step 500 Loss: **74.52**
  - Step 1000 Loss (Unfreeze): **28.16**
  - Step 2000 Loss: **18.42**
  - Step 3000 Loss: **14.15**
  - Step 4000 Loss: **11.16**
- **Hardware Profile:**
  - GPU VRAM: 20.73 GB peak allocated.
  - Compute Speed: ~4.0s per step (0.25 steps/s, ~8.0 utterances/s).
  - DGX Spark Stability: 0 crashes, waaz-asr service and desktop apps protected throughout.
- **Checkpoints Saved:**
  - `experiments/stage29_omnilingual_kashmiri_asr/checkpoints/O29-01/best_model.pt` (1.9 GB)
  - `experiments/stage29_omnilingual_kashmiri_asr/checkpoints/O29-01/latest_checkpoint.pt` (1.9 GB)

---

## 4. O29-02 Execution Profile (omniASR_CTC_1B_v2)

- **Architecture:** 48 Transformer layers, 1280 model dim, 5120 FFN dim, 16 heads (975.67M parameters).
- **Base Checkpoint:** [omniASR-CTC-1B-v2.pt](file:///home/musa/Kashmiri%20Project/experiments/stage29_omnilingual_kashmiri_asr/checkpoints/omni_1b_v2/omniASR-CTC-1B-v2.pt) (SHA-256: `354f981756aa8f41591ea363e45b9c4eba1ec5144c2273af82e747efbb08919c`).
- **Feature Extractor (CNN):** 7 temporal convolution blocks frozen for initial 1,000 steps; unfrozen at step 1,000.
- **Batching:** Micro-batch 16 $\times$ 2 accumulation = 32 effective batch size.
- **Validation Convergence History:**
  - Step 500: Disjoint WER: **67.40%** | WAAZ WER: **62.97%**
  - Step 1000: Disjoint WER: **55.16%** | WAAZ WER: **52.66%**
  - Step 1500: Disjoint WER: **48.80%** | WAAZ WER: **48.93%**
  - Step 2000: Disjoint WER: **46.68%** | WAAZ WER: **46.39%**
  - Step 2500: Disjoint WER: **44.94%** | WAAZ WER: **46.23%**
  - Step 3000: Disjoint WER: **44.27%** | WAAZ WER: **46.15%**
  - Step 3500: Disjoint WER: **43.98%** (Best) | WAAZ WER: **46.31%**
  - Step 4000: Disjoint WER: **44.15%** | WAAZ WER: **46.23%**
- **Hardware Profile:**
  - GPU VRAM: 17.59 GB peak allocated during training.
  - Compute Speed: ~9.1s per step (0.11 steps/s, ~3.5 utterances/s).
  - DGX Spark Stability: Ran continuously for 4,000 steps without system interruption.
- **Checkpoints Saved:**
  - `experiments/stage29_omnilingual_kashmiri_asr/checkpoints/O29-02/best_model.pt` (5.5 GB)
  - `experiments/stage29_omnilingual_kashmiri_asr/checkpoints/O29-02/latest_checkpoint.pt` (5.5 GB)
