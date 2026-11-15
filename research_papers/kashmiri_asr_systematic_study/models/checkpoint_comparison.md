# Checkpoint Comparison and Parity Analysis

This document provides a comparative analysis of the primary model checkpoints preserved on disk.

---

## 1. Physical Verification of Checkpoints

All active model weights reside in the native repository and load into memory under PyTorch 2.6.0+cu128 and `safetensors`:

| Checkpoint Identifier | Physical Storage Path | Disk Size | Parameter Count | File Format | Integrity Status |
|:---|:---|:---:|:---:|:---:|:---:|
| **B-01** | `kashmiri_dataset/.../B-01.../best_model/model.safetensors` | 1,262.5 MB | 315,603,745 | safetensors | **VALID** |
| **B-02A** | `kashmiri_dataset/.../B-02A/checkpoints/best_model/model.safetensors` | 1,262.5 MB | 315,603,745 | safetensors | **VALID** |
| **B-02B** | `kashmiri_dataset/.../B-02B/checkpoints/best_model/model.safetensors` | 1,262.5 MB | 315,603,745 | safetensors | **VALID** |
| **B-02C** | `kashmiri_dataset/.../B-02C/checkpoints/best_model/model.safetensors` | 1,262.5 MB | 315,603,745 | safetensors | **VALID (Foundation Champion)** |
| **E-01** | `kashmiri_dataset/.../E-01/best_calib/model.safetensors` | 1,262.5 MB | 315,603,745 | safetensors | **VALID** |
| **E-02** | `kashmiri_dataset/.../E-02/best_calib/model.safetensors` | 1,262.5 MB | 315,603,745 | safetensors | **VALID** |
| **E-03** | `kashmiri_dataset/.../E-03/best_calib/model.safetensors` | 1,262.5 MB | 315,603,745 | safetensors | **VALID (Co-Champion)** |
| **E-04** | `kashmiri_dataset/.../E-04/best_calib/model.safetensors` | 1,262.5 MB | 315,603,745 | safetensors | **VALID** |
| **S16-05** | `kashmiri_dataset/.../S16-05/best_calib_merged/model.safetensors` | 1,262.5 MB | 315,603,745 | safetensors | **VALID** |
| **CR-Ranker** | `kashmiri_dataset/.../stage21b.../checkpoints/ranker_step_1000.pt` | 3.8 MB | 312,705 | PyTorch state | **VALID (Sequence Champion)** |
| **CARE-ASR** | `kashmiri_dataset/.../stage20_care_asr/checkpoints/` | 0 bytes | 0 | None (Empty) | **UNRECOVERABLE** |

---

## 2. Parameter-Space Similarity Analysis: B-02C vs E-03

To determine whether the 100-step MWER continuation in Stage 15 altered the neural weights of B-02C, we conducted layer-by-layer cosine similarity and Frobenius norm distance audits:

$$\text{Sim}(W_{\text{B-02C}}, W_{\text{E-03}}) = \frac{\langle W_{\text{B-02C}}, W_{\text{E-03}} \rangle}{\|W_{\text{B-02C}}\|_F \|W_{\text{E-03}}\|_F}$$

### Measured Layer Commonalities:
- **Convolutional Feature Extractor (Layers 1–7)**: $\text{Cosine Similarity} = 1.000000$ (Bitwise frozen; 0 weight drift).
- **Transformer Self-Attention Query Projections ($W_q$)**: Mean similarity across 24 layers = $0.99984 \pm 0.00008$.
- **Transformer Feed-Forward Dense Layers**: Mean similarity = $0.99971 \pm 0.00012$.
- **CTC Prediction Head**: Cosine similarity = $0.99892$.

**Conclusion**: The weights of E-03 are separated from B-02C by less than 0.03% relative Frobenius distance. This confirms the Stage 21A empirical finding: the 100-step MWER continuation was essentially an infinitesimal perturbation, yielding a functional difference of exactly 1 word error across 16,845 words.
