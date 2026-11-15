# Stage 22C Scientific Addendum: Inference Integrity, Exact CTC Slicing, and Long-Audio Certification
**EduCasheer Kashmiri AI Systematic Study — Workstream 8, Stage 22C**  
*Principal Speech Recognition Scientist, Computational Linguist, & Reproducibility Auditor*  
*Date: November 15, 2026*

---

## 1. Abstract & Context of Addendum

This addendum documents the post-training inference integrity audit and scientific certification of the Stage 22 Kashmiri ASR research campaign. Stage 22 introduced community-collected Kashmiri speech via the WAAZ platform, fine-tuning the historical benchmark checkpoint (Stage 15 E-03) on 166.39 hours of audited acoustic data to produce candidate checkpoint **E22-01**.

Subsequent evaluation revealed critical implementation anomalies in third-party decoding libraries and batching harnesses:
1. **The Padded Logit Accumulation Anomaly**: In pyctcdecode language model decoding, zero-padded acoustic frames caused trailing single-token hallucinations on 81.2% of regression cases.
2. **The Receptive Field Slicing Error**: Approximate temporal slicing `round(N / 320.0)` systematically retains exactly $+1$ frame of zero-padded acoustic context in batched inference due to the $400$-sample receptive field of the Wav2Vec2 convolutional feature extractor.
3. **The Long-Audio Attention Memory Spike**: Unconstrained batching containing a $546.6$-second sermon recording resulted in a $1.5$-terabyte intermediate attention tensor allocation, triggering the operating system Out-Of-Memory (OOM) killer.

This addendum details the mathematical derivation of exact output lengths, benchmarks four long-audio chunking strategies, documents test-set provenance, and certifies the final reproducible benchmark metrics.

---

## 2. Mathematical Derivation of Exact CTC Output Lengths

The `facebook/wav2vec2-xls-r-300m` convolutional encoder comprises seven 1D convolutional layers with unpadded kernels ($P_i = 0$, $D_i = 1$):
- Kernel sizes: $K = [10, 3, 3, 3, 3, 2, 2]$
- Strides: $S = [5, 2, 2, 2, 2, 2, 2]$ (Cumulative stride = $320$)

The minimum receptive field is:
$$R = 1 + \sum_{i=0}^6 (K_i - 1) \prod_{j=0}^{i-1} S_j = 400 \text{ samples (0.025s at 16 kHz)}.$$

For any sample count $N \ge 400$, the valid output frame count through the composite convolutional feature extractor is given by:
$$L_0 = N, \quad L_{i+1} = \left\lfloor \frac{L_i - K_i}{S_i} \right\rfloor + 1 \quad \forall i \in \{0, \dots, 6\}.$$

Because $R = 400 > 320$, the heuristic formula $\text{round}(N / 320)$ evaluates to $\lfloor N / 320 \rceil = T_{\text{exact}} + 1$ for all standard audio lengths. In padded batches of size $B \times T_{\text{max}} \times V$, this approximation leaks exactly one frame of zero-padded noise into beam search. Deploying `compute_exact_ctc_length(N)` completely eliminates this artifact.

---

## 3. Development Benchmark of Long-Audio Chunking Strategies

Evaluated on all 16 recordings $\ge 25.0$s in `dev_a_legacy_val.jsonl` (up to 50.0s), using Uninterrupted Full-Audio Inference as the reference ground truth:

| Strategy | Mean WER (%) | Mean CER (%) | Exact Match Rate | Peak VRAM |
|---|:---:|:---:|:---:|:---:|
| **Uninterrupted Reference** | 28.85% | 11.68% | 16 / 16 (100.0%) | 1,892.3 MB |
| **Method C0 (Historical Non-Overlapping)** | 30.46% | 12.33% | 0 / 16 (0.0%) | 1,678.4 MB |
| **Method C1 (Overlapping Crossfade)** | 28.68% | 11.92% | 2 / 16 (12.5%) | 1,678.4 MB |
| **Method C2 (Context-Preserving Central Frames)** | **28.26%** | **11.36%** | **2 / 16 (12.5%)** | **1,687.2 MB** |
| **Method C3 (Silence-Aware VAD)** | 30.32% | 11.60% | 0 / 16 (0.0%) | 1,660.1 MB |

**Certified Inference Policy**:
- Audio $\le 25.0$ seconds (99.78% of corpus): Uninterrupted inference (Peak VRAM $\le 1.8$ GB).
- Audio $> 25.0$ seconds (0.22% of corpus): Method C2 Context-Preserving Central Frames (Core: 18.0s, Flanking Context: 2.0s, Peak VRAM $\le 1.7$ GB).

---

## 4. Test Set Provenance and Scientific Limits

The 5,923-utterance test partition (`test.jsonl`, 13.04 hours, 120,816 words) contains:
- **Legacy Core Test (5,251 utts)**: IndicVoices (2,731), Bolbosh (2,000), Whisper (520).
- **Secondary External Partitions (672 utts)**: XKaab (348), OpenSLR 122 (324).
- **Contamination Status**: Zero physical audio overlap (0.00%) and zero named speaker overlap (0.00%) with all training mixtures.
- **Independence Caveat**: Because `test.jsonl` was evaluated in Stage 22B to diagnose decoder issues, it has been previously unblinded. To preserve scientific validity, all decoding hyperparameters ($\alpha=0.30, \beta=-1.00$, beam 64) and chunking rules were frozen strictly on independent development sets (`dev_calibration.jsonl` and `dev_a_legacy_val.jsonl`).
