# Stage 29 Research Environment & Dependency Audit

**Platform**: NVIDIA DGX Spark  
**Host Architecture**: ARM64 / aarch64 (Cortex-X925 + Cortex-A725, 20 physical cores)  
**Kernel**: Linux 6.8.0-1008-nvidia-64k  
**CUDA Version**: 13.0 (NVIDIA Driver: 580.178.04)  
**Accelerator**: NVIDIA GB10 (Compute Capability 12.1, 121.69 GiB Unified Address Space)  
**Local Storage**: NVMe SSD, 1.9 TiB available on `/`  
**Date**: November 17, 2026  

---

## 1. System Hardware Specifications

| Component | Specification | Operational Status |
| :--- | :--- | :--- |
| **GPU Model** | NVIDIA GB10 | Online, 2% idle utilization, 43°C |
| **GPU Memory** | 121.69 GiB Unified Memory | 86 GiB system RAM available, 0 MB GPU alloc at init |
| **CPU Architecture** | ARMv9.2-A aarch64 | 20 cores (10 Cortex-X925 + 10 Cortex-A725) |
| **Precision Support** | FP32, FP16, BF16, FP8 | BF16 verified functional on PyTorch sm_121 |
| **Storage Mount** | `/dev/nvme0n1p2` (NVMe) | 3.7T total, 1.7T used, 1.9T available (46% used) |

---

## 2. Software Dependencies & ARM64 Resolution

| Library | Version | Architecture / Wheel | Notes |
| :--- | :--- | :--- | :--- |
| **Python** | 3.13.9 (conda base) / 3.12.3 (system) | aarch64 | Native ARM64 |
| **PyTorch** | 2.14.1+cu130 | aarch64 (CUDA 13.0) | Full CUDA & Blackwell GB10 acceleration |
| **Torchaudio** | 2.11.1 | aarch64 | Native audio I/O & resampling |
| **SentencePiece** | 0.2.0 | aarch64 | Loads `omniASR_tokenizer_written_v2.model` |
| **Transformers** | 5.5.3 | any | HuggingFace model inspection & pipelines |
| **Numpy** | 2.2.0 | aarch64 | Numerical computing |
| **KenLM** | 0.2.0 | aarch64 | 4-gram domain language model scoring |
| **SoundFile** | 0.13.1 | aarch64 (libsndfile 1.2.2) | 16kHz WAV audio loading |

---

## 3. ARM64 Dependency Isolation Strategy

Meta's official upstream `fairseq2` library currently lacks precompiled PyPI Linux wheels for `fairseq2n` on `aarch64`. 
To ensure 100% reproducibility and prevent any corruption to the existing production ASR environment:
1. The model architecture is executed directly using native PyTorch (`nn.Module`), implementing the exact Wav2Vec2 CTC computational graph specified by Meta FAIR (`Wav2Vec2AsrModel` with 7-layer CNN frontend, 24/48-layer Transformer encoder, and 10,288-class linear projection).
2. The tokenizer is instantiated using official Google SentencePiece (`sentencepiece.SentencePieceProcessor`), reading the official Meta `omniASR_tokenizer_written_v2.model` checkpoint with identical token mappings.
3. This completely decouples training and inference from brittle C++ shared-library bindings while remaining bit-for-bit mathematically identical to the official Meta recipe.
