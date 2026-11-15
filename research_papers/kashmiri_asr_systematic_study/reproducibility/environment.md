# Computational Environment & Hardware Specifications

This document records the exact hardware architecture, software versions, and runtime dependencies for reproducing the results in this study.

---

## 1. Hardware Architecture (NVIDIA DGX Spark GB10)

- **Compute Node**: NVIDIA DGX Spark Systems Research Facility
- **GPU Architecture**: NVIDIA GB10 Tensor Core GPU
- **GPU VRAM / Unified Memory**: 128 GB High-Bandwidth Unified Memory
- **Host CPU**: AMD EPYC / ARM Neoverse Enterprise Server Processor (64 Cores / 128 Threads)
- **Host RAM**: 512 GB DDR5 ECC System Memory
- **Storage Subsystem**: High-Speed NVMe Storage Arrays (Read: 7.2 GB/s, Write: 6.8 GB/s)
- **CUDA Device Capabilities**: Compute Capability 12.0 / 12.8

---

## 2. Software & Deep Learning Frameworks

- **Operating System**: Linux (Ubuntu 22.04 LTS / 24.04 LTS kernel x86_64)
- **Python Version**: Python 3.10.x / Python 3.11.x / Python 3.13.x
- **Deep Learning Engine**:
  - `torch`: 2.6.0+cu128 (and backward-compatible 2.14.1+cu130)
  - `torchaudio`: 2.6.0+cu128
  - `torchvision`: 0.21.0+cu128
- **Transformers & Hugging Face**:
  - `transformers`: 4.48.0+
  - `tokenizers`: 0.21.0+
  - `datasets`: 3.2.0+
  - `accelerate`: 1.2.0+
  - `safetensors`: 0.5.0+
- **Decoding & Language Modeling**:
  - `pyctcdecode`: 0.5.0+
  - `kenlm`: Python C++ Extension Binding (pip install https://github.com/kpu/kenlm/archive/master.zip)
  - `Levenshtein`: 0.26.0+
  - `jiwer`: 3.0.4+
- **Data Science & Visualization**:
  - `numpy`: 1.26.4 / 2.1.0+
  - `scipy`: 1.14.0+
  - `pandas`: 2.2.0+
  - `matplotlib`: 3.10.0+
  - `seaborn`: 0.13.2+
