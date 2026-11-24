# Physical Storage & Filesystem Consumption Analysis
**EduCasheer Kashmiri AI Research Project**

## 1. Storage Overview

A complete filesystem audit conducted across `/home/musa/Kashmiri Project` and `/home/musa/Desktop/Projects/DataCollection` revealed **256,496 physical files** consuming **733.65 GB** of NVMe disk storage.

| Data Modality / File Category | Physical File Count | Disk Consumption (GB) | Percentage of Total Storage | Typical Formats |
| :--- | :--- | :--- | :--- | :--- |
| **Model Checkpoints** | 493 | 535.38 GB | 73.0% | `.pt`, `.safetensors`, `.ckpt` |
| **Columnar Datasets** | 536 | 121.44 GB | 16.6% | `.parquet`, `.arrow` (e.g. Koshur Pixel) |
| **Physical Audio Recordings** | 222,903 | 45.02 GB | 6.1% | `.wav` (PCM 16-bit), `.mp3` |
| **Compressed Archives** | 38 | 15.20 GB | 2.1% | `.tar.gz`, `.zip` |
| **Text Corpora & Manifests** | 8,271 | 8.65 GB | 1.2% | `.jsonl`, `.txt`, `.csv` |
| **Databases** | 5 | 1.09 GB | 0.1% | `.db` (SQLite WAAZ DB & backups) |
| **Documents & Companion Texts**| 112 | 0.67 GB | <0.1% | `.pdf`, `.html` (Kachru textbooks) |
| **Code & Scripts** | 1,846 | 0.02 GB | <0.1% | `.py`, `.sh` |
| **Other / Intermediate Logits**| 22,292 | 6.19 GB | 0.8% | `.npz`, `.log` |
| **TOTAL** | **256,496** | **733.65 GB** | **100.0%** | |

## 2. Logical vs. Physical Storage

- **Deduplication of Manifests vs. Audio**: Audio representations on disk total 45.02 GB. Multiple experiment manifests reference the same underlying audio files in `kashmiri_dataset/processed/audio/` without duplicating physical audio on disk.
- **Symlink Handling**: 24 symbolic links were detected across the repository (e.g. `DATASET_REGISTRY.md -> docs/DATASET_REGISTRY.md`). These are resolved to zero additional physical disk storage.
- **Google Drive Shard Decompression**: The 22 split zip shards in `Data/` contain redundant copies of raw audio that were subsequently extracted and standardized into `kashmiri_dataset/` and `data_acquisition/`.
