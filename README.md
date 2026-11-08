# KathAI (کتھ) — Kashmiri Speech & NLP Pipeline

KathAI is an end-to-end framework and research repository for **Kashmiri (Koshur)** speech recognition (ASR), Perso-Arabic text normalization, multimodal dataset curation, and acoustic model training (XLS-R, Whisper).

---

## 🌟 Key Features

- **Perso-Arabic Kashmiri Normalization**: Standardizes orthography, Arabic/Perso-Arabic diacritics, and detects code-switching (English/Urdu/Hindi).
- **Audio Validation & Standardization**: 16kHz PCM mono conversion, VAD segmentation, and SNR-based audio quality filtering.
- **Strict Speaker-Independent Splitting**: Guarantees zero speaker overlap across training, validation, and test partitions (`clean`, `noisy`, `code_switched`, `speaker_unseen`).
- **Subword & CTC Tokenizers**: Custom Kashmiri SentencePiece tokenizers (8k, 16k, 24k, 32k) and character CTC vocabularies.
- **Acoustic Model Training**: Training scripts and evaluation benchmarks for wav2vec2 / XLS-R CTC and OpenAI Whisper.

---

## 📁 Repository Structure

```text
.
├── kashmiri_dataset/
│   ├── documentation/       # Comprehensive linguistic analysis, benchmarks, and guides
│   │   ├── KASHMIRI_LINGUISTIC_ANALYSIS.md
│   │   ├── MODEL_BENCHMARKS.md
│   │   ├── PIPELINE_GUIDE.md
│   │   └── RESEARCH_REPORT.md
│   ├── paper/               # Dataset specification and reference bibliography
│   ├── quality_reports/     # Quality audit reports and statistical summaries
│   ├── research_log/        # Experiment logs and throughput benchmarks
│   ├── scripts/             # Data processing, normalization, and training scripts
│   │   ├── audit_dataset.py
│   │   ├── build_manifest.py
│   │   ├── create_splits.py
│   │   ├── normalize_transcripts.py
│   │   ├── quality_filter.py
│   │   ├── run_full_pipeline.py
│   │   ├── train_b01_xlsr.py
│   │   └── validate_audio.py
│   └── tokenizers/          # Trained SentencePiece tokenizers and configs
├── .gitignore
└── README.md
```

---

## 🚀 Quickstart

### Kashmiri Text Normalization
```python
from kashmiri_dataset.scripts.normalize_transcripts import normalize_kashmiri_text, detect_code_switching

raw_text = "مِہ چھ ناو توسیف جیلانی"
normalized_text = normalize_kashmiri_text(raw_text)
cs_metrics = detect_code_switching(raw_text)

print(normalized_text)
print(cs_metrics)
```

### Running the End-to-End Pipeline
```bash
python3 kashmiri_dataset/scripts/run_full_pipeline.py
```

---

## 📊 Evaluation Suite & Empirical Benchmarks

The repository includes a standardized 6-split evaluation benchmark (`kashmiri_dataset/experiments/asr_baselines/eval_splits/`) designed to evaluate Kashmiri ASR across distinct acoustic regimes:

| Evaluation Split | Focus Condition | Samples | Master Status |
| :--- | :--- | :---: | :--- |
| **`eval_clean_read`** | Studio-grade near-field read speech | 100 | OpenSLR SLR122 & KTTS |
| **`eval_conversational_field`** | Rural/urban spontaneous native dialogue | 100 | IndicVoices Kashmiri |
| **`eval_multispeaker_bolbosh`** | Multi-speaker gender-balanced speech | 100 | GAASH-Lab Bolbosh |
| **`eval_unseen_speakers`** | Strictly disjoint held-out speakers | 100 | Held-out speaker isolation |
| **`eval_code_switched`** | English/Urdu technical and loan words | 50 | Code-switching benchmark |
| **`eval_noisy_low_snr`** | Environmental noise (SNR < 18 dB) | 50 | Acoustic stress test |
| **`master_benchmark_eval`** | Representative global cross-section | 297 | Combined comparative ranking |

For full empirical metrics across IndicConformer, Whisper, and native XLS-R models, see [MODEL_BENCHMARKS.md](file:///home/musa/Kashmiri%20Project/kashmiri_dataset/documentation/MODEL_BENCHMARKS.md).

---

## 📄 License & Attribution
Developed by [Musavir Khaliq](https://github.com/Musavirkhaliq).
