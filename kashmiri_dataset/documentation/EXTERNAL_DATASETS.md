# Kashmiri Dataset Registry & External Resource Catalog

Comprehensive catalog of all identified public, academic, and government resources for Kashmiri language AI.

## Licensing Taxonomy

- **Open & Commercially Reusable**: CC-BY-4.0, Apache-2.0, MIT, CC-BY-SA 3.0 (Bolbosh, IndicVoices, Sangraha, Wikipedia, NawabHussain).
- **Non-Commercial / Research-Only**: CC-BY-NC 3.0/4.0 (KTTS, SMUQamar 270K).
- **Copyleft**: GPL-3.0 (Programindz Audio Corpus).
- **Gated**: Hugging Face gated access (Koshur Pixel, KS-PRET-5M, SMUQamar 270K).
- **Restricted**: LDC-IL (formal institutional clearance required; no redistribution).

## Catalog of Discovered Datasets

### 1. Bolbosh (GAASH Lab)

- **Source**: GAASH Lab, NIT Srinagar & University of Kashmir
- **URL**: [https://huggingface.co/datasets/GAASH-Lab/Bolbosh](https://huggingface.co/datasets/GAASH-Lab/Bolbosh)
- **Data Type**: Multi-speaker Speech + Transcript Alignment
- **Language & Script**: Kashmiri (Perso-Arabic script)
- **Sample Count**: 25 Parquet Shards (~10k+ paired utterances)
- **Hours / Tokens / Size**: 11.82 GB high-fidelity audio (~25+ hours)
- **License**: `CC-BY-4.0`
- **Commercial Use**: Permitted (with attribution)
- **Redistribution Permission**: Permitted (with attribution)
- **Speaker Information**: Multiple native male and female Kashmiri speakers
- **Quality**: Studio-grade, dereverberated, normalized
- **Availability Status**: Open / Ungated
- **Access / Download Method**: `huggingface_hub snapshot_download('GAASH-Lab/Bolbosh')`
- **Recommended Role in Research**: Kashmiri ASR clean evaluation benchmark & neural TTS acoustic modeling

### 2. IndicVoices Kashmiri Split

- **Source**: AI4Bharat (IIT Madras)
- **URL**: [https://huggingface.co/datasets/humair025/KashmiriSpeech-IndicVoices](https://huggingface.co/datasets/humair025/KashmiriSpeech-IndicVoices)
- **Data Type**: Conversational Speech + Perso-Arabic Transcripts
- **Language & Script**: Kashmiri (kas / Perso-Arabic)
- **Sample Count**: 41 Parquet Shards (~100k+ utterances)
- **Hours / Tokens / Size**: 16.94 GB audio (~45-65 hours)
- **License**: `CC-BY-4.0 (AI4Bharat open release)`
- **Commercial Use**: Permitted (with attribution)
- **Redistribution Permission**: Permitted (with attribution)
- **Speaker Information**: 450+ native speakers from Jammu & Kashmir
- **Quality**: High natural diversity, field recordings across districts
- **Availability Status**: Open / Ungated
- **Access / Download Method**: `huggingface_hub snapshot_download('humair025/KashmiriSpeech-IndicVoices')`
- **Recommended Role in Research**: Foundation ASR acoustic pretraining and dialect-robust fine-tuning

### 3. OpenSLR SLR122 Kashmiri Data Corpus

- **Source**: OpenSLR
- **URL**: [https://openslr.org/122/](https://openslr.org/122/)
- **Data Type**: Conversational Speech + Timestamped Transcriptions
- **Language & Script**: Kashmiri (Perso-Arabic)
- **Sample Count**: 1,983 sentence segments across 43 long sessions
- **Hours / Tokens / Size**: 3.38 total hours (2.54 hours active speech)
- **License**: `OpenSLR Open License`
- **Commercial Use**: Verify specific OpenSLR card; typically permitted with attribution
- **Redistribution Permission**: Permitted
- **Speaker Information**: Native university students and residents from Srinagar, Shopian, Pulwama
- **Quality**: Natural conversational speech, colloquial pacing, varied background noise
- **Availability Status**: Open / Publicly Downloaded
- **Access / Download Method**: `Direct tar.gz download from openslr.org/122`
- **Recommended Role in Research**: Conversational ASR training and code-switching evaluation

### 4. KTTS Single Speaker Dataset

- **Source**: Mendeley Data
- **URL**: [https://data.mendeley.com/datasets/5c4dcvxdmb/1](https://data.mendeley.com/datasets/5c4dcvxdmb/1)
- **Data Type**: Read Speech Audio + CSV Transcripts
- **Language & Script**: Kashmiri (Perso-Arabic)
- **Sample Count**: 2,984 audio recordings + 2,984 transcript sentences
- **Hours / Tokens / Size**: ~3.5 hours
- **License**: `CC BY-NC 3.0`
- **Commercial Use**: NO (Non-Commercial condition applies)
- **Redistribution Permission**: Permitted for research/non-commercial with attribution
- **Speaker Information**: Single adult male native Kashmiri speaker (Srinagar standard)
- **Quality**: Clean studio recording, consistent microphone distance, high SNR
- **Availability Status**: Open / Publicly Downloaded
- **Access / Download Method**: `Mendeley Data direct download`
- **Recommended Role in Research**: Single-speaker TTS anchor and clean acoustic baseline

### 5. Kashmiri Audio Corpus (programindz)

- **Source**: Hugging Face — programindz
- **URL**: [https://huggingface.co/datasets/programindz/kashmiri-audio-corpus](https://huggingface.co/datasets/programindz/kashmiri-audio-corpus)
- **Data Type**: Segmented Speech + Perso-Arabic Text
- **Language & Script**: Kashmiri (kas)
- **Sample Count**: 1,955 records
- **Hours / Tokens / Size**: 2.53 hours (9,107 seconds)
- **License**: `GPL-3.0`
- **Commercial Use**: Permitted under copyleft GPL-3.0 requirements
- **Redistribution Permission**: Permitted under GPL-3.0
- **Speaker Information**: Native Kashmiri speakers
- **Quality**: Standard 16kHz mono audio
- **Availability Status**: Open / Publicly Downloaded
- **Access / Download Method**: `Hugging Face dataset download`
- **Recommended Role in Research**: ASR acoustic model fine-tuning

### 6. XKaab ASR Kashmiri 4hrs

- **Source**: Hugging Face — XKaab
- **URL**: [https://huggingface.co/datasets/XKaab/ASR-Kashmiri_4hrs](https://huggingface.co/datasets/XKaab/ASR-Kashmiri_4hrs)
- **Data Type**: Speech Audio + Transcriptions
- **Language & Script**: Kashmiri (kas)
- **Sample Count**: Parquet archive (1,500+ clips)
- **Hours / Tokens / Size**: 226 MB (~4.0 hours)
- **License**: `Open Research`
- **Commercial Use**: Unspecified; research recommended
- **Redistribution Permission**: Permitted
- **Speaker Information**: Native Kashmiri speakers
- **Quality**: Clean 16kHz segmented speech
- **Availability Status**: Open / Ungated (Downloaded)
- **Access / Download Method**: `huggingface_hub snapshot_download('XKaab/ASR-Kashmiri_4hrs')`
- **Recommended Role in Research**: ASR validation and evaluation benchmark

### 7. Aya Kashmiri Instruction Dataset

- **Source**: Cohere For AI / humairmunirawn
- **URL**: [https://huggingface.co/datasets/humairmunirawn/aya-kashmiri-only](https://huggingface.co/datasets/humairmunirawn/aya-kashmiri-only)
- **Data Type**: Instruction-Tuning Q&A / Conversations
- **Language & Script**: Kashmiri (kas / Perso-Arabic)
- **Sample Count**: 100k+ instruction-response pairs
- **Hours / Tokens / Size**: 1.29 GB text data
- **License**: `Apache-2.0 / CC-BY-4.0`
- **Commercial Use**: Permitted
- **Redistribution Permission**: Permitted
- **Speaker Information**: N/A (Text)
- **Quality**: High-quality human-curated and translated instruction pairs
- **Availability Status**: Open / Ungated (Downloaded)
- **Access / Download Method**: `huggingface_hub snapshot_download('humairmunirawn/aya-kashmiri-only')`
- **Recommended Role in Research**: Instruction fine-tuning for Kashmiri LLM/SLM

### 8. Sangraha (Kashmiri Verified)

- **Source**: AI4Bharat (IIT Madras)
- **URL**: [https://huggingface.co/datasets/ai4bharat/sangraha](https://huggingface.co/datasets/ai4bharat/sangraha)
- **Data Type**: Pretraining Text Corpus
- **Language & Script**: Kashmiri (kas / Perso-Arabic)
- **Sample Count**: Curated clean document partitions
- **Hours / Tokens / Size**: ~1.5 MB clean Kashmiri web & literary text
- **License**: `CC-BY-4.0`
- **Commercial Use**: Permitted
- **Redistribution Permission**: Permitted
- **Speaker Information**: N/A (Text)
- **Quality**: Human and algorithmically verified
- **Availability Status**: Open / Ungated (Downloaded)
- **Access / Download Method**: `huggingface_hub snapshot_download('ai4bharat/sangraha', allow_patterns=['verified/kas/*'])`
- **Recommended Role in Research**: LLM pretraining and vocabulary training

### 9. Kashmiri Wikipedia Articles Dump

- **Source**: Wikimedia Foundation
- **URL**: [https://dumps.wikimedia.org/kswiki/latest/](https://dumps.wikimedia.org/kswiki/latest/)
- **Data Type**: Encyclopedic Text Corpus
- **Language & Script**: Kashmiri (Perso-Arabic)
- **Sample Count**: Thousands of structured encyclopedic articles
- **Hours / Tokens / Size**: 10.7 MB compressed (multi-million tokens)
- **License**: `CC-BY-SA 3.0 / GFDL`
- **Commercial Use**: Permitted under ShareAlike terms
- **Redistribution Permission**: Permitted
- **Speaker Information**: N/A (Text)
- **Quality**: Formal Kashmiri, wide domain coverage (history, geography, literature)
- **Availability Status**: Open / Publicly Downloaded
- **Access / Download Method**: `Direct download from dumps.wikimedia.org with valid user-agent`
- **Recommended Role in Research**: Domain pretraining and language modeling

### 10. Kashmiri-English Dataset 270K

- **Source**: SMUQamar (Hugging Face)
- **URL**: [https://huggingface.co/datasets/SMUQamar/Kashmiri-English-Dataset-270K](https://huggingface.co/datasets/SMUQamar/Kashmiri-English-Dataset-270K)
- **Data Type**: Parallel Translation Corpus
- **Language & Script**: Kashmiri (Perso-Arabic) ↔ English (Latin)
- **Sample Count**: 270,000 parallel sentence pairs
- **Hours / Tokens / Size**: ~40 MB
- **License**: `CC-BY-NC-SA 4.0`
- **Commercial Use**: NO (Non-Commercial condition)
- **Redistribution Permission**: Permitted under NC-SA terms
- **Speaker Information**: N/A (Text)
- **Quality**: Extensive coverage, cleaned alignments
- **Availability Status**: Gated (terms acceptance on Hugging Face)
- **Access / Download Method**: `Hugging Face token login required`
- **Recommended Role in Research**: Machine translation (NMT) and cross-lingual Kashmiri-English LLM alignment

### 11. LDC-IL Kashmiri Raw Speech Corpus

- **Source**: Central Institute of Indian Languages (CIIL / LDC-IL)
- **URL**: [https://data.ldcil.org/speech-raw-corpus](https://data.ldcil.org/speech-raw-corpus)
- **Data Type**: Raw Continuous & Segmented Speech
- **Language & Script**: Kashmiri
- **Sample Count**: 16,380 audio segments
- **Hours / Tokens / Size**: 28.17 hours (18 GB)
- **License**: `LDC-IL Restrictive License`
- **Commercial Use**: Requires formal licensing agreement
- **Redistribution Permission**: Restricted (redistribution strictly prohibited)
- **Speaker Information**: 150 speakers (78 female, 72 male) across Pulwama, Srinagar, Anantnag
- **Quality**: Government standardized linguistic corpus
- **Availability Status**: Restricted (Formal application to LDC-IL required)
- **Access / Download Method**: `Submit formal requisition to LDC-IL portal`
- **Recommended Role in Research**: Secondary validation if formal project access is granted
