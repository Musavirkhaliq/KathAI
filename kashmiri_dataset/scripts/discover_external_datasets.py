#!/usr/bin/env python3
"""
External Kashmiri Dataset Cataloger & Registry
===============================================
Generates documentation/EXTERNAL_DATASETS.md with structured metadata for all
discovered public resources according to the 15 required criteria:
  source, URL, dataset name, data type, language/script, samples, hours/tokens,
  license, commercial permission, redistribution permission, speaker info,
  quality, availability, access method, recommended use.
"""

import os
import json

DOCS_DIR = "/home/musa/Kashmiri Project/kashmiri_dataset/documentation"
os.makedirs(DOCS_DIR, exist_ok=True)

CATALOG = [
    {
        "dataset_name": "Bolbosh (GAASH Lab)",
        "source": "GAASH Lab, NIT Srinagar & University of Kashmir",
        "url": "https://huggingface.co/datasets/GAASH-Lab/Bolbosh",
        "data_type": "Multi-speaker Speech + Transcript Alignment",
        "language_script": "Kashmiri (Perso-Arabic script)",
        "samples": "25 Parquet Shards (~10k+ paired utterances)",
        "hours_tokens": "11.82 GB high-fidelity audio (~25+ hours)",
        "license": "CC-BY-4.0",
        "commercial_use": "Permitted (with attribution)",
        "redistribution": "Permitted (with attribution)",
        "speaker_info": "Multiple native male and female Kashmiri speakers",
        "quality": "Studio-grade, dereverberated, normalized",
        "availability": "Open / Ungated",
        "how_to_download": "huggingface_hub snapshot_download('GAASH-Lab/Bolbosh')",
        "recommended_use": "Kashmiri ASR clean evaluation benchmark & neural TTS acoustic modeling"
    },
    {
        "dataset_name": "IndicVoices Kashmiri Split",
        "source": "AI4Bharat (IIT Madras)",
        "url": "https://huggingface.co/datasets/humair025/KashmiriSpeech-IndicVoices",
        "data_type": "Conversational Speech + Perso-Arabic Transcripts",
        "language_script": "Kashmiri (kas / Perso-Arabic)",
        "samples": "41 Parquet Shards (~100k+ utterances)",
        "hours_tokens": "16.94 GB audio (~45-65 hours)",
        "license": "CC-BY-4.0 (AI4Bharat open release)",
        "commercial_use": "Permitted (with attribution)",
        "redistribution": "Permitted (with attribution)",
        "speaker_info": "450+ native speakers from Jammu & Kashmir",
        "quality": "High natural diversity, field recordings across districts",
        "availability": "Open / Ungated",
        "how_to_download": "huggingface_hub snapshot_download('humair025/KashmiriSpeech-IndicVoices')",
        "recommended_use": "Foundation ASR acoustic pretraining and dialect-robust fine-tuning"
    },
    {
        "dataset_name": "OpenSLR SLR122 Kashmiri Data Corpus",
        "source": "OpenSLR",
        "url": "https://openslr.org/122/",
        "data_type": "Conversational Speech + Timestamped Transcriptions",
        "language_script": "Kashmiri (Perso-Arabic)",
        "samples": "1,983 sentence segments across 43 long sessions",
        "hours_tokens": "3.38 total hours (2.54 hours active speech)",
        "license": "OpenSLR Open License",
        "commercial_use": "Verify specific OpenSLR card; typically permitted with attribution",
        "redistribution": "Permitted",
        "speaker_info": "Native university students and residents from Srinagar, Shopian, Pulwama",
        "quality": "Natural conversational speech, colloquial pacing, varied background noise",
        "availability": "Open / Publicly Downloaded",
        "how_to_download": "Direct tar.gz download from openslr.org/122",
        "recommended_use": "Conversational ASR training and code-switching evaluation"
    },
    {
        "dataset_name": "KTTS Single Speaker Dataset",
        "source": "Mendeley Data",
        "url": "https://data.mendeley.com/datasets/5c4dcvxdmb/1",
        "data_type": "Read Speech Audio + CSV Transcripts",
        "language_script": "Kashmiri (Perso-Arabic)",
        "samples": "2,984 audio recordings + 2,984 transcript sentences",
        "hours_tokens": "~3.5 hours",
        "license": "CC BY-NC 3.0",
        "commercial_use": "NO (Non-Commercial condition applies)",
        "redistribution": "Permitted for research/non-commercial with attribution",
        "speaker_info": "Single adult male native Kashmiri speaker (Srinagar standard)",
        "quality": "Clean studio recording, consistent microphone distance, high SNR",
        "availability": "Open / Publicly Downloaded",
        "how_to_download": "Mendeley Data direct download",
        "recommended_use": "Single-speaker TTS anchor and clean acoustic baseline"
    },
    {
        "dataset_name": "Kashmiri Audio Corpus (programindz)",
        "source": "Hugging Face — programindz",
        "url": "https://huggingface.co/datasets/programindz/kashmiri-audio-corpus",
        "data_type": "Segmented Speech + Perso-Arabic Text",
        "language_script": "Kashmiri (kas)",
        "samples": "1,955 records",
        "hours_tokens": "2.53 hours (9,107 seconds)",
        "license": "GPL-3.0",
        "commercial_use": "Permitted under copyleft GPL-3.0 requirements",
        "redistribution": "Permitted under GPL-3.0",
        "speaker_info": "Native Kashmiri speakers",
        "quality": "Standard 16kHz mono audio",
        "availability": "Open / Publicly Downloaded",
        "how_to_download": "Hugging Face dataset download",
        "recommended_use": "ASR acoustic model fine-tuning"
    },
    {
        "dataset_name": "XKaab ASR Kashmiri 4hrs",
        "source": "Hugging Face — XKaab",
        "url": "https://huggingface.co/datasets/XKaab/ASR-Kashmiri_4hrs",
        "data_type": "Speech Audio + Transcriptions",
        "language_script": "Kashmiri (kas)",
        "samples": "Parquet archive (1,500+ clips)",
        "hours_tokens": "226 MB (~4.0 hours)",
        "license": "Open Research",
        "commercial_use": "Unspecified; research recommended",
        "redistribution": "Permitted",
        "speaker_info": "Native Kashmiri speakers",
        "quality": "Clean 16kHz segmented speech",
        "availability": "Open / Ungated (Downloaded)",
        "how_to_download": "huggingface_hub snapshot_download('XKaab/ASR-Kashmiri_4hrs')",
        "recommended_use": "ASR validation and evaluation benchmark"
    },
    {
        "dataset_name": "Aya Kashmiri Instruction Dataset",
        "source": "Cohere For AI / humairmunirawn",
        "url": "https://huggingface.co/datasets/humairmunirawn/aya-kashmiri-only",
        "data_type": "Instruction-Tuning Q&A / Conversations",
        "language_script": "Kashmiri (kas / Perso-Arabic)",
        "samples": "100k+ instruction-response pairs",
        "hours_tokens": "1.29 GB text data",
        "license": "Apache-2.0 / CC-BY-4.0",
        "commercial_use": "Permitted",
        "redistribution": "Permitted",
        "speaker_info": "N/A (Text)",
        "quality": "High-quality human-curated and translated instruction pairs",
        "availability": "Open / Ungated (Downloaded)",
        "how_to_download": "huggingface_hub snapshot_download('humairmunirawn/aya-kashmiri-only')",
        "recommended_use": "Instruction fine-tuning for Kashmiri LLM/SLM"
    },
    {
        "dataset_name": "Sangraha (Kashmiri Verified)",
        "source": "AI4Bharat (IIT Madras)",
        "url": "https://huggingface.co/datasets/ai4bharat/sangraha",
        "data_type": "Pretraining Text Corpus",
        "language_script": "Kashmiri (kas / Perso-Arabic)",
        "samples": "Curated clean document partitions",
        "hours_tokens": "~1.5 MB clean Kashmiri web & literary text",
        "license": "CC-BY-4.0",
        "commercial_use": "Permitted",
        "redistribution": "Permitted",
        "speaker_info": "N/A (Text)",
        "quality": "Human and algorithmically verified",
        "availability": "Open / Ungated (Downloaded)",
        "how_to_download": "huggingface_hub snapshot_download('ai4bharat/sangraha', allow_patterns=['verified/kas/*'])",
        "recommended_use": "LLM pretraining and vocabulary training"
    },
    {
        "dataset_name": "Kashmiri Wikipedia Articles Dump",
        "source": "Wikimedia Foundation",
        "url": "https://dumps.wikimedia.org/kswiki/latest/",
        "data_type": "Encyclopedic Text Corpus",
        "language_script": "Kashmiri (Perso-Arabic)",
        "samples": "Thousands of structured encyclopedic articles",
        "hours_tokens": "10.7 MB compressed (multi-million tokens)",
        "license": "CC-BY-SA 3.0 / GFDL",
        "commercial_use": "Permitted under ShareAlike terms",
        "redistribution": "Permitted",
        "speaker_info": "N/A (Text)",
        "quality": "Formal Kashmiri, wide domain coverage (history, geography, literature)",
        "availability": "Open / Publicly Downloaded",
        "how_to_download": "Direct download from dumps.wikimedia.org with valid user-agent",
        "recommended_use": "Domain pretraining and language modeling"
    },
    {
        "dataset_name": "Kashmiri-English Dataset 270K",
        "source": "SMUQamar (Hugging Face)",
        "url": "https://huggingface.co/datasets/SMUQamar/Kashmiri-English-Dataset-270K",
        "data_type": "Parallel Translation Corpus",
        "language_script": "Kashmiri (Perso-Arabic) ↔ English (Latin)",
        "samples": "270,000 parallel sentence pairs",
        "hours_tokens": "~40 MB",
        "license": "CC-BY-NC-SA 4.0",
        "commercial_use": "NO (Non-Commercial condition)",
        "redistribution": "Permitted under NC-SA terms",
        "speaker_info": "N/A (Text)",
        "quality": "Extensive coverage, cleaned alignments",
        "availability": "Gated (terms acceptance on Hugging Face)",
        "how_to_download": "Hugging Face token login required",
        "recommended_use": "Machine translation (NMT) and cross-lingual Kashmiri-English LLM alignment"
    },
    {
        "dataset_name": "LDC-IL Kashmiri Raw Speech Corpus",
        "source": "Central Institute of Indian Languages (CIIL / LDC-IL)",
        "url": "https://data.ldcil.org/speech-raw-corpus",
        "data_type": "Raw Continuous & Segmented Speech",
        "language_script": "Kashmiri",
        "samples": "16,380 audio segments",
        "hours_tokens": "28.17 hours (18 GB)",
        "license": "LDC-IL Restrictive License",
        "commercial_use": "Requires formal licensing agreement",
        "redistribution": "Restricted (redistribution strictly prohibited)",
        "speaker_info": "150 speakers (78 female, 72 male) across Pulwama, Srinagar, Anantnag",
        "quality": "Government standardized linguistic corpus",
        "availability": "Restricted (Formal application to LDC-IL required)",
        "how_to_download": "Submit formal requisition to LDC-IL portal",
        "recommended_use": "Secondary validation if formal project access is granted"
    }
]

def generate_markdown():
    lines = []
    lines.append("# Kashmiri Dataset Registry & External Resource Catalog\n")
    lines.append("Comprehensive catalog of all identified public, academic, and government resources for Kashmiri language AI.\n")
    lines.append("## Licensing Taxonomy\n")
    lines.append("- **Open & Commercially Reusable**: CC-BY-4.0, Apache-2.0, MIT, CC-BY-SA 3.0 (Bolbosh, IndicVoices, Sangraha, Wikipedia, NawabHussain).")
    lines.append("- **Non-Commercial / Research-Only**: CC-BY-NC 3.0/4.0 (KTTS, SMUQamar 270K).")
    lines.append("- **Copyleft**: GPL-3.0 (Programindz Audio Corpus).")
    lines.append("- **Gated**: Hugging Face gated access (Koshur Pixel, KS-PRET-5M, SMUQamar 270K).")
    lines.append("- **Restricted**: LDC-IL (formal institutional clearance required; no redistribution).\n")
    lines.append("## Catalog of Discovered Datasets\n")

    for idx, d in enumerate(CATALOG, 1):
        lines.append(f"### {idx}. {d['dataset_name']}\n")
        lines.append(f"- **Source**: {d['source']}")
        lines.append(f"- **URL**: [{d['url']}]({d['url']})")
        lines.append(f"- **Data Type**: {d['data_type']}")
        lines.append(f"- **Language & Script**: {d['language_script']}")
        lines.append(f"- **Sample Count**: {d['samples']}")
        lines.append(f"- **Hours / Tokens / Size**: {d['hours_tokens']}")
        lines.append(f"- **License**: `{d['license']}`")
        lines.append(f"- **Commercial Use**: {d['commercial_use']}")
        lines.append(f"- **Redistribution Permission**: {d['redistribution']}")
        lines.append(f"- **Speaker Information**: {d['speaker_info']}")
        lines.append(f"- **Quality**: {d['quality']}")
        lines.append(f"- **Availability Status**: {d['availability']}")
        lines.append(f"- **Access / Download Method**: `{d['how_to_download']}`")
        lines.append(f"- **Recommended Role in Research**: {d['recommended_use']}\n")

    out_file = os.path.join(DOCS_DIR, "EXTERNAL_DATASETS.md")
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"External datasets catalog written to: {out_file}")

if __name__ == '__main__':
    generate_markdown()
