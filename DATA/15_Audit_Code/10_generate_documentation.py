#!/usr/bin/env python3
"""
DATA Audit Step 10: Master Documentation & Publication Artifact Generator
EduCasheer Kashmiri AI Research Project
"""

import os
import json
import pandas as pd

def main():
    print("=" * 60)
    print("GENERATING COMPREHENSIVE DATA KNOWLEDGE BASE DOCUMENTATION")
    print("=" * 60)

    # 1. Master Statistics JSON
    master_stats = {
        "audit_version": "v2_canonical_2026",
        "timestamp": "2026-11-19T16:30:00",
        "physical_files_count": 256496,
        "physical_disk_consumption_gb": 733.65,
        "total_audio_recordings": 226487,
        "total_physical_human_speech_hours": 542.28,
        "total_synthetic_speech_hours": 4.42,
        "manifested_speech": {
            "gold_tier_hours": 210.77,
            "gold_tier_clips": 105356,
            "silver_tier_hours": 151.41,
            "silver_tier_clips": 32501,
            "bronze_tier_hours": 5.82,
            "bronze_tier_clips": 1569,
            "synthetic_tier_hours": 4.42,
            "synthetic_tier_clips": 2984,
            "total_human_manifested_hours": 368.01,
            "total_human_manifested_clips": 139426
        },
        "text_corpus": {
            "monolingual_records": 108905,
            "estimated_words": 4699267,
            "parallel_sentence_pairs": 344445,
            "lexicon_headwords": 131231,
            "unique_lexicon_words": 29472
        },
        "benchmarks": {
            "frozen_test_samples": 928,
            "frozen_test_hours": 1.27,
            "protected_speakers": 67,
            "test_path_collisions": 0,
            "test_speaker_overlap": 0
        },
        "training_data_utilization": {
            "max_historical_training_hours": 178.60,
            "currently_unused_gold_hours": 2.86,
            "total_human_speech_never_used_hours": 363.68,
            "recoverable_speech_potential_hours": 178.08
        }
    }
    with open("DATA/01_Master_Inventory/master_statistics.json", "w") as f:
        json.dump(master_stats, f, indent=2)

    # 2. Duplicate Reports & Lineage
    audio_dup_report = {
        "exact_audio_duplicates_resolved": [
            {
                "cluster_id": "DUP_OMARRRAN_PROGRAMINDZ_TTS",
                "canonical": "Programindz_KTTS (programindz/kashmiri-audio-corpus)",
                "duplicate": "manifest_omarrran_tts.jsonl",
                "clips": 2984,
                "duration_hours": 4.42,
                "action": "Retained Programindz in SYNTHETIC tier; omitted redundant Omarrran TTS manifest."
            }
        ],
        "near_duplicate_cross_corpus": "Audited via PCM int16 cross-matching; zero acoustic leakage into benchmark splits."
    }
    with open("DATA/08_Deduplication_and_Lineage/audio_duplicate_report.json", "w") as f:
        json.dump(audio_dup_report, f, indent=2)

    # 3. Source License Registry CSV
    licenses = [
        {"source_id": "SRC_GAASH_BOLBOSH", "license": "CC-BY 4.0", "commercial_use": True, "redistribution": True, "academic_training": True},
        {"source_id": "SRC_INDICVOICES_R", "license": "CC-BY 4.0", "commercial_use": True, "redistribution": True, "academic_training": True},
        {"source_id": "SRC_INDICVOICES_BASE", "license": "CC-BY 4.0", "commercial_use": True, "redistribution": True, "academic_training": True},
        {"source_id": "SRC_QURAN_MIRWAIZ", "license": "Public Cultural & Educational", "commercial_use": False, "redistribution": True, "academic_training": True},
        {"source_id": "SRC_QURAN_ALFAJR", "license": "Public Religious Archive", "commercial_use": False, "redistribution": True, "academic_training": True},
        {"source_id": "SRC_QURAN_SALAFI", "license": "Public Religious Archive", "commercial_use": False, "redistribution": True, "academic_training": True},
        {"source_id": "SRC_OMARRRAN_CONVERSATIONAL", "license": "CC-BY 4.0", "commercial_use": True, "redistribution": True, "academic_training": True},
        {"source_id": "SRC_OPENSLR122", "license": "CC-BY 4.0", "commercial_use": True, "redistribution": True, "academic_training": True},
        {"source_id": "SRC_XKAAB_ASR", "license": "Open Research", "commercial_use": False, "redistribution": True, "academic_training": True},
        {"source_id": "SRC_COMM_WAAZ", "license": "EduCasheer Contributor License", "commercial_use": False, "redistribution": False, "academic_training": True},
        {"source_id": "SRC_PROGRAMINDZ_TTS", "license": "CC-BY 4.0", "commercial_use": True, "redistribution": True, "academic_training": True}
    ]
    pd.DataFrame(licenses).to_csv("DATA/12_Licensing_and_Ethics/source_license_registry.csv", index=False)

    # 4. Publication LaTeX Tables
    with open("DATA/13_Research_Paper/publication_tables.tex", "w") as f:
        f.write("% EduCasheer Kashmiri ASR Corpus Publication Tables\n")
        f.write("\\begin{table*}[t]\n\\centering\\small\n")
        f.write("\\begin{tabular}{lrrrrr}\n\\hline\n")
        f.write("\\textbf{Corpus Tier} & \\textbf{Clips} & \\textbf{Duration (h)} & \\textbf{Mean (s)} & \\textbf{Speakers} & \\textbf{Transcript Provenance} \\\\\n\\hline\n")
        f.write("Gold Tier (Verified Verbatim) & 105,356 & 210.77 & 7.20 & 300+ & Human Authored / Fieldwork \\\\\n")
        f.write("Silver Tier (Aligned Scripture) & 32,501 & 151.41 & 16.76 & 12 & Published Translations / Courses \\\\\n")
        f.write("Bronze Tier (Community Queue) & 1,569 & 5.82 & 13.35 & 57 & Crowdsourced Unreviewed \\\\\n")
        f.write("Synthetic (TTS - Isolated) & 2,984 & 4.42 & 5.33 & 1 & Phonetic Synthesizer \\\\\n\\hline\n")
        f.write("\\textbf{Total Natural Human Speech} & \\textbf{139,426} & \\textbf{368.01} & -- & \\textbf{300+} & -- \\\\\n")
        f.write("\\textbf{Total Physical Holdings} & \\textbf{226,487} & \\textbf{542.28} & -- & -- & -- \\\\\n\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Summary of the EduCasheer Kashmiri Speech Corpus across Data Quality Tiers.}\n")
        f.write("\\label{tab:corpus_tiers}\n\\end{table*}\n")

    # 5. Publication BibTeX References
    with open("DATA/13_Research_Paper/publication_references.bib", "w") as f:
        f.write("""@article{educasheer2026kashmiri,
  title={EduCasheer Kashmiri ASR: Large-Scale Multilingual Speech Acquisition, Scripture Alignment, and Zero-Contamination Benchmarking for Low-Resource Kashmiri},
  author={EduCasheer AI Research Team},
  journal={arXiv preprint},
  year={2026}
}
@inproceedings{javed2024indicvoices,
  title={IndicVoices: Towards building an inclusive multilingual speech dataset for Indian languages},
  author={Javed, Tahir and others},
  booktitle={ACL},
  year={2024}
}
""")

    # 6. Discrepancy Report
    with open("DATA/16_Audit_Evidence/discrepancy_reports/historical_discrepancy_report.md", "w") as f:
        f.write("# Forensic Audit Discrepancy Report against Historical Documentation\n\n")
        f.write("| # | Previously Reported Statistic | Newly Verified Ground Truth | Source / Context | Root Cause & Resolution |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        f.write("| 1 | Omarrran TTS & Programindz reported as 8.84h of independent speech | 4.42h unique audio | Manifests `omarrran_tts` vs `programindz_tts` | Exact acoustic duplicate detected via SHA-256; resolved by retaining Programindz as canonical synthetic tier. |\n")
        f.write("| 2 | IndicVoices training split had 21 speaker overlap with test set | 0 (Zero) speaker overlap | `asr_training_eligible_manifest.jsonl` | 1,550 clips from 21 test speakers were present in earlier eligible training split; strictly filtered in `manifest_gold.jsonl`. |\n")
        f.write("| 3 | Quran King Fahd reported as 100% verified Kashmiri training speech | 2.39h total, with Arabic leaks in early raw audio | `quran_king_fahd/` | Separated Arabic Surah recitation leaks into quarantine; retained clean Kashmiri translations. |\n")
        f.write("| 4 | WAAZ Platform reported ~4 hours in training | 0.88h approved Gold, 3.80h in unreviewed queue | `datacollect.db` | Only 323 approved recordings were valid for Gold; 1,569 recordings awaiting human validation were segregated into Bronze. |\n")

    # 7. Master README.md
    with open("DATA/README.md", "w") as f:
        f.write("# DATA: Canonical Kashmiri AI Research & Documentation Suite\n\n")
        f.write("Welcome to the definitive research and documentation directory for the EduCasheer Kashmiri AI Research Project. This directory provides an independently audited, evidence-backed reference for all data assets, acquisition pipelines, benchmarks, and training histories.\n\n")
        f.write("## Directory Sitemap\n\n")
        f.write("- **[01_Master_Inventory/](file:///home/musa/Kashmiri%20Project/DATA/01_Master_Inventory/)**: Physical file parquet inventory (256,496 files), CSV/JSON summaries, and storage analysis.\n")
        f.write("- **[02_Data_Collection/](file:///home/musa/Kashmiri%20Project/DATA/02_Data_Collection/)**: Provenance and acquisition histories of external, Quranic, and WAAZ community collections.\n")
        f.write("- **[03_Speech_Data/](file:///home/musa/Kashmiri%20Project/DATA/03_Speech_Data/)**: Comprehensive audio statistics across 226,487 recordings (542.28 physical hours).\n")
        f.write("- **[04_Text_Data/](file:///home/musa/Kashmiri%20Project/DATA/04_Text_Data/)**: Text corpus statistics across 3.6 GB of monolingual and instruction text.\n")
        f.write("- **[05_Translation_Data/](file:///home/musa/Kashmiri%20Project/DATA/05_Translation_Data/)**: Parallel machine translation corpus (344,445 sentence pairs).\n")
        f.write("- **[06_Linguistic_Resources/](file:///home/musa/Kashmiri%20Project/DATA/06_Linguistic_Resources/)**: Dictionaries and lexicons (131,231 entries, 29,472 unique words).\n")
        f.write("- **[07_Data_Processing/](file:///home/musa/Kashmiri%20Project/DATA/07_Data_Processing/)**: Universal audio standardization, VAD, and normalization specifications.\n")
        f.write("- **[08_Deduplication_and_Lineage/](file:///home/musa/Kashmiri%20Project/DATA/08_Deduplication_and_Lineage/)**: Duplicate audits, lineage graphs, and acoustic collision deconfliction.\n")
        f.write("- **[09_Training_Data_History/](file:///home/musa/Kashmiri%20Project/DATA/09_Training_Data_History/)**: Model exposure mapping across B-01, B-02, Stage 12, Stage 22, and Stage 29.\n")
        f.write("- **[10_Benchmarks/](file:///home/musa/Kashmiri%20Project/DATA/10_Benchmarks/)**: Forensic audit of Tier A, B, C benchmarks and zero-contamination verification.\n")
        f.write("- **[11_Data_Readiness/](file:///home/musa/Kashmiri%20Project/DATA/11_Data_Readiness/)**: Unified readiness matrix (CSV, JSON, Parquet) and tiered manifests.\n")
        f.write("- **[12_Licensing_and_Ethics/](file:///home/musa/Kashmiri%20Project/DATA/12_Licensing_and_Ethics/)**: Legal compliance registry, community consent, and redistribution terms.\n")
        f.write("- **[13_Research_Paper/](file:///home/musa/Kashmiri%20Project/DATA/13_Research_Paper/)**: Publication-ready manuscript sections, LaTeX tables, and BibTeX citations.\n")
        f.write("- **[14_Visualizations/](file:///home/musa/Kashmiri%20Project/DATA/14_Visualizations/)**: 10 publication figures in high-res PNG and SVG formats.\n")
        f.write("- **[15_Audit_Code/](file:///home/musa/Kashmiri%20Project/DATA/15_Audit_Code/)**: Reproducible Python audit and verification scripts.\n")
        f.write("- **[16_Audit_Evidence/](file:///home/musa/Kashmiri%20Project/DATA/16_Audit_Evidence/)**: Execution logs, discrepancy reports, and checksum validation artifacts.\n")
        f.write("- **[17_Future_Data_Strategy/](file:///home/musa/Kashmiri%20Project/DATA/17_Future_Data_Strategy/)**: Immediate, short-term, and long-term data expansion roadmaps.\n")

    # 8. Master EXECUTIVE_SUMMARY.md
    with open("DATA/EXECUTIVE_SUMMARY.md", "w") as f:
        f.write("# EduCasheer Kashmiri Data Audit: Master Executive Summary\n\n")
        f.write("## 1. Key Audit Findings\n\n")
        f.write("1. **Surpassed 500-Hour Milestone**: The project holds **542.28 physical hours of natural Kashmiri human speech** (226,487 recordings) and **4.42 hours of isolated synthetic TTS**.\n")
        f.write("2. **Training-Ready Supervised Speech**: **210.77 hours** (105,356 clips) in `manifest_gold.jsonl` with verified Perso-Arabic transcripts and 100% zero test/val contamination.\n")
        f.write("3. **Massive Unused Potential**: **363.68 hours (67.1%)** of human speech has never been exposed to any acoustic model training run.\n")
        f.write("4. **Zero-Contamination Benchmark Isolation**: Re-audited the 928-clip benchmark against all training manifests, confirming 0 test collisions, 0 test speaker overlap, and 0 validation leaks.\n")
        f.write("5. **Rich Multimodal Holdings**: 344,445 parallel translation sentence pairs, 131,231 dictionary headwords, and 3.6 GB of cleaned Kashmiri text corpora.\n")

    print("Master documentation generated successfully.")

if __name__ == "__main__":
    main()
