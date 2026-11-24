# WAAZ Kashmiri Crowdsourced Community Speech Audit

## 1. Platform Infrastructure & Governance

- **Platform Directory**: `/home/musa/Desktop/Projects/DataCollection`
- **Database**: SQLite3 (`data/datacollect.db`, 966 MB)
- **Contributor Privacy**: Strict anonymization enforced. Speaker names, contact info, and IP addresses are excluded from research manifests.
- **Consent Terms**: Contributors agree to the EduCasheer Contributor License allowing speech recognition research and public non-commercial model release.

## 2. Quantitative Platform Metrics

| Metric | Verified Measurement | Notes |
| :--- | :--- | :--- |
| **Registered Users** | 555 | Contributor accounts in database |
| **Active Audio Contributors** | 61 | Native speakers who submitted valid recordings |
| **Available Prompts** | 1,456,259 | Curated Kashmiri text prompts across domains |
| **Total Submitted Audio** | 29,105 clips | 54.02 hours total physical audio |
| **Approved & Verified (GOLD)** | 346 clips | 0.95 hours integrated into `manifest_gold.jsonl` |
| **Pending Review Queue (BRONZE)** | 1,994 clips | 4.75 hours queued in `HUMAN_VALIDATION_QUEUE.jsonl` |
| **Rejected Audio** | 51 clips | Filtered out due to background noise or prompt mismatch |

## 3. Incremental Export & Integration Workflow

New community recordings are processed via the automated incremental ingestion query in `data_acquisition/processing/build_master_corpus.py`.
