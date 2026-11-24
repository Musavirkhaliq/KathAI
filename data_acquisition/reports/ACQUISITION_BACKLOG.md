# Kashmiri Speech Data Acquisition Backlog & Restricted Source Dossiers

## 1. Executive Overview

While the open and acquired data in our possession already totals **542.28 physical hours of Kashmiri human speech** (with 368.01 hours fully structured into Gold, Silver, and Bronze manifests), several high-value speech collections are governed by institutional undertakings, gated repository access, or commercial licensing.

This document compiles the complete acquisition dossiers, technical characteristics, legal requirements, and actionable next steps for these pending sources. Once unlocked, these sources will contribute an estimated **179.2+ additional hours** of diverse Kashmiri speech, directly advancing the long-term 1,000-hour corpus goal.

---

## 2. Priority Acquisition Dossiers

### 2.1 LDC-IL Kashmiri Raw Speech Corpus
- **Institutional Custodian**: Central Institute of Indian Languages (CIIL), Department of Higher Education, Ministry of Education, Government of India, Mysore.
- **Official Portal**: https://data.ldcil.org/kashmiri-raw-speech-corpus and https://datat.ldcil.org/kashmiri-raw-speech-corpus
- **Corpus Characteristics**:
  - **Total Duration**: 28.17 hours
  - **Recordings**: 16,380 distinct audio files
  - **Speakers**: 150 native Kashmiri speakers
  - **Acoustic Domains**: Literary readings, broadcast news, interactive commands, phonetically balanced sentences, names, and vocabulary lists.
- **Current Acquisition Status**:
  - Sample audio archive, prompt sheets, and full official documentation downloaded and archived in `data_acquisition/raw/multilingual/ldcil/`.
  - Official undertaking document (`Undertaking_Research.pdf`) acquired.
- **Required Action to Complete Acquisition**:
  1. Complete and sign the CIIL Research Undertaking form on behalf of the EduCasheer Kashmiri AI Research Project.
  2. Submit the signed undertaking along with institutional affiliation credentials to LDC-IL data administration (`data@ldcil.org` / `ldcil.ciil@gmail.com`).
  3. Upon verification, download the complete 28.17-hour dataset from the authenticated LDC-IL FTP/HTTP portal.

---

### 2.2 Chaashini Multilingual Telephony Speech Corpus (Kashmiri Configuration)
- **Author / Custodian**: Kapture CX Research (Hugging Face: `kapturecx/Chaashini`)
- **Repository URL**: https://huggingface.co/datasets/kapturecx/Chaashini
- **Corpus Characteristics**:
  - **Language Split**: `ks` (Kashmiri)
  - **Total Duration**: 11.04 hours
  - **Recordings**: 4,733 telephony-grade clips
  - **Storage Format**: 168 Parquet shards located under `data/ks/`
  - **Acoustic Environment**: Conversational and customer support telephony speech (8 kHz / 16 kHz narrowband and wideband).
- **Current Acquisition Status**:
  - Gated manual access on Hugging Face.
  - Access request formally submitted and currently pending author review.
- **Required Action to Complete Acquisition**:
  1. Monitor Hugging Face repository approval notification.
  2. Alternatively, reach out directly to the research maintainers at Kapture CX referencing the EduCasheer non-commercial academic research initiative.
  3. Once approved, execute automated shard download using `huggingface-cli download kapturecx/Chaashini --repo-type dataset --include "data/ks/*"`.

---

### 2.3 Josh Talks Kashmiri Conversational Audio
- **Custodian**: Josh Talks AI Research / Language Division
- **Corpus Characteristics**:
  - **Estimated Duration**: ~25.00 hours
  - **Speakers**: 50+ native speakers from Kupwara, Baramulla, and Srinagar districts
  - **Acoustic Environment**: Spontaneous conversational audio recorded with high-grade studio microphones.
- **Access Model**: Commercial / Partner Research Licensing
- **Required Action**:
  1. Initiate formal contact with the Josh Talks Data Partnerships team.
  2. Request non-commercial academic research licensing terms for low-resource Kashmiri speech recognition development.

---

### 2.4 Speechocean Multilingual Kashmiri Speech Datasets
- **Custodian**: Speechocean Inc. (Kingline Information Technology)
- **Corpus Characteristics**:
  - **Estimated Duration**: Up to 115.00 hours
  - **Format**: High-fidelity 16 kHz / 48 kHz studio-recorded prompt speech with human phonemic transcripts.
- **Access Model**: Commercial Procurement
- **Required Action**:
  1. Request an institutional quotation for the academic research tier of the Kashmiri speech corpus.

---

### 2.5 Faith Comes By Hearing (FCBH) & BibleBrain API
- **Custodian**: Faith Comes By Hearing / Digital Bible Platform
- **Corpus Characteristics**:
  - **Editions**: Kashmiri New Testament (`KASBSI` and `KASWCO`), Matthew to Revelation.
  - **Estimated Duration**: ~20.00 hours per edition (up to 40 hours across multiple narrators).
- **Access Requirements**:
  1. Obtain a developer API Key from BibleBrain (https://biblebrain.com).
  2. Extract verse-level audio and align with machine-readable Kashmiri scripture text.

---

## 3. Backlog Pipeline Summary

| Source Identifier | Source Name | Estimated Hours | Speakers | Access Pathway | Target Tier |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `SRC_LDC_IL_KASHMIRI` | LDC-IL Kashmiri Raw Speech Corpus | 28.17h | 150 | Institutional Undertaking (CIIL) | GOLD |
| `SRC_CHAASHINI_KASHMIRI` | Chaashini Telephony Speech (Kashmiri) | 11.04h | Multi | Hugging Face Gated Review | SILVER |
| `SRC_JOSH_TALKS_KASHMIRI` | Josh Talks Conversational Kashmiri | 25.00h | 50+ | Partner Research Agreement | GOLD |
| `SRC_SPEECHOCEAN_KASHMIRI` | Speechocean Kashmiri Speech | 115.00h | 100+ | Commercial Licensing | GOLD |
| `SRC_BIBLE_BRAIN_KASHMIRI` | BibleBrain Kashmiri New Testament | 40.00h | Multiple | Developer API Key | SILVER |
| **TOTAL BACKLOG** | | **219.21h** | **300+** | | |
