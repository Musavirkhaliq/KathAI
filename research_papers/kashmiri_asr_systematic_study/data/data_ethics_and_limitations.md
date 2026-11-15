# Dataset Ethics, Licensing, and Sociolinguistic Limitations

This document addresses the legal provenance, ethical stewardship, redistribution rights, and sociolinguistic representation of the speech resources curated for Kashmiri ASR.

---

## 1. Ethical Stewardship and Human Speech Consent

### 1.1 Community and Crowdsourced Recordings
Speech data acquired from community initiatives (Programindz, crowdsourced Whisper alignments) was gathered under community participation agreements where speakers contributed speech for open linguistic research and technology preservation. All participant names and identifying metadata were removed and replaced with non-invertible anonymous tokens (`spk_XXXX`).

### 1.2 Privacy Considerations
Recordings were filtered to eliminate private conversations containing personal identifiable information (PII), phone numbers, street addresses, or banking details. No personal communications were transcribed or included in the training sets.

---

## 2. Redistribution Rights & Legal Licensing Tiers

To respect intellectual property rights and data usage agreements, the curated dataset is structured into distinct legal tiers:

| Tier | Included Datasets | Licensing Framework | Public Redistribution Policy |
|:---|:---|:---|:---|
| **Tier 1: Permissive Open** | Bolbosh, IndicVoices, IndicVoices-R, XKaab | Creative Commons Attribution 4.0 (CC-BY 4.0) | **Fully permitted** with appropriate institutional attribution. |
| **Tier 2: ShareAlike Open** | OpenSLR SLR122 | Creative Commons Attribution-ShareAlike 4.0 (CC-BY-SA 4.0) | **Permitted**; derived datasets must adopt compatible CC-BY-SA terms. |
| **Tier 3: Non-Commercial Benchmark** | Meta Omnilingual Kashmiri Slice | Creative Commons Attribution-NonCommercial 4.0 (CC-BY-NC 4.0) | **Evaluation only**; prohibited from commercial redistribution. |
| **Tier 4: Restricted Educational** | Mirwaiz Quran, Dr. War Quran, KTTS | Educational & Research Grant | **RESTRICTED**. Audio files must NOT be redistributed publicly. Fine-tuned model checkpoints are permissible for non-commercial academic research. |

> **Redistribution Compliance Guideline**:  
> In public repository releases and replication packages, Tier 4 audio files must be excluded. Manifests providing download links, segmentation timestamps, and public scripts must be provided instead of raw audio bytes.

---

## 3. Sociolinguistic Limitations & Dialectal Representation

### 3.1 Dialectal Skew
The compiled corpus exhibits a significant regional skew toward the **Yamraz (Central / Srinagar Standard)** and **Maraz (Southern Kashmiri)** dialect zones, which account for over 85% of total audio duration.
- **Under-Represented Dialects**:
  - **Kamraz (Northern Kashmir / Baramulla / Kupwara)**: Approximately 10% representation.
  - **Kishtwari and Poguli**: Less than 3% representation.
  - **Neelum Valley / Azad Kashmir Kashmiri**: Negligible representation.

As a result, acoustic models fine-tuned on this corpus achieve lower recognition accuracy on peripheral Dardic dialects with distinct vowel length shifts and phonetic reductions.

### 3.2 Gender and Age Representation
- **Gender Balance**: The dataset exhibits an approximate 65% male to 35% female speaker distribution, primarily driven by broadcast and religious audio collections where male speakers predominate.
- **Age Demographics**: Adult speakers (ages 18–60) represent over 92% of the speech duration. Child speech (< 14 years) and geriatric speech (> 70 years) are under-represented, leading to higher acoustic error rates on these demographic cohorts.
