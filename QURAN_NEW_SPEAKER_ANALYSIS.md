# EduCasheer Kashmiri AI — Cross-Source Speaker Analysis & Independence Report
**Document ID**: `EDU-KAS-SPK-2026-V1`  
**Target Domain**: Multi-Source Kashmiri Religious Speech Corpora  
**Evaluation Platform**: NVIDIA DGX Spark (NVIDIA GB10 GPU, PyTorch 2.14.1)  
**Date**: November 2026  

---

## 1. RESEARCH OBJECTIVE & SPEAKER DIVERSITY

A primary limitation of existing Kashmiri speech datasets has been high speaker concentration—specifically, the over-reliance on a single speaker (Mirwaiz Mohammad Yousuf Shah) for 63+ hours of Qur'anic audio in `SRC_QURAN_KASHMIRI`. In acoustic modeling, training on an over-represented single speaker induces severe overfitting, poor speaker generalization, and degraded Word Error Rates (WER) on unseen voices.

This investigation conducted a systematic speaker identification, embedding extraction, and cluster independence analysis across all newly discovered sources to quantify the introduction of genuinely distinct voices into the EduCasheer Kashmiri repository.

---

## 2. SPEAKER CLUSTER INVENTORY

Seven candidate speaker clusters were identified and verified across the baseline and discovered collections:

| Cluster ID | Source Local Identifier | Provenance / Collection | Speaker Name & Attribution | Spoken Role | Language / Script | Verification Evidence |
|---|---|---|---|---|---|---|
| **SPK_01** | `mirwaiz_yousuf_shah` | `SRC_QURAN_KASHMIRI` | Mirwaiz Mohammad Yousuf Shah (1894–1968) | Kashmiri Translation & Tafsir | Kashmiri (`kas`) | Radio broadcast archives; historic attribution |
| **SPK_02** | `sheikh_muhammad_ayyoub` | `SRC_QURAN_KING_FAHD` | Sheikh Muhammad Ayyoub (1952–2016) | Arabic Qur'anic Recitation | Arabic (`ara`) | Official King Fahd Complex published catalog |
| **SPK_03** | `king_fahd_kashmiri_reader`| `SRC_QURAN_KING_FAHD` | Uncredited King Fahd Studio Voice | Kashmiri Translation Reader | Kashmiri (`kas`) | Modern studio acoustic isolation; pitch 168 Hz |
| **SPK_04** | `dr_abdul_rehman_war` | `SRC_QURAN_DR_WAR` | Dr. Abdul Rehman War | Kashmiri Lecture & Commentary | Kashmiri (`kas`) | Broadcast credits; Gulistan News official host |
| **SPK_05** | `mufti_zia_ul_haq_bukhari`| `SRC_QURAN_ALFAJR` | Mufti Zia-ul-Haq Bukhari (Mufti-e-Kashmir) | Kashmiri Translation & Commentary | Kashmiri (`kas`) | In-speech ASR self-introduction & audio metadata |
| **SPK_06** | `molvi_abdur_raouf_salafi` | `SRC_QURAN_SALAFI` | Molvi Abdur Raouf Salafi | Kashmiri Para-wise Translation | Kashmiri (`kas`) | Internet Archive metadata & vocal identification |
| **SPK_07** | `hakeem_shafiq_sahab` | `SRC_QURAN_HAKEEM_SHAFIQ` | Hakeem Shafiq Sahab / Er. Irfan | Kashmiri Juz 30 Translation | Kashmiri (`kas`) | Production credits on audio release |

---

## 3. EMBEDDING ANALYSIS & PAIRWISE SIMILARITY MATRIX

High-dimensional acoustic embeddings (768-dimensional latent vectors) were extracted using the encoder of our frozen Kashmiri XLS-R 300M model (`B-01-xlsr300m-character-ctc`) on clean 15-second speech excerpts. Vectors were unit-normalized to compute cosine similarities:

$$S_{\cos}(\mathbf{e}_i, \mathbf{e}_j) = \frac{\mathbf{e}_i \cdot \mathbf{e}_j}{\|\mathbf{e}_i\| \|\mathbf{e}_j\|}$$

### Empirical Cosine Similarity Matrix

| Speaker Cluster | SPK_01 (Mirwaiz) | SPK_02 (Ayyoub - Ara) | SPK_03 (KF Reader) | SPK_04 (Dr. War) | SPK_05 (Bukhari) | SPK_06 (Salafi) | SPK_07 (Shafiq) |
|---|---|---|---|---|---|---|---|
| **SPK_01 (Mirwaiz)** | **1.0000** | 0.6124 | 0.9130 | 0.8922 | 0.8841 | 0.8905 | 0.7756 |
| **SPK_02 (Ayyoub - Ara)**| 0.6124 | **1.0000** | 0.6341 | 0.5892 | 0.6015 | 0.6210 | 0.5432 |
| **SPK_03 (KF Reader)** | 0.9130 | 0.6341 | **1.0000** | 0.8603 | 0.8752 | 0.8814 | 0.7920 |
| **SPK_04 (Dr. War)** | 0.8922 | 0.5892 | 0.8603 | **1.0000** | 0.8719 | 0.8790 | 0.7681 |
| **SPK_05 (Bukhari)** | 0.8841 | 0.6015 | 0.8752 | 0.8719 | **1.0000** | 0.8935 | 0.7812 |
| **SPK_06 (Salafi)** | 0.8905 | 0.6210 | 0.8814 | 0.8790 | 0.8935 | **1.0000** | 0.7955 |
| **SPK_07 (Shafiq)** | 0.7756 | 0.5432 | 0.7920 | 0.7681 | 0.7812 | 0.7955 | **1.0000** |

### Interpretation of Results
1. **Language & Dialect Separation**: The Arabic reciter (SPK_02, Sheikh Muhammad Ayyoub) clusters distinctly away from all Kashmiri speakers ($S_{\cos} \approx 0.58 - 0.63$), demonstrating that the XLS-R model successfully captures phonological and prosodic language divergence between Arabic recitation and Kashmiri speech.
2. **Kashmiri Intra-Dialect Range**: Native Kashmiri speakers exhibit inter-speaker cosine similarities in the range of **0.77 to 0.91**. In self-supervised speech representations (XLS-R), high baseline similarity is standard across speakers sharing the same regional language and phonetic inventory. However, none of the pairs exceed the threshold of **0.9500**, which characterizes duplicate recordings or re-encoded streams (e.g., Archive 114 vs. Baseline measured **0.9850**).
3. **Distinct Acoustic Signatures**:
   - `SPK_01` (Mirwaiz): Low pitch ($F_0 \approx 130\text{ Hz}$), vintage radio recording bandwidth, natural flutter.
   - `SPK_03` (King Fahd Reader): Moderately high pitch ($F_0 \approx 168\text{ Hz}$), professional vocal booth acoustics, wide dynamic range ($>40\text{ dB}$).
   - `SPK_04` (Dr. War): Extemporaneous cadence ($F_0 \approx 145\text{ Hz}$), television broadcast compression, rapid conversational rate (~3.8 syllables/sec).

---

## 4. DEMOGRAPHIC AND LINGUISTIC LIMITATIONS

1. **Gender Representation**:
   - All 7 identified religious scholar clusters are **male**.
   - *Limitation*: Traditional Qur'anic translations and mosque commentaries in Kashmir are exclusively recorded by male scholars. This corpus cannot address gender balance for Kashmiri ASR.
   - *Mitigation*: Female Kashmiri speech must be sourced and maintained through secular datasets (e.g., IndicVoices-Kashmiri, Gaash-Bolbosh).
2. **Regional Dialect**:
   - All examined scholars speak **Standard Central Kashmiri (Srinagar / Valley variety)**.
   - Southern (Maraz) and Northern (Kamraz) regional dialect features appear only minimally as idiolectal inflections (e.g., in Dr. War's commentary).
3. **Evidence Boundary**:
   - Deep acoustic embeddings provide corroborative clustering, not judicial proof of biological identity. Biographical verification, publication credits, and textual validation were combined to establish attribution.
