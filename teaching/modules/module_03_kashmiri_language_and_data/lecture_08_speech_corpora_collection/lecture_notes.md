# Lecture 08: Speech Corpora Collection, Curation, and Noise Assessment

---

## 1. The Speech Data Scarcity Landscape for Kashmiri

Prior to the EduCasheer research project, Kashmiri was classified by international computational linguistics as a **severely digitally under-resourced language**. While high-resource languages like English enjoy thousands of hours of curated speech (e.g., LibriSpeech 1,000h, GigaSpeech 10,000h, People's Speech 30,000h), public Kashmiri speech resources totaled fewer than 20 hours of fragmented, noisy recordings.

To train deep acoustic models capable of robust generalization, the project assembled, audited, and curated a multi-source corpus from **12 distinct acoustic sources**.

---

## 2. The 12 Speech Sources of the EduCasheer Ecosystem

| Source ID | Source Name | Raw Duration | Nature & Acoustic Environment |
| :--- | :--- | :--- | :--- |
| **SRC-01** | Bolbosh Kashmiri | 18.4 hours | Studio and clean read speech; original 22.05 kHz recording |
| **SRC-02** | IndicVoices Kashmiri | 24.2 hours | Multi-speaker read and prompted speech; diverse rural/urban accents |
| **SRC-03** | Quranic Kashmiri Broadcasts | 14.8 hours | Expository religious discourse; formal high register with Arabic loans |
| **SRC-04** | WAAZ Spontaneous Sermons | 32.6 hours | Spontaneous open-air mosque sermons; heavy reverberation and crowd noise |
| **SRC-05** | Community Mobile Voice | 16.5 hours | Crowdsourced smartphone recordings; variable near-field microphone quality |
| **SRC-06** | Radio Kashmir Broadcasts | 28.0 hours | Broadcast news and cultural features; high linguistic standard |
| **SRC-07** | Educational Lectures | 12.1 hours | Classroom and podium lectures; whiteboard pauses and room echo |
| **SRC-08** | Folk Tales & Oral Stories | 15.3 hours | Elderly native speakers; rich dialectal idioms and archaic vocabulary |
| **SRC-09** | Conversational Dialogues | 19.7 hours | Spontaneous two-party conversations; overlapping speech and laughter |
| **SRC-10** | Children's Kashmiri Speech | 8.4 hours | Primary school reading; higher pitch ($F_0 > 280 \text{ Hz}$) and articulation pauses |
| **SRC-11** | Judicial & Administrative | 11.2 hours | Formal legal and administrative reading; complex Persianized syntax |
| **SRC-12** | Synthetic Alignment Verification | 13.4 hours | High-fidelity synthetic speech used for automated pipeline calibration |

**Total Repository Raw Holdings**: **214.65 hours** across 106,487 audio segments.

---

## 3. The 4-Stage SNR and Quality Curation Pipeline

Raw scraped speech cannot be ingested directly into neural training. EduCasheer engineered an automated 4-stage curation pipeline (`src/data/curation.py`):

```
[ Raw Audio Files ]
         │
         ▼
[ Stage 1: Physical Audio Audit ] ── Check sample rate, bit depth, channel count, header corruption
         │
         ▼
[ Stage 2: Energy & SNR Filtering ] ── WADA-SNR calculation; drop audio with SNR < 12 dB
         │
         ▼
[ Stage 3: Clipping & Dynamic Range ] ── Detect digital clipping (> 1% samples at ±1.0)
         │
         ▼
[ Stage 4: Duration Slicing ] ── Enforce 0.8s <= duration <= 20.0s for stable CTC alignment
         │
         ▼
[ Training-Eligible Clean Manifest ]
```

### Signal-to-Noise Ratio (SNR) Estimation:
Speech SNR is computed using the **Waveform Amplitude Distribution Analysis (WADA-SNR)** algorithm, which models clean speech amplitude as a Gamma distribution and additive noise as a Gaussian distribution. Recordings with $\text{SNR} < 12 \text{ dB}$ are routed to noise-augmentation quarantine rather than clean training splits.
