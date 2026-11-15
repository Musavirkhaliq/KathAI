# Dataset Card: Consolidated Kashmiri Speech Corpus

## 1. Dataset Summary
The Consolidated Kashmiri Speech Corpus is a curated, audited, and partitioned multi-source speech dataset designed for automatic speech recognition (ASR) research in Kashmiri (*Kōshur*, ISO 639-3: `kas`). It aggregates 12 distinct audio collections totaling **106,487 recordings (214.65 hours)** across broadcast, read, conversational, mobile field, and religious recitation speech genres.

The dataset addresses the acute scarcity and fragmentation of Dardic speech resources by establishing standardized 16 kHz 16-bit mono audio representations, cryptographic provenance, and strictly speaker-disjoint splits.

```
Total Utterances: 106,487
Total Audio Duration: 214.65 Hours
Native Speakers: 630+ Verified Speakers
Dialect Zones: Maraz (South), Kamraz (North), Yamraz (Central / Srinagar)
Primary Script: Perso-Arabic (kas_Arab) with standardized central vowel diacritics
Acoustic Format: 16,000 Hz, 16-bit Mono PCM WAV
```

---

## 2. Dataset Structure

### 2.1 Subsets and Partitions
To support rigorous machine learning development while preventing data contamination, the corpus is partitioned into distinct functional subsets:

1. **Master Raw Inventory (`Data/` & `kashmiri_dataset/raw/`)**: Complete uncompressed historical shards (106,487 audio files, 214.65 hours).
2. **Verified Training Eligible Manifest (`asr_training_eligible_manifest.jsonl`)**: 106,487 samples passing duration ($0.6\text{s} \le t \le 25.0\text{s}$) and CTC feasibility gates.
3. **Acoustic Training Series Splits**:
   - `B-01`: 53,460 utterances (92.94 hours)
   - `B-02A`: 64,522 utterances (104.93 hours)
   - `B-02B`: 80,106 utterances (123.45 hours)
   - `B-02C`: 96,664 utterances (165.32 hours)
   - `Tier A Train`: 63,078 high-confidence utterances
   - `Tier B Train`: 31,617 weakly labeled utterances
4. **Development Validation (`val.jsonl`)**: 6,728 canonical utterances (104,685 words) with 0.00% speaker overlap against training splits.
5. **Diagnostic Assessment (`dev_assessment.jsonl`)**: 1,001 utterances (16,845 words; 570 16 kHz native, 431 22.05 kHz Bolbosh).
6. **Diagnostic Calibration (`dev_calibration.jsonl`)**: 1,000 utterances (583 16 kHz native, 417 22.05 kHz Bolbosh).
7. **Sealed Benchmark (`benchmark_v1_1_primary.jsonl`)**: Multi-tier quarantined test corpus.

### 2.2 Schema of Standard Manifest Records
Each record in the `.jsonl` manifests contains the following audited fields:
```json
{
  "id": "bolbosh_002931",
  "audio": "kashmiri_dataset/processed/audio/bolbosh/bolbosh_002931.wav",
  "text": "تمِ وۆن زِ کٲشِر زَبان چھِ اَسَن",
  "duration": 4.12,
  "speaker_id": "spk_bolbosh_042",
  "sample_rate": 22050,
  "source": "bolbosh",
  "tier": "A",
  "snr_db": 24.5,
  "sha256": "4b91e9f1a26d7f9038d8f7e8c16d5e50942e88b8f2c050014b281f6ee3252516"
}
```

---

## 3. Data Ingestion & Preprocessing

### 3.1 Audio Standard and Resampling Integrity
Audio files originate with heterogeneous recording sample rates:
- **16,000 Hz Native**: AI4Bharat IndicVoices, IndicVoices-R, OpenSLR-122, XKaab, KTTS, Mirwaiz Quran.
- **22,050 Hz Native**: Gaash-Lab Bolbosh (39,996 utterances, 58.40 hours).

**Canonical Audio Ingestion Contract**:  
All audio passed to acoustic feature extractors must be processed via `load_canonical_audio()` (`kashmiri_dataset/audio/canonical_audio.py`). If the native sample rate is 22,050 Hz, a 64-point Kaiser windowed sinc interpolator ($\beta=14.77$, rolloff=0.948) downsamples the waveform to 16,000 Hz. Native 16,000 Hz files bypass resampling with zero numerical distortion.

### 3.2 Text Normalization & Orthography
Kashmiri Perso-Arabic text exhibits severe encoding divergence across keyboards and source platforms. We apply a strict normalization pipeline (`kashmiri_dataset/scripts/normalize_transcripts.py`):
1. **Unicode NFKC Normalization**: Resolves precomposed vs decomposed character representations.
2. **Kashmiri Grapheme Canonicalization**:
   - Arabic Kaf (`U+0643`) $\rightarrow$ Keheh (`U+06A9`).
   - Arabic Heh (`U+0647`) $\rightarrow$ Heh Goal (`U+06C1`), reserving Doachashmee Heh (`U+06BE`) strictly for consonant aspiration (`کھ`, `تھ`, `چھ`).
   - Dual-dot Arabic Yeh (`U+064A`) $\rightarrow$ Farsi Yeh (`U+06CC`).
   - Central Vowel Marks: Precomposed Alef with Wavy Hamza (`U+0672`) preserved; independent combining wavy hamza below (`U+065F`) canonicalized.
3. **Punctuation & Digit Conversion**: Western punctuation is stripped or converted; Arabic digits are expanded into spoken word forms for ASR training.

---

## 4. Quality Control & Validation Gates

Every audio sample must pass five automated validation gates before inclusion:
- **Gate 1 (Format Integrity)**: Valid PCM WAV format readable by `torchaudio`.
- **Gate 2 (Duration Feasibility)**: $0.6 \text{ s} \le t \le 25.0 \text{ s}$.
- **Gate 3 (CTC Frame Feasibility)**: Acoustic feature frames $T = \lfloor 16000 \times t / 320 \rfloor \ge L$ (transcript characters).
- **Gate 4 (Signal-to-Noise Ratio)**: Welford-RMS energy ratio $>12$ dB (Tier B) or $>20$ dB (Tier A). Samples $<10$ dB quarantined.
- **Gate 5 (Transcript Verification)**: 0 characters outside the 159-token inventory.

---

## 5. Licensing and Ethical Considerations
- **Public & Permissive Sources**: Bolbosh (CC-BY 4.0), IndicVoices (CC-BY 4.0), IndicVoices-R (CC-BY 4.0), XKaab (CC-BY 4.0), OpenSLR-122 (CC-BY-SA 4.0).
- **Restricted Research Sources**: Mirwaiz Quran and Dr. War Quran are licensed strictly for non-commercial educational and linguistic research. Raw audio must NOT be redistributed.
- **Privacy & Anonymization**: All speaker IDs are anonymized hashes (`spk_XXXX`). No personally identifiable information (PII) is retained in the metadata manifests.
