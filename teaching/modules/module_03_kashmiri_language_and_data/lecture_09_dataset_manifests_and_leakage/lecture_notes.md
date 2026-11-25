# Lecture 09: Dataset Manifest Engineering and the Zero-Overlap Speaker Split

---

## 1. The Contamination Crisis in Machine Learning Benchmarks

A machine learning benchmark is only as valid as the independence of its evaluation sets. In speech recognition, **speaker leakage** represents the most pernicious form of data contamination:
- When audio from the same speaker appears in both training and test partitions, deep neural acoustic models memorize speaker-specific biometric characteristics: fundamental frequency ($F_0$), vocal tract length, characteristic formant bandwidths, and recording channel impulse responses.
- In historical experiments (Stage 06 vs. Stage 14), evaluating on a contaminated split produced an **artificially optimistic 16.8% WER**, whereas evaluating the exact same checkpoint on a strictly disjoint split revealed a true **34.2% WER**—a catastrophic 17.4% performance illusion!

---

## 2. Bipartite Graph Partitioning for Zero-Speaker Overlap

To guarantee zero speaker leakage across thousands of utterances where metadata may be imperfect, EduCasheer formalized data splitting as a **Graph Partitioning Problem**:

Let $G = (V_S \cup V_U, E)$ be a bipartite graph where:
- $V_S$ is the set of speaker entities.
- $V_U$ is the set of audio utterances.
- An edge $(s, u) \in E$ exists if speaker $s$ uttered segment $u$.

### Partition Constraints:
We partition utterances into three sets: $\text{Train} (U_{\text{tr}})$, $\text{Validation} (U_{\text{val}})$, and $\text{Test} (U_{\text{te}})$ such that:
1. **Strict Speaker Disjointness**:

$$S(U_{\text{tr}}) \cap S(U_{\text{val}}) = \emptyset, \quad S(U_{\text{tr}}) \cap S(U_{\text{te}}) = \emptyset, \quad S(U_{\text{val}}) \cap S(U_{\text{te}}) = \emptyset$$

2. **Acoustic Diversity Balance**:
   Each partition must maintain proportional representation of gender, dialect regions, and recording acoustic types (studio vs. field).
3. **Temporal Target Ratios**:
   Target duration split: $80\%$ Train, $10\%$ Validation, $10\%$ Test.

---

## 3. The 8 Frozen Disjoint Evaluation Domains

To prevent overfitting to any single acoustic or linguistic distribution, EduCasheer established **8 Frozen Disjoint Benchmark Domains** (`benchmarks/`):

1. **Domain 1: Clean Studio Read Speech** (Bolbosh Disjoint)
2. **Domain 2: Prompted Conversational** (IndicVoices Disjoint)
3. **Domain 3: Formal Broadcast News** (Radio Kashmir)
4. **Domain 4: Quranic & Expository Commentary** (High Perso-Arabic loan register)
5. **Domain 5: Spontaneous Mosque Sermons** (WAAZ Reverberant)
6. **Domain 6: Community Mobile Audio** (Near-field smartphone microphones)
7. **Domain 7: Elderly Dialectal Folk Narratives** (Rural South/North Kashmir)
8. **Domain 8: Classroom & Educational Audio** (Podium reverberation and chalk pauses)

Any checkpoint claiming state-of-the-art performance must be evaluated across all 8 domains without test-set tuning.

---

## 4. Production Manifest Architecture: JSON Lines (`.jsonl`)

EduCasheer standardizes all training manifests in JSON Lines format (`manifests/*.jsonl`):

```json
{"audio_filepath": "data/audio/bolbosh/utt_001.wav", "duration": 4.52, "text": "کٲشُر چھُ اَکھ قٔدیٖم زَبان", "speaker_id": "SPK_B012", "domain": "studio_clean"}
```

### Mandatory Manifest Schema Fields:
- `audio_filepath`: Relative path to certified 16 kHz 16-bit mono WAV.
- `duration`: Float duration in seconds (enables batch bucket sorting to minimize padding).
- `text`: Certified Normalizer v3 normalized transcription.
- `speaker_id`: Unique speaker identifier used by graph partition verification.
- `domain`: Acoustic domain identifier.
