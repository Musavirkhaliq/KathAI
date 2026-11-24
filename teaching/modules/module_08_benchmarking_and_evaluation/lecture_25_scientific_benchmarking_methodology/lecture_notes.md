# Lecture 25: Scientific Benchmarking: The 8-Domain Kashmiri Evaluation Protocol

---

## 1. The Reproducibility Crisis in AI Benchmarks

In automatic speech recognition research, benchmark figures are frequently non-reproducible across laboratories:
- Different researchers evaluate on unstandardized text normalizers.
- Test audio contains silent speaker leakage from training partitions.
- Audio resampling introduces high-frequency roll-off.
- Evaluations are reported only on clean studio speech, concealing failure on natural spontaneous conversations.

To establish an authoritative, unassailable scientific standard for Kashmiri ASR, EduCasheer instituted the **8-Domain Frozen Evaluation Protocol** (`benchmarks/`).

---

## 2. The 8 Evaluation Domains and Acoustic Profiles

| Domain ID | Domain Name | Audio Profile | Primary Acoustic / Linguistic Challenge | Reference Hours |
| :---: | :--- | :--- | :--- | :---: |
| **DOM-01** | Studio Clean Read | Studio condenser mic, zero reverberation | Baseline phonetic clarity, vowel length contrasts | 2.50h |
| **DOM-02** | Prompted Conversational | Mobile headsets, diverse regional accents | Natural tempo variation, unstressed central vowels | 3.10h |
| **DOM-03** | Formal Broadcast News | Radio studio, standard literary Kashmiri | Complex Persianized vocabulary, formal syntax | 3.50h |
| **DOM-04** | Quranic & Expository | Professional vocal delivery, high register | Dense Arabic loanwords, pharyngealized consonants | 2.20h |
| **DOM-05** | Spontaneous Mosque (WAAZ) | Large hall reverberation, crowd coughs, echo | Severe non-stationary acoustics, spontaneous elisions | 4.20h |
| **DOM-06** | Community Smartphone | Near-field Android/iOS mics, ambient home noise | Variable gain, background traffic, room reverberation | 2.80h |
| **DOM-07** | Dialectal Folk Narratives | Elderly speakers, Southern/Northern rural accents | Archaic morphology, regional phonetic shifts | 2.40h |
| **DOM-08** | Educational Lectures | Podium dynamic mics, chalk writing, room echo | Code-switching with English terms, whiteboard pauses | 2.00h |

**Total Frozen Benchmark**: **22.70 hours** across 12,840 audio utterances with 0.00% speaker overlap against all training manifests.

---

## 3. The Protocol Invariants

To certify any model checkpoint in EduCasheer, an evaluation run must satisfy **five non-negotiable invariants**:
1. **Zero Contamination**: Absolute speaker disjointness verified by graph bipartite audit.
2. **Deterministic Text Normalization**: Transcripts and predictions must both pass through Certified Normalizer v3 before metric calculation.
3. **Certified Polyphase Resampling**: Audio must be resampled using Kaiser sinc interpolation ($\beta = 8.5555$, rolloff = 0.94).
4. **Frozen Decoder Hyperparameters**: Decoder parameters ($T=1.08, \alpha=0.50, \beta=1.50$) must remain constant across all 8 domains; per-domain test tuning is disqualified as hyperparameter leakage.
5. **Micro-Averaged Reporting**: Results must report both per-domain micro-averaged WER and the composite multi-domain micro-averaged WER.
