# Kashmiri ASR Benchmark Construction Methodology

This document details the design, filtering, curation, and integrity verification protocols governing the creation of standardized evaluation benchmarks for Kashmiri ASR.

---

## 1. Objectives of Benchmark Construction

Prior academic evaluations of Kashmiri ASR have suffered from:
1. **Unstratified Test Sets**: Evaluating on small, unrepresentative subsets (e.g., 100 sentences from a single speaker) that fail to capture regional dialect variance or conversational dynamics.
2. **Hidden Contamination**: Leaking speakers or resliced audio segments between training and evaluation splits.
3. **Preprocessing Inconsistencies**: Testing with varying sample rates, unstandardized diacritics, or divergent text normalizations.

The EduCasheer Kashmiri ASR Benchmark was constructed to provide a reproducible, multi-tiered, and speaker-disjoint evaluation foundation.

---

## 2. Multi-Tier Benchmark Architecture

The benchmark categorizes evaluation speech into four functional tiers reflecting distinct acoustic and linguistic environments:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   EDUCASHEER BENCHMARK ARCHITECTURE                    │
├───────────────────┬───────────────────┬────────────────────────────────┤
│ Tier Identifier   │ Audio Domain      │ Acoustic Characteristics       │
├───────────────────┼───────────────────┼────────────────────────────────┤
│ Tier A (Primary)  │ Clean & Convers.  │ Studio read & near-field phone │
│                   │ Native Dialogue   │ SNR > 20 dB, native 16 kHz     │
├───────────────────┼───────────────────┼────────────────────────────────┤
│ Tier B (Field)    │ Conversational    │ Ambient background noise       │
│                   │ Online & Crowds.  │ SNR 12–20 dB, mobile handsets  │
├───────────────────┼───────────────────┼────────────────────────────────┤
│ Tier C (Religious)│ Classical Quranic │ Formal Arabic-Kashmiri register│
│                   │ Recitation & Comm.│ High studio fidelity           │
├───────────────────┼───────────────────┼────────────────────────────────┤
│ Challenge Sets    │ OOD & Dialectal   │ Meta Omnilingual, Long Bolbosh │
│                   │ Stress Testing    │ Variable duration, low SNR     │
└───────────────────┴───────────────────┴────────────────────────────────┘
```

---

## 3. Benchmark Construction Pipeline

### Step 1: Candidate Ingestion
Audio recordings from candidate pools (IndicVoices-R quarantined, EduCasheer online recordings, King Fahd translations) were aggregated and indexed.

### Step 2: Quality Gating
Candidates were filtered against the five pretraining validation gates:
- Duration: $0.6 \text{ s} \le t \le 25.0 \text{ s}$
- CTC Frame Feasibility: $T \ge L$
- SNR Feasibility: Samples $< 10 \text{ dB}$ excluded.

### Step 3: Exact Audio and Speaker Overlap De-duplication
Cross-corpus hashing verified that zero benchmark candidates matched training audio. All candidate speakers were mapped against training speaker IDs to enforce strict disjointness.

### Step 4: Transcript Verification & Normalization
Transcripts were audited for Perso-Arabic orthographic fidelity, ensuring all central vowel diacritics were correctly encoded in Unicode NFKC format.
