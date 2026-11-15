# Remaining Scientific and Linguistic Challenges in Kashmiri ASR

This document catalogs the unresolved acoustic, linguistic, and engineering challenges that must be addressed to push Kashmiri speech recognition past the current 24.3% WER frontier.

---

## 1. Challenge 1: Acoustic Out-of-Beam Error Dominance (78.6%)

As established in the Oracle analysis, **3,240 out of 4,121 errors (78.6%) never enter the 16-best search beam**.
- Rescorers, rankers, and language model decoders operate strictly on hypotheses present within the beam. No decoding optimization can correct an error if the ground truth words are completely absent from the acoustic beam.
- Overcoming this challenge requires expanding base acoustic representation capacity:
  1. Fine-tuning larger foundation models (e.g., XLS-R 1B or 2B parameters, or Conformer CTC architectures).
  2. Ingesting 500+ additional hours of high-fidelity conversational audio with verified speaker-disjoint boundaries.

---

## 2. Challenge 2: Orthographic Inconsistency and Diacritic Ambiguity

Kashmiri lacks a universally standardized spelling convention across publishers, digital keyboards, and dialect regions:
- **Diacritic Volatility**: Transcribers frequently omit or exchange central vowel markers (`U+0672`, `U+065F`, `U+065A`). In many cases, an "ASR error" represents a valid spelling variant that disagrees with an unnormalized ground-truth transcript.
- **Future Direction**: Developing an audited Kashmiri orthographic normalizer that collapses purely cosmetic spelling variants without destroying phonemic minimal pairs.

---

## 3. Challenge 3: Dialectal and Prosodic Variation

- **Peripheral Dialects**: Speakers from Kamraz (Northern Kashmir) and Kishtwar exhibit distinct vowel shifts and consonant elisions that are poorly modeled by models trained primarily on Srinagar / Maraz speech.
- **Code-Switching**: Spontaneous Kashmiri discourse is heavily code-switched with English, Urdu, and Hindi. Monolingual character CTC vocabularies struggle when speakers seamlessly transition into complex English technical vocabulary.

---

## 4. Challenge 4: Held-Out Benchmark Independence

All diagnostic tuning in recent stages utilized `dev_assessment.jsonl`. To achieve peer-reviewed publication rigor, the models must be evaluated blindly on the sealed 6,315-sample Benchmark v1.1, ensuring zero contamination and definitive verification of generalizability.
