# Research Connections: Lecture 06

---

## 1. Repository Grounding
- **`configs/vocab.json`**: The 47-token vocabulary file formalizing the Kashmiri alphabet.
- **`src/normalization/kashmiri_normalizer.py`**: Enforces strict Unicode mapping for central vowels and palatalization characters.

---

## 2. Research History
In Stage 01, the team tested an Urdu Kaldi acoustic model. Because Urdu lacks the central vowels /ɨ/ and /əː/, the model mapped Kashmiri words to random phonetic substitutions, resulting in an 88.4% WER failure.
