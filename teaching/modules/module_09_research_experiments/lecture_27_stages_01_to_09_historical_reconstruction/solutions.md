# Solutions: Lecture 27 (Research Archaeology: Stages 01 to 09)

---

## Solution to Exercise 1
1. In Kaldi, an alignment WFST $H \circ C \circ L$ is composed from HMM states ($H$), context-dependency triphones ($C$), and the pronunciation lexicon ($L$). If a word in the transcription cannot be mapped to a valid sequence of acoustic phonemes in $L$, the composition graph yields an empty set (no valid paths from start to final state).
2. The Urdu lexicon had no phonetic entries for Kashmiri central vowels (/ɨ/, /əː/) and palatalized consonants ($C^j$). When transcribers wrote words containing these vowels, the forced aligner found zero matching paths and discarded the entire utterance.
3. Discarded percentage:

$$100\% - 61.8\% = \mathbf{38.2\% \text{ of utterances discarded!}}$$

---

## Solution to Exercise 2
1. Early Stage 05 benchmarks evaluated on generic news broadcasts where sibilants and affricates formed a modest fraction of total characters.
2. Overall WER was high (~42%), masking the specific 14% sibilant error rate within general acoustic noise.
3. The defect was only discovered during fine-grained phonetic confusion matrix auditing in Stage 26.
