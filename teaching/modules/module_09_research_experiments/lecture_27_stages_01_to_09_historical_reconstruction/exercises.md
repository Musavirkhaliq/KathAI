# Exercises: Lecture 27 (Research Archaeology: Stages 01 to 09)

---

## Exercise 1: Autopsy of the Urdu Kaldi Baseline
Review the historical Stage 01 log excerpt:
`"WARNING: Fst has no paths. Discarding utterance uttr_0412. Total aligned: 61.8%"`
1. Explain what `"Fst has no paths"` means in Kaldi WFST alignment.
2. Why did this occur specifically on Kashmiri audio files when processed through an Urdu pronunciation lexicon?
3. Calculate the percentage of utterances discarded during alignment.

---

## Exercise 2: Tracing the Sibilant Defect
In Stage 03, the Bolbosh audio was resampled with a filter cutting off frequencies above 5.8 kHz.
Explain why this bug escaped detection during early Stage 05 evaluation and was only uncovered when the model was evaluated on sibilant-rich test sentences.
