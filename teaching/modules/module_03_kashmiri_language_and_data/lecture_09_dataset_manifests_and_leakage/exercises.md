# Exercises: Lecture 09 (Dataset Manifest Engineering)

---

## Exercise 1: Calculating Leakage Contamination
A research intern creates a test split of 200 utterances.
Upon forensic audit, 32 of those utterances belong to speakers who also contributed 140 utterances to the training split.
1. What is the speaker contamination rate of this test set?
2. If the intern reports an 18.2% WER, explain why this metric is untrustworthy and predict the directional shift in WER when evaluated on a disjoint split.

---

## Exercise 2: Bucket Sorting for Dynamic Batching
A training manifest contains audio files with durations ranging from 1.0 second to 18.0 seconds.
Explain why training with random batch sampling causes excessive GPU memory waste due to `<pad>` tokens, and calculate the padding efficiency of sorting samples into duration buckets.
