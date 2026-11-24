# Instructor Delivery Script: Lecture 21 (Greedy vs. Beam Search Decoding)

## Timetable (90 Minutes)
```
00:00 - 00:20 | The MAP Decoding Paradox: Why Greedy Argmax Fails
00:20 - 00:50 | Mathematical Counterexample: The Multi-Path Marginalization Trap
00:50 - 01:15 | Step-by-Step Walkthrough: Prefix Beam Search (P_b and P_nb)
01:15 - 01:25 | Code Walkthrough: src/decoding/beam_search.py
01:25 - 01:30 | Summary & Homework Assignment
```

---

## Instructor Notes & Dialogue

### 00:20 - 00:50 | The Board Counterexample
- Write the probabilities on the board:
  - Frame 1: $P(\text{ك}) = 0.40, P(\text{ت}) = 0.35, P(\epsilon) = 0.25$
  - Frame 2: $P(\text{ٲ}) = 0.40, P(\text{ت}) = 0.35, P(\epsilon) = 0.25$
- Look at the class:
  > "What does greedy search pick?
  > Frame 1: 'ك' (40%). Frame 2: 'ٲ' (40%).
  > Greedy output: 'کٲ' with probability 16%.
  > Now calculate the probability of the single letter 'ت':
  > It could be 'ت' followed by blank (12.25%).
  > It could be blank followed by 'ت' (12.25%).
  > It could be 'ت' followed by 'ت' which collapses to 'ت' (12.25%).
  > Total probability of 'ت': 36.75%!
  > 'ت' is more than twice as likely as 'کٲ'!
  > But greedy search was blinded by local argmax and missed it completely.
  > This is why beam search with prefix path merging is mathematically mandatory."
