# Exercises: Lecture 30 (Checkpoint Forensics)

---

## Exercise 1: File vs. Tensor Hashing
Explain why running `sha256sum checkpoint.pt` in Linux bash can produce two different hashes for two files that contain mathematically identical neural network weights.

---

## Exercise 2: State Dict Dissection
A forensic auditor inspects three checkpoint state dicts:
- Checkpoint A: `lm_head.weight` has shape `[32, 1024]`
- Checkpoint B: `lm_head.weight` has shape `[47, 1024]`, `lm_head.bias[0] = 0.0`
- Checkpoint C: `lm_head.weight` has shape `[47, 1024]`, `lm_head.bias[0] = 3.0`
Match each checkpoint to its historical identity:
1. `S20R` (B-02C recovery checkpoint)
2. `S22E` (Certified E22-01 production checkpoint)
3. Early Urdu baseline checkpoint
Explain your reasoning.
