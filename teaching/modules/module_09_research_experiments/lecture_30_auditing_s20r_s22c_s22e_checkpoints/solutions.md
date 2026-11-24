# Solutions: Lecture 30 (Checkpoint Forensics)

---

## Solution to Exercise 1
1. PyTorch's `torch.save()` serializes state dicts into a zipped container format (`application/zip`).
2. The zip container embeds metadata including file creation timestamps, filesystem permissions, and local user IDs.
3. Consequently, re-saving the exact same floating-point tensor array on two different days produces different zip headers, yielding different file-level SHA-256 hashes even though the tensor values are 100% bit-for-bit identical.

---

## Solution to Exercise 2
1. **Checkpoint A (`[32, 1024]`)**: Corresponds to the **Early Urdu baseline**. Standard Urdu Perso-Arabic alphabets without Kashmiri extensions contain ~32 characters.
2. **Checkpoint B (`[47, 1024]`, bias = 0.0)**: Corresponds to **`S20R` (B-02C)**. It has the 47 Kashmiri characters, but was trained before the blank bias initialization strategy was adopted in Stage 22.
3. **Checkpoint C (`[47, 1024]`, bias = 3.0)**: Corresponds to **`S22E` (Certified E22-01)**. It features the complete 47-character vocabulary and the positive blank bias initialization (+3.0).
