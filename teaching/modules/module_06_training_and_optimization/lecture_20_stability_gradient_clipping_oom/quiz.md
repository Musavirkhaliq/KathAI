# Quiz: Lecture 20 (Numerical Stability and OOM Recovery)

---

### Question 1:
What causes the majority of CUDA OOM crashes in deep speech model training?
- [A] The hard drive running out of space.
- [B] Sudden spikes in sequence frame length ($T_{\max}^2$) combined with GPU memory fragmentation from variable-length audio allocations.
- [C] Using Python instead of C++.
- [D] Monophonic audio.

**Correct Answer**: **[B]**  
*Explanation*: Quadratic self-attention memory spikes when long audio clips enter batches.

---

### Question 2:
What PyTorch command releases unused cached memory blocks back to the GPU driver?
- [A] `torch.cuda.empty_cache()`
- [B] `torch.cuda.delete()`
- [C] `os.system("rm -rf")`
- [D] `torch.clear()`

**Correct Answer**: **[A]**  
*Explanation*: `empty_cache()` frees cached unallocated memory blocks.

---

### Question 3:
What flag in `torch.nn.CTCLoss` clamps infinite loss values when $T < U$?
- [A] `clamp=True`
- [B] `zero_infinity=True`
- [C] `safe_mode=True`
- [D] `ignore_nan=True`

**Correct Answer**: **[B]**  
*Explanation*: `zero_infinity=True` replaces infinite losses with zero, preventing gradient explosion.

---

### Question 4:
Why is bucket sorting by duration effective at preventing OOM errors?
- [A] It deletes all clips longer than 5 seconds.
- [B] It groups clips of similar length together, preventing a single long clip from forcing the entire batch to pad to maximum length.
- [C] It converts 16 kHz audio to 8 kHz.
- [D] It changes the model architecture.

**Correct Answer**: **[B]**  
*Explanation*: Bucketing minimizes both padding waste and peak memory spikes.

---

### Question 5:
What should a production training loop do when an OOM exception is caught?
- [A] Terminate the entire script immediately.
- [B] Call `empty_cache()`, zero the gradients, skip the problematic batch (or halve micro-batch size), and continue training seamlessly.
- [C] Reinstall CUDA.
- [D] Delete the dataset.

**Correct Answer**: **[B]**  
*Explanation*: Graceful OOM recovery preserves training continuity.
