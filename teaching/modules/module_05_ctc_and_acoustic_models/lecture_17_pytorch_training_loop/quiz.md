# Quiz: Lecture 17 (The Complete PyTorch CTC Training Loop)

---

### Question 1:
What is the primary role of `GradScaler` in PyTorch Automatic Mixed Precision (AMP)?
- [A] To speed up audio resampling.
- [B] To scale loss values by a large factor before backpropagation, preventing small gradient values from underflowing to zero in 16-bit floating point (FP16).
- [C] To normalize audio waveforms to [-1, 1].
- [D] To decode text hypotheses.

**Correct Answer**: **[B]**  
*Explanation*: Dynamic scaling prevents FP16 underflow in deep networks.

---

### Question 2:
What gradient norm clipping threshold is standard in EduCasheer training?
- [A] 0.01
- [B] 1.0
- [C] 100.0
- [D] No clipping

**Correct Answer**: **[B]**  
*Explanation*: Clipping at $\le 1.0$ prevents destabilizing step updates while preserving gradient direction.

---

### Question 3:
Why must loss be divided by `accum_steps` when using gradient accumulation?
- [A] To save disk space.
- [B] Because gradients accumulate additively; dividing by $K$ ensures the accumulated gradient equals the true arithmetic mean over the effective batch.
- [C] To convert from natural log to log base 10.
- [D] Because PyTorch requires integer loss.

**Correct Answer**: **[B]**  
*Explanation*: PyTorch accumulates gradients additively; scaling by $1/K$ computes the correct average gradient.

---

### Question 4:
What occurs if `scaler.unscale_(optimizer)` is omitted before calling `clip_grad_norm_`?
- [A] Gradients are clipped based on their artificially scaled values rather than their true magnitudes, resulting in overly aggressive clipping.
- [B] PyTorch crashes with a segmentation fault.
- [C] Model weights double in size.
- [D] The audio sample rate drops.

**Correct Answer**: **[A]**  
*Explanation*: Clipping must operate on true unscaled gradients to apply the correct threshold $M=1.0$.

---

### Question 5:
What tensor dimension ordering is expected by PyTorch `nn.CTCLoss` for log-probabilities?
- [A] `[Batch, Time, Vocab]`
- [B] `[Time, Batch, Vocab]`
- [C] `[Vocab, Time, Batch]`
- [D] `[Time, Vocab]`

**Correct Answer**: **[B]**  
*Explanation*: Native PyTorch `CTCLoss` requires `[Time, Batch, Vocab]` format unless `batch_first=True` is explicitly specified.
