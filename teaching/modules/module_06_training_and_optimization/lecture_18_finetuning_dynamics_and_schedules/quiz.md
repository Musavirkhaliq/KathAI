# Quiz: Lecture 18 (Fine-Tuning Dynamics and Schedules)

---

### Question 1:
What is the primary motivation for applying Layer-wise Learning Rate Decay (LLRD)?
- [A] To reduce GPU memory consumption.
- [B] To allow higher layers to adapt to new linguistic characters while preserving universal acoustic physics in lower layers.
- [C] To automatically quantize model weights.
- [D] To eliminate CTC blank emissions.

**Correct Answer**: **[B]**  
*Explanation*: LLRD protects low-level general acoustic features while allowing top layers to learn task-specific mappings.

---

### Question 2:
What typical decay factor $\xi$ is applied in XLS-R fine-tuning?
- [A] 0.01
- [B] 0.75
- [C] 1.00 (no decay)
- [D] 2.50

**Correct Answer**: **[B]**  
*Explanation*: $\xi = 0.75 - 0.80$ is the standard empirical value for Transformer speech models.

---

### Question 3:
What occurs during the linear warmup phase of a learning rate schedule?
- [A] The learning rate starts at zero and increases linearly to maximum $\eta_{\max}$ to prevent early gradient shocks from destabilizing weights.
- [B] All audio files are converted to mono.
- [C] The model emits only vowels.
- [D] The batch size doubles every step.

**Correct Answer**: **[A]**  
*Explanation*: Warmup stabilizes early optimization when gradients are noisy.

---

### Question 4:
Why is AdamW preferred over standard Adam?
- [A] AdamW runs without GPU support.
- [B] AdamW decouples weight decay from gradient updates, properly regularizing weights without distortion from running moment estimates.
- [C] AdamW has zero parameters.
- [D] AdamW eliminates the learning rate parameter.

**Correct Answer**: **[B]**  
*Explanation*: Decoupled weight decay correctly applies L2 regularization in adaptive optimizers.

---

### Question 5:
What percentage of total training steps is standardly allocated to warmup in EduCasheer training runs?
- [A] 1%
- [B] 10%
- [C] 50%
- [D] 90%

**Correct Answer**: **[B]**  
*Explanation*: 10% warmup provides adequate stabilization across diverse speech batches.
