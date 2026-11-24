# Quiz: Lecture 15 (CTC Loss Mathematics)

---

### Question 1:
What is the length of the extended target sequence $\mathbf{l}'$ for a target transcription of length $U$?
- [A] $U$
- [B] $U + 1$
- [C] $2U + 1$
- [D] $U^2$

**Correct Answer**: **[C]**  
*Explanation*: Extended sequence inserts a blank before, between, and after every symbol: $U + (U + 1) = 2U + 1$.

---

### Question 2:
What is the analytical gradient of the CTC loss with respect to unnormalized logit $z_k^t$?
- [A] $y_k^t - \gamma_t(k)$
- [B] $(y_k^t)^2$
- [C] $\log(y_k^t)$
- [D] $y_k^t \cdot \gamma_t(k)$

**Correct Answer**: **[A]**  
*Explanation*: The gradient is simply the difference between model output probability $y_k^t$ and alignment posterior $\gamma_t(k)$.

---

### Question 3:
Under what condition can state $s$ receive a transition directly from state $s-2$ in the forward recursion?
- [A] When state $s$ is the blank token.
- [B] When state $s$ is a non-blank token that is distinct from state $s-2$ ($\mathbf{l}'_s \neq \epsilon$ and $\mathbf{l}'_s \neq \mathbf{l}'_{s-2}$).
- [C] When $t > 50$.
- [D] Under all conditions.

**Correct Answer**: **[B]**  
*Explanation*: Blank-skipping is permissible only between distinct non-blank symbols.

---

### Question 4:
Why does CTC training produce 'peaky' emission spikes surrounded by blank tokens?
- [A] Because of hardware bugs in GPUs.
- [B] Because blanks can participate in almost all lattice paths, concentrating probability mass into blanks during transition frames and reserving spikes for high-confidence frames.
- [C] Because characters are normalized with NFKC.
- [D] To save battery power.

**Correct Answer**: **[B]**  
*Explanation*: Blanks provide a low-risk fallback for ambiguous frames, concentrating phonetic evidence into narrow spikes.

---

### Question 5:
What occurs if input acoustic frame length $T$ is less than target text length $U$ ($T < U$)?
- [A] Training proceeds normally.
- [B] CTC loss is mathematically undefined (returns $+\infty$ or zero-infinity clamp) because no valid monotonic alignment can fit $U$ characters into fewer than $U$ frames.
- [C] Text characters are automatically duplicated.
- [D] The audio sample rate doubles.

**Correct Answer**: **[B]**  
*Explanation*: CTC requires $T \ge U$; otherwise the preimage $\mathcal{B}^{-1}(\mathbf{l})$ is empty.
