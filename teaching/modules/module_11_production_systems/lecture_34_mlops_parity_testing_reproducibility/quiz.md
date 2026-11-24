# Quiz: Lecture 34 (MLOps and Golden Logit Parity Testing)

---

### Question 1:
What is a 'golden logit parity test'?
- [A] A test that pays gold to successful engineers.
- [B] A regression test asserting that a refactored model emits frame-level logits matching a verified reference tensor within tight floating-point tolerance ($\Delta < 10^{-5}$).
- [C] A test of microphone quality.
- [D] An audio volume calibration test.

**Correct Answer**: **[B]**  
*Explanation*: Parity tests guarantee that code refactoring does not alter model predictions.

---

### Question 2:
What tolerance threshold $\tau$ is standard for FP32 logit parity testing?
- [A] $1.0$
- [B] $10^{-5}$
- [C] $10^{-20}$
- [D] Zero tolerance

**Correct Answer**: **[B]**  
*Explanation*: $\tau = 10^{-5}$ safely absorbs floating-point non-associativity across hardware while preserving decoding fidelity.

---

### Question 3:
Why must regression test suites be run in Continuous Integration (CI)?
- [A] To increase GPU power consumption.
- [B] To automatically catch bugs, normalizer regressions, and performance drops before code is merged into production branches.
- [C] Because PyTorch requires it.
- [D] To slow down development.

**Correct Answer**: **[B]**  
*Explanation*: CI automated testing ensures production stability.

---

### Question 4:
What stage of the EduCasheer project certified code parity between historical scripts and the modular codebase?
- [A] Stage 01
- [B] Stage 12
- [C] Stage 27
- [D] Stage 19

**Correct Answer**: **[C]**  
*Explanation*: Stage 27 certified complete inference parity and model reproducibility.

---

### Question 5:
What causes slight numerical drift in GPU floating-point calculations across runs?
- [A] Room temperature fluctuations.
- [B] Non-associativity of floating-point addition in parallel thread reductions.
- [C] Python garbage collection.
- [D] Unicode encoding.

**Correct Answer**: **[B]**  
*Explanation*: Parallel atomic operations add floating point numbers in non-deterministic order.
