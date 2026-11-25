# Lecture 17: The Complete PyTorch CTC Training Loop

---

## 1. Systems Engineering in Deep Acoustic Model Training

Training a 317-million-parameter acoustic model over 170+ hours of speech requires meticulous systems engineering. A naive training script will rapidly suffer from GPU out-of-memory (OOM) crashes, numerical underflows in mixed precision, corrupted padding losses, and non-reproducible metric checkpoints.

The canonical training loop (`src/training/trainer.py`) orchestrates five critical systems:
1. **Dynamic Batch Bucketing**: Minimizes padding waste.
2. **Automatic Mixed Precision (AMP)**: Accelerates matrix multiplication with `torch.cuda.amp.autocast` while preserving numerical stability via `GradScaler`.
3. **Gradient Accumulation**: Simulates massive effective batch sizes ($B=64$ or $128$) on modest hardware.
4. **Gradient Norm Clipping**: Constrains gradient norm ($\|\mathbf{g}\| \le 1.0$) to prevent exploding step updates.
5. **Atomic Checkpointing**: Atomically serializes model weights, optimizer state, scheduler state, and scalar evaluation history.

---

## 2. Automatic Mixed Precision (AMP) and GradScaler Mechanics

Standard FP32 floating point uses 32 bits (1 sign, 8 exponent, 23 mantissa). FP16 uses 16 bits (1 sign, 5 exponent, 10 mantissa).
- While FP16 doubles Tensor Core compute throughput and halves memory footprint, its dynamic range is restricted: numbers smaller than $2^{-24} \approx 5.96 \times 10^{-8}$ underflow to zero.
- In CTC loss backpropagation, gradients in deep Transformer layers frequently drop below $10^{-7}$. Under plain FP16, these gradients underflow to zero, causing training to freeze.

### The Solution: Dynamic Gradient Scaling
`torch.cuda.amp.GradScaler` multiplies the loss by a large scale factor $S$ (e.g., $S = 2^{15} = 32,768$) before backpropagation:

$$\mathbf{g}_{\text{scaled}} = \nabla_{\mathbf{W}} (S \cdot \mathcal{L})$$

This shifts small gradients into the representable range of FP16.
Before the optimizer updates the weights, the scaler unscales the gradients:

$$\mathbf{g} = \frac{\mathbf{g}_{\text{scaled}}}{S}$$

If infs or NaNs are detected, the optimizer step is skipped, and the scale factor $S$ is halved.

---

## 3. Effective Batch Size through Gradient Accumulation

CTC loss is notoriously noisy on small batch sizes ($B=4$ or $B=8$). Stable convergence requires an **effective batch size of 64 or 128 utterances**.
When GPU VRAM limits physical batch size to $B_{\text{phys}} = 8$:
- The loop accumulates gradients over $K = 8$ micro-steps:

$$\mathbf{g}_{\text{accum}} = \frac{1}{K} \sum_{k=1}^K \nabla_{\mathbf{W}} \mathcal{L}_k$$

- The optimizer step and gradient zeroing are triggered only every $K$ steps.
