# Lecture 20: Numerical Stability, Gradient Explosions, and CUDA OOM Recovery

---

## 1. The Anatomy of GPU Out-of-Memory (OOM) in Speech Models

In computer vision and NLP, input tensors typically have fixed rectangular dimensions (e.g., $224 \times 224$ pixels or 512 tokens).
In speech recognition, audio files have **wildly varying durations**, ranging from 0.8 seconds (12,800 samples = 40 frames) to 20.0 seconds (320,000 samples = 1,000 frames).

Because self-attention memory scales **quadratically with frame length** ($\mathcal{O}(T^2)$), a single 20-second utterance requires **625 times more attention memory** than a 0.8-second utterance!

If an un-bucketed dataloader groups several 18-second utterances together, GPU VRAM spikes instantly from 6 GB to 24 GB, triggering a catastrophic **CUDA Out-Of-Memory (OOM)** crash.

---

## 2. Dynamic Memory Fragmentation and PyTorch Caching Allocator

A frequent misconception among ML engineers is that `torch.cuda.OutOfMemoryError` means the model parameters and batch tensors exceed physical VRAM.
In reality, $> 70\%$ of OOM crashes in variable-length speech training are caused by **GPU Memory Fragmentation**:
1. When audio lengths fluctuate, PyTorch's caching allocator constantly allocates and frees variable-sized memory blocks.
2. Over thousands of training steps, memory becomes fragmented into non-contiguous chunks: plenty of total memory is free, but no single contiguous block is large enough to allocate the next intermediate attention tensor.

### Production Mitigations (`src/training/trainer.py`):
1. **Dynamic Bucket Batching**: Form batches from utterances of similar length, ensuring uniform tensor allocation.
2. **`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`**: Restricts the allocator from creating huge fragmented blocks.
3. **Graceful OOM Recovery**: Catch `torch.cuda.OutOfMemoryError`, immediately invoke `torch.cuda.empty_cache()`, halve the micro-batch size, and resume training without dropping the checkpoint!

---

## 3. Gradient Explosions and NaN Prevention

CTC loss can emit sudden infinite or NaN gradients if an utterance contains corrupted audio where acoustic frame length $T$ is shorter than label length $U$ ($T < U$).
- In standard PyTorch, a single NaN gradient instantly propagates into all optimizer running moments ($\mathbf{m}_t, \mathbf{v}_t$), permanently corrupting model weights into NaNs.
- **Production Defense**:
  1. `zero_infinity=True` in `nn.CTCLoss`.
  2. Gradient clipping norm ($\|\mathbf{g}\| \le 1.0$).
  3. Pre-step NaN checking: if `torch.isnan(loss)` or `torch.isinf(loss)` is detected, skip the step and quarantine the audio file.
