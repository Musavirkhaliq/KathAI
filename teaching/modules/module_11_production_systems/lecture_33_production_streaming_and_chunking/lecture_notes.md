# Lecture 33: Production Streaming, Overlap-Add Chunking, and Real-Time Factor

---

## 1. The Challenge of Long-Form and Real-Time Audio

In production deployment, an ASR service does not receive pre-trimmed 5-second audio snippets. It receives:
1. **Long-Form Audio**: Podcasts, news broadcasts, legal proceedings, and sermons lasting 15 minutes to 2 hours.
2. **Streaming Live Audio**: Real-time microphone input where transcription must be returned with sub-second latency.

Attempting to process a 30-minute audio file in a single forward pass causes immediate failure:
- **Memory Collapse**: At 16 kHz, 30 minutes is $28.8 \times 10^6$ samples. In self-attention, $(90,000)^2$ frames would require **32 Gigabytes of GPU RAM** for a single attention matrix!
- **Positional Drift**: The convolutional relative positional encodings in XLS-R were trained on spans of $\le 20 \text{ seconds}$; feeding 30 minutes causes attention distributions to disintegrate.

---

## 2. Method C2: Overlap-Add Sliced Chunking Architecture

To process arbitrary-duration audio with bounded GPU memory and zero boundary deletion errors, EduCasheer engineered **Method C2 Chunking** (`src/inference/chunking.py`):

```
Audio Timeline:
[======== Chunk 1 (8.0s) ========]
                    [======== Chunk 2 (8.0s) ========]
                                        [======== Chunk 3 (8.0s) ========]
<--- 7.0s Core ---><-- 1.0s Overlap --><--- 7.0s Core --->
```

### Parameters of Method C2:
- **Window Length ($W$)**: **8.0 seconds** (128,000 samples = 250 frames).
- **Overlap Duration ($O$)**: **1.0 second** (16,000 samples = 31 frames).
- **Hop Size ($H$)**: $W - O = \mathbf{7.0 \text{ seconds}}$ (112,000 samples).

### Boundary Artifact Suppression:
Naive chunking without overlap truncates words spoken across the boundary, causing catastrophic word mutilation.
Method C2 applies **logit cross-fading and prefix stitching**:
1. Adjacent chunks overlap by 1.0 second.
2. In the overlap zone, frame-level CTC logits are blended with a linear cross-fade window:

$$\mathbf{z}_{\text{blended}}(t) = (1 - w(t)) \mathbf{z}_{\text{chunk } k}(t) + w(t) \mathbf{z}_{\text{chunk } k+1}(t)$$

3. The stitched logit tensor is decoded in a single continuous beam search pass, completely eliminating boundary artifacts.

---

## 3. Real-Time Factor (RTF) Performance

The production streaming service (`scripts/asr_service.py`) operates as a background daemon evaluated on **Real-Time Factor (RTF)**:

$$\text{RTF} = \frac{\text{Wall-Clock Processing Time (seconds)}}{\text{Audio Duration (seconds)}}$$

- An $\text{RTF} < 1.0$ is required for real-time streaming.
- On a single NVIDIA RTX 3090 GPU with FP16 Tensor Cores, EduCasheer Method C2 achieves:

$$\text{RTF} \approx \mathbf{0.082}$$

  Processing 10 seconds of audio takes only **0.82 seconds**, enabling real-time streaming with comfortable latency headroom.
