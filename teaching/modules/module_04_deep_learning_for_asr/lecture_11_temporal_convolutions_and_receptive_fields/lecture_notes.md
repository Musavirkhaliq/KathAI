# Lecture 11: Temporal Convolutions and Receptive Fields in Speech Models

---

## 1. The Role of the 1D Convolutional Front-End

In raw waveform speech models like **wav2vec 2.0** and **XLS-R**, raw audio is not fed directly into the Transformer. At 16,000 samples per second, a 10-second audio clip contains 160,000 samples. Processing a 160,000-token sequence through quadratic $\mathcal{O}(T^2)$ self-attention would require:

$$\text{Memory} \propto 160,000^2 = 2.56 \times 10^{10} \text{ elements} \approx 102 \text{ GB GPU RAM per attention head}$$

To make self-attention computationally feasible, the audio must undergo **temporal downsampling**. This is executed by a **7-layer 1D Convolutional Feature Extractor** that compresses raw audio by a factor of **320**:

$$T_\text{frames} = \left\lfloor \frac{T_\text{samples}}{320} \right\rfloor$$

For 10 seconds of speech, the sequence length collapses from 160,000 to **500 frames** (a frame rate of 50 Hz, or 20 ms per frame), rendering Transformer attention fast and memory-efficient.

---

## 2. Mathematical Architecture of the XLS-R Feature Extractor

The feature encoder consists of 7 stacked 1D temporal convolutional blocks. Each block consists of:
`Conv1D -> Dropout -> LayerNorm (or GroupNorm) -> GELU`

### Exact Layer Specifications:

| Layer $i$ | Kernel Size $K_i$ | Stride $S_i$ | Input Channels | Output Channels | Output Temporal Stride |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Conv1** | 10 | 5 | 1 | 512 | 5 |
| **Conv2** | 3 | 2 | 512 | 512 | 10 |
| **Conv3** | 3 | 2 | 512 | 512 | 20 |
| **Conv4** | 3 | 2 | 512 | 512 | 40 |
| **Conv5** | 3 | 2 | 512 | 512 | 80 |
| **Conv6** | 2 | 2 | 512 | 512 | 160 |
| **Conv7** | 2 | 2 | 512 | 512 | **320** |

---

## 3. Rigorous Derivation of Total Downsampling and Receptive Field

### Downsampling Factor ($D$):
The cumulative temporal stride $D$ across the 7 layers is the product of individual strides:

$$D = \prod_{i=1}^7 S_i = 5 \times 2 \times 2 \times 2 \times 2 \times 2 \times 2 = 5 \times 64 = 320$$

At $f_s = 16,000 \text{ Hz}$, each output frame spans:

$$\Delta t = \frac{320 \text{ samples}}{16,000 \text{ samples/sec}} = 0.020 \text{ seconds} = 20.0 \text{ ms}$$

### Receptive Field Derivation:
The receptive field $R_l$ of an output neuron at layer $l$ measures the span of input audio samples that influence that neuron.
The recurrence relation for receptive field across convolutional layers is:

$$R_l = R_{l-1} + (K_l - 1) \cdot \prod_{i=1}^{l-1} S_i$$

with base condition $R_0 = 1$ and cumulative stride $J_{l-1} = \prod_{i=1}^{l-1} S_i$:

1. **Layer 1** ($K_1=10, S_1=5$):

$$R_1 = 1 + (10 - 1) \times 1 = 10 \text{ samples}$$

2. **Layer 2** ($K_2=3, S_2=2$):

$$R_2 = 10 + (3 - 1) \times 5 = 10 + 10 = 20 \text{ samples}$$

3. **Layer 3** ($K_3=3, S_3=2$):

$$R_3 = 20 + (3 - 1) \times 10 = 20 + 20 = 40 \text{ samples}$$

4. **Layer 4** ($K_4=3, S_4=2$):

$$R_4 = 40 + (3 - 1) \times 20 = 40 + 40 = 80 \text{ samples}$$

5. **Layer 5** ($K_5=3, S_5=2$):

$$R_5 = 80 + (3 - 1) \times 40 = 80 + 80 = 160 \text{ samples}$$

6. **Layer 6** ($K_6=2, S_6=2$):

$$R_6 = 160 + (2 - 1) \times 80 = 160 + 80 = 240 \text{ samples}$$

7. **Layer 7** ($K_7=2, S_7=2$):

$$R_7 = 240 + (2 - 1) \times 160 = 240 + 160 = \mathbf{400 \text{ samples}}$$

### Physical Acoustic Interpretation:
$$R_{\text{total}} = 400 \text{ samples} = \frac{400}{16,000} \text{ s} = 0.025 \text{ s} = \mathbf{25.0 \text{ ms}}$$

Each latent feature vector output by the 7-layer CNN has an **effective receptive field of 25.0 ms** and advances by **20.0 ms per frame**!
Notice the elegant acoustic alignment: this matches almost identically to the standard classical windowing parameters ($25 \text{ ms}$ window, $10-20 \text{ ms}$ shift) discovered through 50 years of speech science!
