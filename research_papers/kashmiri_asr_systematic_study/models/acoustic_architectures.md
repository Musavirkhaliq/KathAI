# Acoustic Model Architectures

This document details the neural architecture of the foundation acoustic models deployed in the EduCasheer Kashmiri ASR systematic study.

---

## 1. Primary Acoustic Architecture: XLS-R 300M + Character CTC

All primary acoustic models are built upon `facebook/wav2vec2-xls-r-300m`, a self-supervised cross-lingual speech representation model pre-trained on 436,000 hours of uncurated speech across 128 languages.

### 1.1 Architectural Specifications
- **Base Encoder**: 7-layer temporal convolutional feature extractor.
  - Channels: 512 per layer.
  - Strides: $(5, 2, 2, 2, 2, 2, 2)$, yielding a cumulative temporal downsampling factor of:
    $$\text{Stride} = 5 \times 2^6 = 320$$
  - At 16,000 Hz, this yields feature representations at a frame rate of 50 Hz (one acoustic frame every 20 ms, with an effective receptive field of 25 ms).
- **Context Transformer Network**:
  - Layers: 24 Transformer encoder blocks.
  - Model Dimension ($d_{\text{model}}$): 1,024.
  - Feed-Forward Inner Dimension ($d_{\text{ffn}}$): 4,096.
  - Attention Heads: 16 heads ($d_{\text{head}} = 64$).
  - Positional Embedding: Relative convolutional positional encoding with kernel size 128 and 16 groups.
  - Normalization: Post-LayerNorm with layer dropout $p = 0.1$.
- **Prediction Head**:
  - Linear classification projection: $\mathbb{R}^{1024} \rightarrow \mathbb{R}^{159}$.
  - Softmax output modeling the 159-token Kashmiri character vocabulary (including blank token `[pad]` and whitespace `|`).
- **Total Parameter Count**:
  $$\text{Parameters}_{\text{Total}} = 315,603,745 \text{ parameters } (\approx 315.6\text{M})$$
  - Convolutional Feature Extractor: 4,180,992 parameters.
  - Transformer Context Network: 311,259,648 parameters.
  - Character CTC Head: 163,105 parameters.

```
Raw Audio Waveform (16,000 Hz, 1-channel PCM)
   │
   ▼
[7-Layer Temporal Convolutional Feature Extractor] (50 Hz frames, dim=512)
   │
   ▼
[Linear Feature Projection + LayerNorm] (dim=1024)
   │
   ▼
[Convolutional Positional Encoding] (kernel=128, groups=16)
   │
   ▼
[24-Layer Transformer Encoder Stack] (dim=1024, heads=16, ffn=4096)
   │
   ▼
[Linear Classification Head] (1024 -> 159 classes)
   │
   ▼
Log-Softmax Emission Probabilities P(π_t | x) (50 frames / sec)
```

---

## 2. Character Vocabulary Construction

The 159-token inventory was constructed directly from empirical corpus audits to resolve Kashmiri orthographic complexity with zero out-of-vocabulary instances:
- **Base Persian-Arabic Letters**: Alef (`ا`), Be (`ب`), Pe (`پ`), Te (`ت`), Te Retroflex (`ٹ`), Se (`ث`), Jeem (`ج`), Che (`چ`), He (`ح`), Khe (`خ`), Dal (`د`), Dal Retroflex (`ڈ`), Zal (`ذ`), Re (`ر`), Re Retroflex (`ڑ`), Ze (`ز`), Zhe (`ژ`), Seen (`س`), Sheen (`ش`), Saad (`ص`), Zaad (`ض`), Toye (`ط`), Zoye (`ظ`), Ain (`ع`), Ghain (`غ`), Fe (`ف`), Qaaf (`ق`), Keheh (`ک`), Gaaf (`گ`), Laam (`ل`), Meem (`م`), Noon (`ن`), Noon Ghunna (`ں`), Waaw (`و`), Heh Goal (`ہ`), Doachashmee Heh (`ھ`), Hamza (`ء`), Chhoti Yeh (`ی`), Bari Yeh (`ے`).
- **Contrastive Central Vowel Graphemes**:
  - `ٲ` (`U+0672` — Alef with Wavy Hamza Above)
  - `ؠ` (`U+0620` — Kashmiri Kashmiri Yeh with small V)
  - `ۆ` (`U+06C6` — Kashmiri Waaw with V)
  - `ۄ` (`U+06C4` — Kashmiri Waaw with circle)
  - `ێ` (`U+06CE` — Kashmiri Yeh with tail)
  - `ۍ` (`U+06CD` — Kashmiri Yeh with ring)
  - `ؽ` (`U+06FD` — Kashmiri Yeh with inverted V)
- **Special CTC Tokens**: `[pad]` (index 0, CTC blank), `|` (word boundary / whitespace), `[unk]` (unknown character fallback).
