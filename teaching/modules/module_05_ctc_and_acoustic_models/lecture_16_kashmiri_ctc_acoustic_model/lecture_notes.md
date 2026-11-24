# Lecture 16: Building the Kashmiri CTC Acoustic Model

---

## 1. Architectural Anatomy: From Pretrained Backbone to CTC Emission

The EduCasheer acoustic model couples a massive self-supervised multilingual backbone (**XLS-R 300M**) with a dedicated task-specific linear projection head:

```
[ Raw Audio x (16 kHz) ]
           │
           ▼
[ 7-Layer CNN Feature Extractor ] ── (Output: 512-dim, 50 Hz frames)
           │
           ▼
[ Feature Projection & LayerNorm ] ── (Maps 512-dim to 1024-dim)
           │
           ▼
[ 24-Layer Transformer Encoder ] ── (1024-dim hidden states h_t)
           │
           ▼
[ Final Dropout (p = 0.1) ]
           │
           ▼
[ Linear CTC Projection Head ] ── (Linear: 1024 -> 47 classes)
           │
           ▼
[ Frame-Level Logits z_t in R^47 ]
```

---

## 2. Linear Head Design and Output Bias Initialization

The classification head is an affine transformation:
$$\mathbf{z}_t = \mathbf{W}_{\text{head}} \mathbf{h}_t + \mathbf{b}_{\text{head}}$$
where $\mathbf{W}_{\text{head}} \in \mathbb{R}^{47 \times 1024}$ and $\mathbf{b}_{\text{head}} \in \mathbb{R}^{47}$.

### Output Bias Initialization Strategy:
In CTC training, $> 90\%$ of frames correspond to the blank token $\epsilon$ (index 0).
- If $\mathbf{b}_{\text{head}}$ is initialized to uniform zeros, the initial Softmax assigns uniform probability $\frac{1}{47} \approx 0.021$ to all tokens.
- For an alignment of length $T=500$, initial loss is massive ($-\log(0.021^{500}) \approx 1930$), causing huge gradient shocks that can destroy pretrained Transformer weights.
- **Remedy**: Initializing the blank bias $b_0$ to a positive value (e.g., $b_0 = 3.0$) and all label biases to $0.0$ starts the network with an inductive bias toward blanks ($P(\epsilon) \approx 0.90$), stabilizing early optimization.

---

## 3. Freeze and Unfreeze Schedules: The Staged Fine-Tuning Recipe

Training all 317 million parameters simultaneously from epoch 1 on a low-resource dataset leads to **catastrophic forgetting**: the delicate low-level acoustic filters in the 7-layer CNN are ruined by noisy initial gradients.

EduCasheer enforces a **staged training schedule**:
1. **Stage 1 (Warmup Phase, Epochs 1–3)**:
   - **Feature Extractor**: Strictly frozen (`requires_grad = False`).
   - **Transformer Layers 1–16**: Frozen.
   - **Transformer Layers 17–24 + CTC Head**: Trainable with low learning rate ($3 \times 10^{-5}$).
2. **Stage 2 (Full Acoustic Adaptation, Epochs 4–20)**:
   - **Feature Extractor**: Remains frozen (preserves universal low-level edge filters).
   - **All 24 Transformer Layers**: Unfrozen with Layer-wise Learning Rate Decay (LLRD).
