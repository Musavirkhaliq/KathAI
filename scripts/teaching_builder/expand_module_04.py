#!/usr/bin/env python3
"""
expand_module_04.py
Expands Lectures 10, 11, 12, and 13 into exhaustive, textbook-grade educational materials:
- Lecture 10: Deep Neural Networks and Temporal Backpropagation
- Lecture 11: Temporal Convolutions and Receptive Fields
- Lecture 12: Multi-Head Self-Attention and Transformers for Speech
- Lecture 13: Self-Supervised Speech Representations (wav2vec 2.0 and XLS-R)
"""

from pathlib import Path

BASE_DIR = Path("/home/musa/Kashmiri Project/teaching/modules/module_04_deep_learning_for_asr")

# ==============================================================================
# LECTURE 10: DEEP NEURAL NETWORKS AND TEMPORAL BACKPROPAGATION
# ==============================================================================
def expand_lecture_10():
    ldir = BASE_DIR / "lecture_10_neural_networks_and_backprop"
    ldir.mkdir(parents=True, exist_ok=True)

    notes = r"""# Lecture 10: Deep Neural Networks and Temporal Backpropagation

---

## 1. The Fundamental Architecture of Deep Neural Acoustic Models

In modern speech recognition, deep neural networks parameterize non-linear vector functions mapping acoustic sequences to linguistic representations. A deep feedforward layer applies an affine transformation followed by an element-wise non-linear activation:

$$\mathbf{h}^{(l)} = \sigma\left( \mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)} \right)$$

where $\mathbf{W}^{(l)} \in \mathbb{R}^{d_l \times d_{l-1}}$ is the weight matrix, $\mathbf{b}^{(l)} \in \mathbb{R}^{d_l}$ is the bias vector, and $\sigma(\cdot)$ is an activation function (GELU, ReLU, or Swish).

In speech models like XLS-R, the standard activation is the **Gaussian Error Linear Unit (GELU)**:
$$\text{GELU}(x) = x \Phi(x) = x P(X \le x) \approx 0.5 x \left( 1 + \tanh\left( \sqrt{\frac{2}{\pi}} \left( x + 0.044715 x^3 \right) \right) \right)$$
GELU introduces smooth non-monotonicity and probabilistic gating, outperforming hard ReLU in deep acoustic transformers.

---

## 2. Temporal Backpropagation and Jacobian-Vector Products

Training deep models over sequences of length $T$ involves computing the gradient of a scalar sequence loss $\mathcal{L}$ with respect to all parameter matrices $\mathbf{W}^{(l)}$.

By the multivariate chain rule, the gradient at layer $l$ satisfies:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(l-1)}} = \left( \frac{\partial \mathbf{h}^{(l)}}{\partial \mathbf{h}^{(l-1)}} \right)^T \frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(l)}} = \left( \mathbf{W}^{(l)} \right)^T \left( \frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(l)}} \odot \sigma'(\mathbf{z}^{(l)}) \right)$$

where $\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}$, and $\odot$ represents the Hadamard (element-wise) product.

For a sequence of $L$ layers:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(0)}} = \left[ \prod_{l=1}^L \left( \mathbf{W}^{(l)} \right)^T \text{diag}(\sigma'(\mathbf{z}^{(l)})) \right] \frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(L)}}$$

### The Vanishing / Exploding Gradient Dilemma:
Let $\lambda_{\max}$ be the maximum singular value of the weight matrices.
- If $\lambda_{\max} < 1$, gradients decay exponentially with depth $L$: $\|\frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(0)}}\| \to 0$ (**Vanishing Gradient**).
- If $\lambda_{\max} > 1$, gradients grow exponentially: $\|\frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(0)}}\| \to \infty$ (**Exploding Gradient**).

---

## 3. Normalization Techniques: LayerNorm vs. BatchNorm in Speech

Speech sequences exhibit variable lengths within every batch (e.g., 2.5s to 15.0s).
- **Batch Normalization (BatchNorm)**: Computes mean and variance across the batch dimension. In speech, because sequences are zero-padded to batch max length, padding frames severely distort running mean $\mu_B$ and running variance $\sigma_B^2$, leading to training instability.
- **Layer Normalization (LayerNorm)**: Computes mean and variance independently for each time frame across the feature channel dimension $D$:
  $$\mu_t = \frac{1}{D} \sum_{i=1}^D x_{t, i}, \quad \sigma_t^2 = \frac{1}{D} \sum_{i=1}^D (x_{t, i} - \mu_t)^2$$
  $$\hat{\mathbf{x}}_t = \frac{\mathbf{x}_t - \mu_t}{\sqrt{\sigma_t^2 + \epsilon}} \odot \boldsymbol{\gamma} + \boldsymbol{\beta}$$
  LayerNorm is completely independent of batch size and padding tokens, making it the universal standard for speech transformers.
"""
    (ldir / "lecture_notes.md").write_text(notes, encoding="utf-8")

    script = r"""# Instructor Delivery Script: Lecture 10 (Deep Neural Networks and Temporal Backpropagation)

## Timetable (90 Minutes)
```
00:00 - 00:20 | Deep Acoustic Modeling: Affine Maps and Non-Linearities (GELU vs. ReLU)
00:20 - 00:45 | Mathematical Proof: The Chain Rule Product and Gradient Singular Values
00:45 - 01:10 | Why BatchNorm Fails in Speech: The Padding Contamination Trap
01:10 - 01:25 | LayerNorm Mechanics and Numerical Stability
01:25 - 01:30 | Synthesis & Transition to Temporal Convolutions
```

---

## Instructor Notes & Dialogue

### 00:20 - 00:45 | Gradient Flow on the Board
- Write the product on the Center Board:
  $$\prod_{l=1}^{24} \mathbf{W}^{(l) T}$$
- Look at the students:
  > "XLS-R has 24 Transformer layers. If each layer scales your gradient by just 1.1, then 1.1 to the power of 24 is 9.85.
  > If each layer scales your gradient by 0.9, then 0.9 to the power of 24 is 0.079.
  > Either your gradients explode into NaNs, or your bottom layers freeze and learn nothing.
  > How does modern deep learning train a 24-layer model on speech?
  > Two architectural miracles: **Residual Connections** ($\mathbf{x} + F(\mathbf{x})$) and **Layer Normalization**.
  > Residual connections create an unhindered gradient highway where $\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial F} + \mathbf{I}$, guaranteeing that the gradient can never decay to zero."
"""
    (ldir / "instructor_script.md").write_text(script, encoding="utf-8")

    math_doc = r"""# Mathematical Foundations: Gradient Flow and Residual Stability

---

## 1. Residual Connection Gradient Preservation

Consider a residual building block:
$$\mathbf{h}^{(l)} = \mathbf{h}^{(l-1)} + \mathcal{F}(\mathbf{h}^{(l-1)}, \mathcal{W}^{(l)})$$

The Jacobian with respect to the input activation is:
$$\frac{\partial \mathbf{h}^{(l)}}{\partial \mathbf{h}^{(l-1)}} = \mathbf{I} + \frac{\partial \mathcal{F}}{\partial \mathbf{h}^{(l-1)}}$$

For any deep network spanning layer $m$ to layer $n$ ($n > m$):
$$\mathbf{h}^{(n)} = \mathbf{h}^{(m)} + \sum_{i=m}^{n-1} \mathcal{F}(\mathbf{h}^{(i)}, \mathcal{W}^{(i+1)})$$

The gradient with respect to $\mathbf{h}^{(m)}$ is:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(m)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(n)}} \frac{\partial \mathbf{h}^{(n)}}{\partial \mathbf{h}^{(m)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(n)}} \left( \mathbf{I} + \frac{\partial}{\partial \mathbf{h}^{(m)}} \sum_{i=m}^{n-1} \mathcal{F} \right)$$

Because of the identity matrix term $\mathbf{I}$, the gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(m)}}$ does not vanish even if the residual branch derivatives $\frac{\partial \mathcal{F}}{\partial \mathbf{h}^{(i)}}$ are close to zero.
"""
    (ldir / "mathematical_foundations.md").write_text(math_doc, encoding="utf-8")

    code_doc = r"""# Code Walkthrough: LayerNorm and GELU Implementation

---

## 1. Pure PyTorch Implementation of LayerNorm

```python
import torch
import torch.nn as nn

class CustomLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, time, channels]
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.gamma + self.beta
```
"""
    (ldir / "code_walkthrough.md").write_text(code_doc, encoding="utf-8")

    lab_doc = r"""# Practical Lab: Gradient Flow and Activation Analysis

---

## Lab Objectives
1. Implement forward and backward passes of a 10-layer MLP in pure PyTorch.
2. Compare gradient norms across layers with and without residual skip connections.
3. Observe gradient vanishing when using sigmoid activations vs. stable flow with GELU + LayerNorm.

---

## Lab Script: `lab_gradient_flow.py`

```python
import torch
import torch.nn as nn

def run_gradient_experiment():
    depth = 12
    dim = 256
    batch_size = 16
    
    # Model without residuals (Plain MLP)
    plain_layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(depth)])
    
    x = torch.randn(batch_size, dim, requires_grad=True)
    h = x
    for layer in plain_layers:
        h = torch.tanh(layer(h))
        
    loss = h.sum()
    loss.backward()
    
    first_grad = plain_layers[0].weight.grad.norm().item()
    last_grad = plain_layers[-1].weight.grad.norm().item()
    print(f"Plain MLP - First Layer Grad Norm: {first_grad:.6f} | Last Layer: {last_grad:.6f}")
    print(f"Gradient Decay Ratio: {first_grad / last_grad:.6f}")
    assert first_grad / last_grad < 0.1, "Expected gradient vanishing in deep plain tanh network"
    print("SUCCESS: Gradient decay demonstrated!")

if __name__ == "__main__":
    run_gradient_experiment()
```
"""
    (ldir / "practical_lab.md").write_text(lab_doc, encoding="utf-8")

    ex_doc = r"""# Exercises: Lecture 10 (Deep Neural Networks and Backpropagation)

---

## Exercise 1: GELU Derivative
1. Write the mathematical definition of GELU using the standard Gaussian CDF $\Phi(x)$.
2. Compute the first derivative $\frac{d}{dx}\text{GELU}(x)$ with respect to $x$.
3. Evaluate the derivative at $x = 0$ and $x = -2.0$. Explain why GELU does not suffer from dead neurons like ReLU.

---

## Exercise 2: LayerNorm Invariance Proof
Prove that LayerNorm is invariant to:
1. Scaling of the input vector by positive scalar $\alpha > 0$ ($\text{LN}(\alpha \mathbf{x}) = \text{LN}(\mathbf{x})$).
2. Constant shift of the input vector by scalar $c \in \mathbb{R}$ ($\text{LN}(\mathbf{x} + c \mathbf{1}) = \text{LN}(\mathbf{x})$).
"""
    (ldir / "exercises.md").write_text(ex_doc, encoding="utf-8")

    sol_doc = r"""# Solutions: Lecture 10 (Deep Neural Networks and Backpropagation)

---

## Solution to Exercise 1
1. Definition: $\text{GELU}(x) = x \Phi(x)$, where $\Phi(x) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^x e^{-t^2/2} dt$.
2. Derivative:
   $$\frac{d}{dx}\text{GELU}(x) = \Phi(x) + x \Phi'(x) = \Phi(x) + x \frac{1}{\sqrt{2\pi}} e^{-x^2 / 2}$$
3. Values:
   - At $x = 0$: $\Phi(0) = 0.5$, $\Phi'(0) = \frac{1}{\sqrt{2\pi}}$, so $\frac{d}{dx}\text{GELU}(0) = 0.5 + 0 = \mathbf{0.5}$.
   - At $x = -2.0$: $\Phi(-2.0) \approx 0.0228$, $\Phi'(-2.0) = \frac{1}{\sqrt{2\pi}} e^{-2} \approx 0.0540$.
     $$\frac{d}{dx}\text{GELU}(-2.0) = 0.0228 + (-2.0)(0.0540) = 0.0228 - 0.108 = \mathbf{-0.0852}$$
   Unlike ReLU which outputs strictly 0 gradient for all negative inputs (causing dead neurons), GELU has non-zero gradients in the negative region, allowing recovery from suboptimal initialization.

---

## Solution to Exercise 2
1. For $\mathbf{y} = \alpha \mathbf{x}$ ($\alpha > 0$):
   $$\mu_y = \alpha \mu_x, \quad \sigma_y^2 = \alpha^2 \sigma_x^2 \implies \sigma_y = \alpha \sigma_x$$
   $$\hat{y}_i = \frac{\alpha x_i - \alpha \mu_x}{\alpha \sigma_x} = \frac{x_i - \mu_x}{\sigma_x} = \hat{x}_i \quad \blacksquare$$
2. For $\mathbf{y} = \mathbf{x} + c \mathbf{1}$:
   $$\mu_y = \mu_x + c, \quad \sigma_y^2 = \frac{1}{D} \sum (x_i + c - (\mu_x + c))^2 = \sigma_x^2$$
   $$\hat{y}_i = \frac{(x_i + c) - (\mu_x + c)}{\sigma_x} = \frac{x_i - \mu_x}{\sigma_x} = \hat{x}_i \quad \blacksquare$$
"""
    (ldir / "solutions.md").write_text(sol_doc, encoding="utf-8")

    quiz_doc = r"""# Quiz: Lecture 10 (Deep Neural Networks and Backpropagation)

---

### Question 1:
Why is Layer Normalization preferred over Batch Normalization in deep speech recognition architectures?
- [A] LayerNorm requires fewer parameters than BatchNorm.
- [B] BatchNorm suffers when batches contain variable-length audio padded with zeros, whereas LayerNorm normalizes across channels independently for each time frame.
- [C] LayerNorm only runs on AMD GPUs.
- [D] BatchNorm cannot be used with CTC loss.

**Correct Answer**: **[B]**  
*Explanation*: Zero-padding in variable-length speech batches corrupts batch-level statistics in BatchNorm. LayerNorm operates along feature dimensions, avoiding padding pollution.

---

### Question 2:
What mathematical mechanism prevents gradients from vanishing in deep 24-layer Transformer backbones?
- [A] Dropout
- [B] Residual skip connections ($\mathbf{x} + \mathcal{F}(\mathbf{x})$)
- [C] Greedy decoding
- [D] Sinc interpolation

**Correct Answer**: **[B]**  
*Explanation*: Skip connections ensure an identity gradient term $\mathbf{I}$ flows backward across layers unattenuated.

---

### Question 3:
What is the mathematical definition of the GELU activation function?
- [A] $\max(0, x)$
- [B] $x \Phi(x)$ where $\Phi(x)$ is the standard Gaussian cumulative distribution function
- [C] $\frac{1}{1 + e^{-x}}$
- [D] $\tanh(x)$

**Correct Answer**: **[B]**  
*Explanation*: GELU weights input $x$ by the probability that a standard normal variable is less than $x$.

---

### Question 4:
What happens to parameter gradients when the maximum singular value of the weight matrices is consistently $> 1.0$ across 24 layers without normalization?
- [A] Gradients vanish to zero.
- [B] Gradients explode to infinity, causing NaN numerical overflows.
- [C] The model converts to a language model.
- [D] The sampling rate increases.

**Correct Answer**: **[B]**  
*Explanation*: Repeated multiplication by matrices with singular values $> 1$ leads to exponential gradient explosion.

---

### Question 5:
Why does GELU not suffer from the 'dead neuron' problem common in standard ReLU?
- [A] Because it is computed on CPU.
- [B] Because its derivative is non-zero in small negative input regimes, allowing neurons to receive gradient updates and recover.
- [C] Because it always outputs positive values.
- [D] Because it uses 64-bit precision.

**Correct Answer**: **[B]**  
*Explanation*: GELU has a smooth dip with non-zero negative gradients, allowing dead units to update.
"""
    (ldir / "quiz.md").write_text(quiz_doc, encoding="utf-8")

    res_doc = r"""# Research Connections: Lecture 10

---

## 1. Repository Grounding
- **`src/models/wav2vec2.py`**: Implementation of LayerNorm and GELU inside the 24-layer Transformer blocks.
- **`configs/acoustic/wav2vec2_kashmiri.yaml`**: Specifies `layer_norm_eps: 1e-5`, `hidden_act: "gelu"`.
"""
    (ldir / "research_connections.md").write_text(res_doc, encoding="utf-8")

    refs_doc = r"""# Annotated References: Lecture 10

---

1. **Hendrycks, D., & Gimpel, K. (2016).**  
   *Gaussian error linear units (GELUs).*  
   *arXiv preprint arXiv:1606.08415*.  
   - **Annotation**: Introduces the GELU activation function used in BERT, GPT, and wav2vec 2.0.

2. **Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016).**  
   *Layer normalization.*  
   *arXiv preprint arXiv:1607.06450*.  
   - **Annotation**: Formulates LayerNorm as a batch-independent normalizer for recurrent and transformer networks.
"""
    (ldir / "references.md").write_text(refs_doc, encoding="utf-8")
    print("Expanded Lecture 10 successfully!")

# ==============================================================================
# LECTURE 11: TEMPORAL CONVOLUTIONS AND RECEPTIVE FIELDS
# ==============================================================================
def expand_lecture_11():
    ldir = BASE_DIR / "lecture_11_temporal_convolutions_and_receptive_fields"
    ldir.mkdir(parents=True, exist_ok=True)

    notes = r"""# Lecture 11: Temporal Convolutions and Receptive Fields in Speech Models

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
"""
    (ldir / "lecture_notes.md").write_text(notes, encoding="utf-8")

    script = r"""# Instructor Delivery Script: Lecture 11 (Temporal Convolutions and Receptive Fields)

## Timetable (90 Minutes)
```
00:00 - 00:20 | The Memory Wall: Why 160,000 Audio Samples Break Self-Attention
00:20 - 00:50 | Step-by-Step Derivation of the Receptive Field (R = 400 Samples = 25 ms)
00:50 - 01:15 | Inspecting Learned Convolutions: The Emergence of Biological Filterbanks
01:15 - 01:25 | Code Inspection: XLS-R ConvFeatureExtractionModel
01:25 - 01:30 | Exercises & Takeaways
```

---

## Instructor Notes & Dialogue

### 00:20 - 00:50 | Receptive Field Derivation on the Board
- Walk through the recursive calculation: $R_l = R_{l-1} + (K_l - 1) \cdot J_{l-1}$.
- Calculate each layer on the board with student participation:
  - $R_1 = 10$
  - $R_2 = 20$
  - $R_3 = 40$
  - $R_4 = 80$
  - $R_5 = 160$
  - $R_6 = 240$
  - $R_7 = 400$
- Highlight the number 400:
  > "Divide 400 by 16,000. What do you get? Exactly 25 milliseconds.
  > The Facebook AI Research engineers didn't pick random numbers for kernel sizes and strides.
  > They designed the convolutional stack so that every single latent frame captures exactly one 25 ms quasi-stationary phonetic window!"
"""
    (ldir / "instructor_script.md").write_text(script, encoding="utf-8")

    math_doc = r"""# Mathematical Foundations: Receptive Field and Feature Length Slicing

---

## 1. General Recursive Receptive Field Formula

Let a cascade of $L$ 1D convolutional layers have kernel sizes $K_l$, strides $S_l$, and dilations $D_l$.
The effective kernel size $K_l'$ considering dilation is:
$$K_l' = K_l + (K_l - 1)(D_l - 1) = 1 + D_l(K_l - 1)$$

The cumulative stride $J_l$ after layer $l$ is:
$$J_l = \prod_{i=1}^l S_i, \quad J_0 = 1$$

The total receptive field $R_l$ satisfies the difference equation:
$$R_l = R_{l-1} + (K_l' - 1) J_{l-1}, \quad R_0 = 1$$

Expanding the telescoping sum:
$$R_L = 1 + \sum_{l=1}^L (K_l' - 1) \prod_{i=1}^{l-1} S_i$$

---

## 2. Output Sequence Length Equation

Given input length $T_\text{in}$, the output sequence length after $L$ convolutional layers with zero padding is:
$$T_{l} = \left\lfloor \frac{T_{l-1} - K_l}{S_l} \right\rfloor + 1$$
"""
    (ldir / "mathematical_foundations.md").write_text(math_doc, encoding="utf-8")

    code_doc = r"""# Code Walkthrough: XLS-R Feature Extractor Inspection

---

## 1. Inspecting the Feature Extractor (`src/models/wav2vec2.py`)

```python
import torch
import torch.nn as nn

class ConvFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Strides: (5, 2, 2, 2, 2, 2, 2) -> Total downsampling = 320
        # Kernels: (10, 3, 3, 3, 3, 2, 2)
        conv_layers = []
        in_c = 1
        configs = [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2)
        ]
        
        for out_c, k, s in configs:
            conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_c, out_c, kernel_size=k, stride=s, bias=False),
                    nn.Dropout(p=0.0),
                    nn.GroupNorm(num_groups=512, num_channels=out_c, affine=True),
                    nn.GELU()
                )
            )
            in_c = out_c
            
        self.conv_layers = nn.ModuleList(conv_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, 1, time_samples]
        for layer in self.conv_layers:
            x = layer(x)
        return x  # Shape: [batch, 512, time_frames]
```
"""
    (ldir / "code_walkthrough.md").write_text(code_doc, encoding="utf-8")

    lab_doc = r"""# Practical Lab: Receptive Field Calculation and Verification

---

## Lab Objectives
1. Implement automated receptive field tracking in PyTorch.
2. Verify empirical receptive field by computing gradient impulse responses from output neurons back to the input audio buffer.
3. Validate output shape assertions for arbitrary audio lengths.

---

## Lab Script: `lab_receptive_field.py`

```python
import torch
import torch.nn as nn

def run_rf_lab():
    # Build a 2-layer toy convolutional stack
    # Layer 1: K=10, S=5
    # Layer 2: K=3, S=2
    # Theoretical RF: R1 = 10, R2 = 10 + (3-1)*5 = 20 samples
    conv1 = nn.Conv1d(1, 16, kernel_size=10, stride=5, bias=False)
    conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=2, bias=False)
    
    # Initialize all weights to 1.0 to trace connections
    nn.init.constant_(conv1.weight, 1.0)
    nn.init.constant_(conv2.weight, 1.0)
    
    x = torch.zeros(1, 1, 100, requires_grad=True)
    out = conv2(conv1(x))
    
    # Backprop from first output frame center neuron
    out[0, 0, 0].backward()
    
    # Count non-zero gradient positions in input x
    grad = x.grad[0, 0]
    active_samples = torch.sum(grad != 0).item()
    print(f"Empirically measured active input samples: {active_samples}")
    print("Theoretical RF = 20 samples")
    assert active_samples == 20, f"Expected 20, got {active_samples}"
    print("SUCCESS: Empirical receptive field perfectly matches theory!")

if __name__ == "__main__":
    run_rf_lab()
```
"""
    (ldir / "practical_lab.md").write_text(lab_doc, encoding="utf-8")

    ex_doc = r"""# Exercises: Lecture 11 (Temporal Convolutions and Receptive Fields)

---

## Exercise 1: Custom Strided Architecture
Suppose you design an alternative ASR convolutional front-end with 5 layers:
- Layer 1: $K=7, S=4$
- Layer 2: $K=5, S=2$
- Layer 3: $K=5, S=2$
- Layer 4: $K=3, S=2$
- Layer 5: $K=3, S=1$
1. Calculate the total temporal downsampling factor $D$.
2. Derive the total receptive field $R_5$ in samples.
3. At 16 kHz, what is the frame shift (ms) and receptive field duration (ms)?

---

## Exercise 2: Frame Count Calculation
For the 7-layer XLS-R feature extractor:
1. Write the explicit formula for $T_\text{frames}$ as a function of $T_\text{samples}$.
2. Calculate the exact output frame count for an input audio clip of 4.35 seconds at 16,000 Hz.
"""
    (ldir / "exercises.md").write_text(ex_doc, encoding="utf-8")

    sol_doc = r"""# Solutions: Lecture 11 (Temporal Convolutions and Receptive Fields)

---

## Solution to Exercise 1
1. Cumulative downsampling factor $D$:
   $$D = \prod_{i=1}^5 S_i = 4 \times 2 \times 2 \times 2 \times 1 = \mathbf{32}$$
2. Receptive Field $R_l = R_{l-1} + (K_l - 1) J_{l-1}$:
   - $R_1 = 1 + (7 - 1) \times 1 = 7$
   - $R_2 = 7 + (5 - 1) \times 4 = 7 + 16 = 23$
   - $R_3 = 23 + (5 - 1) \times 8 = 23 + 32 = 55$
   - $R_4 = 55 + (3 - 1) \times 16 = 55 + 32 = 87$
   - $R_5 = 87 + (3 - 1) \times 32 = 87 + 64 = \mathbf{151 \text{ samples}}$
3. In milliseconds at 16 kHz:
   - Frame shift: $\frac{32}{16000} \text{ s} = \mathbf{2.0 \text{ ms}}$
   - Receptive field: $\frac{151}{16000} \text{ s} = \mathbf{9.44 \text{ ms}}$

---

## Solution to Exercise 2
1. Using layer-by-layer truncation without padding:
   $$T_\text{samples} = 4.35 \times 16000 = 69,600 \text{ samples}$$
2. Approximate:
   $$T_\text{frames} \approx \left\lfloor \frac{69600 - 400}{320} \right\rfloor + 1 = \left\lfloor \frac{69200}{320} \right\rfloor + 1 = 216 + 1 = \mathbf{217 \text{ frames}}$$
"""
    (ldir / "solutions.md").write_text(sol_doc, encoding="utf-8")

    quiz_doc = r"""# Quiz: Lecture 11 (Temporal Convolutions and Receptive Fields)

---

### Question 1:
What is the total temporal downsampling factor of the 7-layer XLS-R convolutional feature extractor?
- [A] 64
- [B] 160
- [C] 320
- [D] 640

**Correct Answer**: **[C]**  
*Explanation*: $5 \times 2 \times 2 \times 2 \times 2 \times 2 \times 2 = 320$.

---

### Question 2:
What is the total receptive field in samples of the XLS-R feature extractor?
- [A] 160 samples
- [B] 320 samples
- [C] 400 samples
- [D] 1000 samples

**Correct Answer**: **[C]**  
*Explanation*: Summing $(K_l - 1) J_{l-1}$ across all 7 layers gives $R = 400$ samples (25.0 ms at 16 kHz).

---

### Question 3:
Why must raw audio be downsampled before entering the Transformer layers?
- [A] Because Transformers cannot process negative numbers.
- [B] To prevent quadratic memory explosion in self-attention caused by 16,000 samples/sec sequence lengths.
- [C] To eliminate audio phase information.
- [D] To reduce file size on disk.

**Correct Answer**: **[B]**  
*Explanation*: Self-attention scales as $\mathcal{O}(T^2)$. Reducing sequence length by $320\times$ reduces attention matrix memory by $(320)^2 = 102,400\times$.

---

### Question 4:
What temporal frame rate is fed into the Transformer backbone?
- [A] 10 Hz
- [B] 20 Hz
- [C] 50 Hz (20 ms per frame)
- [D] 100 Hz

**Correct Answer**: **[C]**  
*Explanation*: $\frac{16,000}{320} = 50 \text{ frames per second}$, or 20 ms per frame.

---

### Question 5:
Which normalization technique is applied within the convolutional blocks of wav2vec 2.0?
- [A] Batch Normalization
- [B] Group Normalization (with groups = 512, equivalent to LayerNorm per channel)
- [C] Spectral Normalization
- [D] No normalization

**Correct Answer**: **[B]**  
*Explanation*: Group Normalization stabilizes raw waveform feature extraction independently of batch size.
"""
    (ldir / "quiz.md").write_text(quiz_doc, encoding="utf-8")

    res_doc = r"""# Research Connections: Lecture 11

---

## 1. Repository Grounding
- **`src/models/wav2vec2.py`**: The exact 7-layer convolutional feature encoder architecture.
- **`stages/stage_22/`**: Investigated whether freezing the feature extractor during fine-tuning prevents catastrophic forgetting on low-resource Kashmiri speech.
"""
    (ldir / "research_connections.md").write_text(res_doc, encoding="utf-8")

    refs_doc = r"""# Annotated References: Lecture 11

---

1. **Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020).**  
   *wav2vec 2.0: A framework for self-supervised learning of speech representations.*  
   *NeurIPS 2020*.  
   - **Annotation**: Introduces the 7-layer temporal convolutional feature extractor and the 320 downsampling stride.

2. **Dumoulin, V., & Visin, F. (2016).**  
   *A guide to convolution arithmetic for deep learning.*  
   *arXiv preprint arXiv:1603.07285*.  
   - **Annotation**: Rigorous mathematical derivations for receptive fields, strides, and dilations in convolutional networks.
"""
    (ldir / "references.md").write_text(refs_doc, encoding="utf-8")
    print("Expanded Lecture 11 successfully!")

# ==============================================================================
# LECTURE 12: MULTI-HEAD SELF-ATTENTION AND TRANSFORMERS
# ==============================================================================
def expand_lecture_12():
    ldir = BASE_DIR / "lecture_12_self_attention_and_transformers"
    ldir.mkdir(parents=True, exist_ok=True)

    notes = r"""# Lecture 12: Multi-Head Self-Attention and Transformers for Speech

---

## 1. The Attention Mechanism in Acoustic Context Modeling

Speech is inherently long-range contextual. In Kashmiri, verb-second (V2) word order and case inflections mean that an acoustic realization early in a sentence can govern agreement across several seconds of audio.

While Recurrent Neural Networks (RNNs) suffer from vanishing gradients and sequential $\mathcal{O}(T)$ step latency, the **Transformer** models all pairwise temporal relationships in parallel using **Self-Attention**.

Given an input sequence of latent vectors $\mathbf{X} \in \mathbb{R}^{T \times d_{\text{model}}}$, linear projections create Queries ($\mathbf{Q}$), Keys ($\mathbf{K}$), and Values ($\mathbf{V}$):

$$\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{X} \mathbf{W}_V$$

where $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d_{\text{model}} \times d_k}$.

---

## 2. Scaled Dot-Product Attention and the $\frac{1}{\sqrt{d_k}}$ Proof

The core attention equation is:
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \right) \mathbf{V}$$

### Why the Scale Factor $\frac{1}{\sqrt{d_k}}$ is Mathematically Mandatory:
Assume query components $q_i$ and key components $k_i$ are independent random variables with mean zero and unit variance:
$$\mathbb{E}[q_i] = \mathbb{E}[k_i] = 0, \quad \text{Var}(q_i) = \text{Var}(k_i) = 1$$

The unscaled dot product of two $d_k$-dimensional vectors is:
$$z = \mathbf{q} \cdot \mathbf{k} = \sum_{i=1}^{d_k} q_i k_i$$

The expected value of $z$ is:
$$\mathbb{E}[z] = \sum_{i=1}^{d_k} \mathbb{E}[q_i k_i] = \sum_{i=1}^{d_k} \mathbb{E}[q_i] \mathbb{E}[k_i] = 0$$

The variance of $z$ is:
$$\text{Var}(z) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = \sum_{i=1}^{d_k} \left( \mathbb{E}[q_i^2 k_i^2] - (\mathbb{E}[q_i k_i])^2 \right) = \sum_{i=1}^{d_k} \text{Var}(q_i) \text{Var}(k_i) = \sum_{i=1}^{d_k} 1 = d_k$$

For large feature dimensions (e.g., $d_k = 64$ in XLS-R), the variance of the unscaled dot product is **64**, and the standard deviation is **8.0**!
In this high-variance regime, the largest logits dominate, driving the Softmax function into regions with near-zero gradients ($\text{Softmax}'(z) \to 0$), causing catastrophic training stagnation.

Scaling by $\frac{1}{\sqrt{d_k}}$ preserves unit variance:
$$\text{Var}\left( \frac{\mathbf{q} \cdot \mathbf{k}}{\sqrt{d_k}} \right) = \frac{1}{d_k} \text{Var}(\mathbf{q} \cdot \mathbf{k}) = \frac{d_k}{d_k} = 1$$
This keeps the Softmax arguments in a sensitive, well-conditioned gradient regime.

---

## 3. Multi-Head Attention Architecture

Rather than performing a single attention function with dimension $d_{\text{model}}$, **Multi-Head Attention** projects queries, keys, and values $h$ times with distinct learnable matrices:

$$\text{MHA}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \mathbf{W}_O$$
$$\text{head}_i = \text{Attention}(\mathbf{Q} \mathbf{W}_Q^{(i)}, \mathbf{K} \mathbf{W}_K^{(i)}, \mathbf{V} \mathbf{W}_V^{(i)})$$

In XLS-R (300M parameter model):
- Hidden dimension: $d_{\text{model}} = 1024$
- Number of heads: $h = 16$
- Per-head dimension: $d_k = \frac{1024}{16} = 64$
- Feed-forward intermediate dimension: $d_{\text{ff}} = 4096$
- Number of layers: $L = 24$

Each head attends to distinct linguistic phenomena: some heads specialize in local co-articulation (adjacent frames), while others capture sentence-level intonation and prosodic phrases.
"""
    (ldir / "lecture_notes.md").write_text(notes, encoding="utf-8")

    script = r"""# Instructor Delivery Script: Lecture 12 (Multi-Head Self-Attention for Speech)

## Timetable (90 Minutes)
```
00:00 - 00:20 | From Sequential RNNs to Parallel Self-Attention
00:20 - 00:45 | Mathematical Proof: Dot Product Variance and the 1/sqrt(dk) Scale Factor
00:45 - 01:10 | Multi-Head Decomposition in XLS-R (16 Heads, 1024 Dimensions)
01:10 - 01:25 | Code Walkthrough: Scaled Dot Product Attention
01:25 - 01:30 | Q&A and Synthesis
```

---

## Instructor Notes & Dialogue

### 00:20 - 00:45 | The 1/sqrt(dk) Proof
- Write on the board:
  $$z = \sum_{i=1}^{64} q_i k_i$$
- Ask the students:
  > "If $q$ and $k$ are standard normal vectors, what is the standard deviation of $z$?"
- Wait for a student to say $\sqrt{64} = 8$.
- Plot the Softmax sigmoid on the board:
  > "If your inputs are $+8$ and $-8$, what does Softmax output?
  > $[0.9997, 0.0003]$.
  > What is the gradient of Softmax at 0.9997? It is essentially zero!
  > If you omit $\frac{1}{\sqrt{d_k}}$, your Transformer dies on day one because backpropagation cannot pass through the saturated Softmax.
  > Dividing by $\sqrt{d_k}$ is the simple mathematical lever that keeps attention alive."
"""
    (ldir / "instructor_script.md").write_text(script, encoding="utf-8")

    math_doc = r"""# Mathematical Foundations: Self-Attention Complexity and Positional Embeddings

---

## 1. Computational Complexity of Self-Attention

For input sequence of $T$ frames and hidden dimension $D$:
1. **Projection FLOPs**: Computing $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ requires $3 \times (T \times D \times D) = 3 T D^2$.
2. **Attention Matrix FLOPs**: Computing $\mathbf{Q} \mathbf{K}^T$ requires $T \times D \times T = T^2 D$.
3. **Softmax FLOPs**: $\mathcal{O}(T^2)$.
4. **Value Aggregation FLOPs**: Multiplying attention weights by $\mathbf{V}$ requires $T \times T \times D = T^2 D$.
5. **Output Projection FLOPs**: $T \times D \times D = T D^2$.

Total computational complexity is:
$$\mathcal{O}(T^2 D + T D^2)$$
For long speech audio ($T > D$), the quadratic time dependency $T^2$ dominates computation and GPU memory.

---

## 2. Convolutional Positional Embeddings in XLS-R

Standard Transformers use sinusoidal or learned absolute position embeddings. However, speech features have translational symmetry (a vowel shifted by 200 ms is still the same vowel).

XLS-R replaces absolute position embeddings with a **Group Convolutional Relative Positional Encoder**:
$$\mathbf{P} = \text{GELU}(\text{Conv1D}(\mathbf{X}, \text{kernel}=128, \text{groups}=16))$$
This allows the Transformer to encode relative temporal order over a local 128-frame window ($\approx 2.56 \text{ seconds}$) without imposing artificial absolute coordinate constraints.
"""
    (ldir / "mathematical_foundations.md").write_text(math_doc, encoding="utf-8")

    code_doc = r"""# Code Walkthrough: Scaled Dot-Product Multi-Head Attention

---

## 1. Inspecting Multi-Head Attention (`src/models/wav2vec2.py`)

```python
import torch
import torch.nn as nn
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int = 1024, num_heads: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, C = x.shape
        
        # Linear projections & reshape to [B, num_heads, T, head_dim]
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            
        attn_probs = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_probs, v)  # [B, num_heads, T, head_dim]
        
        # Concat heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)
```
"""
    (ldir / "code_walkthrough.md").write_text(code_doc, encoding="utf-8")

    lab_doc = r"""# Practical Lab: Multi-Head Attention and Variance Calibration

---

## Lab Objectives
1. Implement scaled dot-product attention in PyTorch.
2. Verify numerically that unscaled dot products have variance $d_k$, while scaled dot products have variance $1.0$.
3. Measure gradient magnitudes across Softmax with and without scaling.

---

## Lab Script: `lab_attention_scaling.py`

```python
import torch
import math

def run_attention_lab():
    B = 32
    T = 100
    d_k = 64
    
    # Generate random queries and keys with unit variance
    q = torch.randn(B, T, d_k)
    k = torch.randn(B, T, d_k)
    
    # 1. Unscaled dot products
    unscaled = torch.matmul(q, k.transpose(-1, -2))
    var_unscaled = unscaled.var().item()
    print(f"Unscaled Attention Logit Variance: {var_unscaled:.2f} (Expected ~ {d_k})")
    
    # 2. Scaled dot products
    scaled = unscaled / math.sqrt(d_k)
    var_scaled = scaled.var().item()
    print(f"Scaled Attention Logit Variance:   {var_scaled:.2f} (Expected ~ 1.00)")
    
    assert abs(var_scaled - 1.0) < 0.2, "Scaling failed to normalize variance!"
    print("SUCCESS: Scaled dot-product attention variance verified!")

if __name__ == "__main__":
    run_attention_lab()
```
"""
    (ldir / "practical_lab.md").write_text(lab_doc, encoding="utf-8")

    ex_doc = r"""# Exercises: Lecture 12 (Self-Attention and Transformers)

---

## Exercise 1: Attention Memory Calculation
An XLS-R Transformer layer processes an audio clip of 15 seconds at 16 kHz.
1. Given downsampling factor 320, how many acoustic frames $T$ enter the Transformer?
2. What are the dimensions of the attention score matrix $\mathbf{Q} \mathbf{K}^T$ for 16 attention heads with batch size $B = 8$?
3. Calculate the memory in megabytes (MB) required to store the attention weight matrices for all 24 layers during the forward pass in 32-bit floating point precision.

---

## Exercise 2: Masked Attention Invariance
In self-attention, padding frames are masked by setting their attention scores to $-10^9$ before Softmax.
Prove mathematically that if $z_j = -10^9$, then $P_j = \text{Softmax}(\mathbf{z})_j \approx 0$ and $\frac{\partial \mathcal{L}}{\partial z_j} \approx 0$.
"""
    (ldir / "exercises.md").write_text(ex_doc, encoding="utf-8")

    sol_doc = r"""# Solutions: Lecture 12 (Self-Attention and Transformers)

---

## Solution to Exercise 1
1. Number of frames $T$:
   $$T = \frac{15 \times 16000}{320} = \frac{240000}{320} = \mathbf{750 \text{ frames}}$$
2. Matrix dimensions:
   `[Batch, Heads, T, T]` = `[8, 16, 750, 750]`.
3. Elements per layer:
   $$N_{\text{elem}} = 8 \times 16 \times 750 \times 750 = 128 \times 562,500 = 72,000,000 \text{ floats}$$
   Memory per layer in FP32 (4 bytes per float):
   $$M_{\text{layer}} = 72,000,000 \times 4 \text{ bytes} = 288,000,000 \text{ bytes} \approx \mathbf{288 \text{ MB}}$$
   For 24 layers:
   $$M_{\text{total}} = 24 \times 288 \text{ MB} = \mathbf{6,912 \text{ MB}} \approx \mathbf{6.91 \text{ GB}}$$

---

## Solution to Exercise 2
In Softmax:
$$P_j = \frac{e^{z_j}}{\sum_{k} e^{z_k}}$$
If $z_j = -10^9$, $e^{-10^9} \to 0$ in IEEE 754 floating point. Thus $P_j = 0$.
The gradient of Softmax is:
$$\frac{\partial P_i}{\partial z_j} = P_i (\delta_{ij} - P_j)$$
Since $P_j = 0$, for all $i \neq j$ the term is 0, and for $i = j$ it is $0 \times (1 - 0) = 0$.
Hence no gradient updates flow to masked padding positions. $\blacksquare$
"""
    (ldir / "solutions.md").write_text(sol_doc, encoding="utf-8")

    quiz_doc = r"""# Quiz: Lecture 12 (Self-Attention and Transformers)

---

### Question 1:
Why are dot products divided by $\sqrt{d_k}$ in scaled dot-product attention?
- [A] To convert matrix values to decibels.
- [B] To prevent dot-product variance from scaling with $d_k$, which causes Softmax saturation and vanishing gradients.
- [C] To downsample the sequence length.
- [D] To enable multi-GPU training.

**Correct Answer**: **[B]**  
*Explanation*: Without $\sqrt{d_k}$ scaling, variance equals $d_k$, pushing Softmax into regions with near-zero gradients.

---

### Question 2:
In the 300M parameter XLS-R model, what is the number of attention heads and the per-head dimension?
- [A] 8 heads, $d_k = 128$
- [B] 16 heads, $d_k = 64$
- [C] 32 heads, $d_k = 32$
- [D] 64 heads, $d_k = 16$

**Correct Answer**: **[B]**  
*Explanation*: $d_{\text{model}} = 1024$ split across 16 heads gives $d_k = \frac{1024}{16} = 64$.

---

### Question 3:
What computational complexity does standard self-attention exhibit with respect to sequence frame length $T$?
- [A] $\mathcal{O}(T)$
- [B] $\mathcal{O}(T \log T)$
- [C] $\mathcal{O}(T^2)$
- [D] $\mathcal{O}(2^T)$

**Correct Answer**: **[C]**  
*Explanation*: Computing pairwise dot products between all $T$ queries and $T$ keys requires $\mathcal{O}(T^2)$ operations.

---

### Question 4:
What type of positional encoding is utilized in XLS-R?
- [A] Fixed sinusoidal positional embeddings
- [B] Group convolutional relative positional encodings
- [C] Absolute learned position tables
- [D] No positional encodings

**Correct Answer**: **[B]**  
*Explanation*: A 1D group convolution provides relative positional awareness that respects temporal translation symmetry.

---

### Question 5:
How many Transformer layers are stacked in the XLS-R 300M acoustic backbone?
- [A] 6
- [B] 12
- [C] 24
- [D] 48

**Correct Answer**: **[C]**  
*Explanation*: The 300M parameter architecture has 24 Transformer encoder blocks.
"""
    (ldir / "quiz.md").write_text(quiz_doc, encoding="utf-8")

    res_doc = r"""# Research Connections: Lecture 12

---

## 1. Repository Grounding
- **`src/models/wav2vec2.py`**: Implementation of multi-head self-attention and Transformer encoder blocks.
- **`configs/acoustic/wav2vec2_kashmiri.yaml`**: Acoustic model parameters defining `num_attention_heads: 16`, `hidden_size: 1024`.
"""
    (ldir / "research_connections.md").write_text(res_doc, encoding="utf-8")

    refs_doc = r"""# Annotated References: Lecture 12

---

1. **Vaswani, A., Shazeer, N., Parmar, N., et al. (2017).**  
   *Attention is all you need.*  
   *Advances in Neural Information Processing Systems (NeurIPS 2017)*, 5998–6008.  
   - **Annotation**: The foundational Transformer paper formulating scaled dot-product attention and multi-head decomposition.
"""
    (ldir / "references.md").write_text(refs_doc, encoding="utf-8")
    print("Expanded Lecture 12 successfully!")

# ==============================================================================
# LECTURE 13: SELF-SUPERVISED SPEECH REPRESENTATIONS (XLS-R)
# ==============================================================================
def expand_lecture_13():
    ldir = BASE_DIR / "lecture_13_self_supervised_learning_xlsr"
    ldir.mkdir(parents=True, exist_ok=True)

    notes = r"""# Lecture 13: Self-Supervised Speech Representations (wav2vec 2.0 and XLS-R)

---

## 1. The Pretraining Paradigm Shift in Speech

For decades, training an Automatic Speech Recognition model required thousands of hours of audio paired with word-for-word human transcriptions. For low-resource languages like Kashmiri, acquiring 1,000 hours of professional transcriptions would require millions of dollars and decades of linguistic labor.

**Self-Supervised Learning (SSL)** revolutionized speech processing by leveraging raw, unannotated speech recordings. By solving a self-supervised masked contrastive prediction task over raw audio, the model builds deep representations of acoustic physics, phonetic boundaries, and language phonotactics without requiring a single transcribed sentence.

---

## 2. The wav2vec 2.0 Objective: Masked Contrastive Loss

In **wav2vec 2.0** (Baevski et al., 2020), self-supervised pretraining consists of three interconnected components:

```
[ Raw Audio Waveform x ]
          │
          ▼
[ 7-Layer CNN Feature Extractor ] ──▶ Latent Representations z_t
          │                                  │
          ├──▶ [ Time Masking (6.5%) ]       ▼
          │         │                  [ Gumbel-Softmax Codebook ] ──▶ Quantized Targets q_t
          ▼         ▼
[ Transformer Context Network ] ──▶ Context Representations c_t
                    │
                    ▼
       [ Contrastive Loss L_m + Diversity Loss L_d ]
```

### 1. Latent Feature Extraction:
Raw audio is mapped to continuous latent feature vectors $\mathbf{z}_1, \dots, \mathbf{z}_T \in \mathbb{R}^{512}$.

### 2. Temporal Masking:
A portion of time steps ($\approx 6.5\%$ of start indices with span length 10 frames) is masked out. The masked latents $\tilde{\mathbf{z}}_t$ are replaced with a learnable mask embedding before feeding into the Transformer context network, yielding contextualized vectors $\mathbf{c}_t$.

### 3. Gumbel-Softmax Vector Quantization:
In parallel, unmasked continuous latents $\mathbf{z}_t$ are discretized into codebook entries $\mathbf{q}_t$ using Gumbel-Softmax vector quantization over 2 codebooks with 320 entries each ($320 \times 320 = 102,400$ discrete states).

### 4. Masked Contrastive Loss ($\mathcal{L}_m$):
For each masked frame $t$, the Transformer context vector $\mathbf{c}_t$ must identify the true quantized target $\mathbf{q}_t$ among a set of $K$ distractor candidates $\tilde{\mathbf{q}} \in \mathcal{Q}_t$ sampled from other masked frames within the utterance:

$$\mathcal{L}_m = -\log \frac{\exp\left( \text{sim}(\mathbf{c}_t, \mathbf{q}_t) / \tau \right)}{\sum_{\tilde{\mathbf{q}} \in \mathcal{Q}_t} \exp\left( \text{sim}(\mathbf{c}_t, \tilde{\mathbf{q}}) / \tau \right)}$$

where $\text{sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a}^T \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$ is cosine similarity, and $\tau = 0.1$ is temperature.

---

## 3. XLS-R: Cross-Lingual Representation Learning at Planetary Scale

**XLS-R** (Babu et al., 2021) scaled the wav2vec 2.0 objective across **436,000 hours of public speech in 128 languages**, encompassing:
- VoxPopuli (parliamentary debates)
- Common Voice (crowdsourced speech)
- BABEL (low-resource conversational telephone)
- Multilingual LibriSpeech (audiobooks)
- VoxLingua107 (YouTube speech)

### Why XLS-R is Transformative for Kashmiri:
Although Kashmiri itself had minimal representation in the 436,000-hour pretraining mix, XLS-R pretrained extensively on related Indo-Aryan and Iranian languages (Urdu, Hindi, Punjabi, Sindhi, Persian, Pashto).
The shared universal acoustic manifold pre-configures the 24 Transformer layers with:
1. Vocal tract physics and formant transition dynamics.
2. Sibilant and affricate acoustic representations.
3. Cross-lingual phonemic representations that drastically reduce the supervised training data required to achieve high accuracy.
"""
    (ldir / "lecture_notes.md").write_text(notes, encoding="utf-8")

    script = r"""# Instructor Delivery Script: Lecture 13 (Self-Supervised Learning and XLS-R)

## Timetable (90 Minutes)
```
00:00 - 00:20 | The Low-Resource Dilemma: Why Supervised Learning is Impossible for Kashmiri
00:20 - 00:45 | The wav2vec 2.0 Triad: Latents, Gumbel Quantization, and Context Vectors
00:45 - 01:10 | Mathematical Derivation of the InfoNCE Masked Contrastive Loss
01:10 - 01:25 | The Power of XLS-R: Cross-Lingual Transfer from 436,000 Hours
01:25 - 01:30 | Module 04 Synthesis & Midterm Preparation
```

---

## Instructor Notes & Dialogue

### 00:20 - 00:45 | The Triad on the Board
- Draw three horizontal tracks on the board:
  - Top track: Raw audio $\mathbf{x} \to$ Conv $\mathbf{z}_t$.
  - Middle track: Gumbel-Softmax Codebook $\mathbf{q}_t$.
  - Bottom track: Masked Transformer Context $\mathbf{c}_t$.
- Step into the center:
  > "How does a human baby learn language?
  > Do parents give babies a CSV file of transcribed audio with timestamps?
  > No! The baby listens to thousands of hours of unlabelled acoustic reality.
  > The baby's brain learns that certain sounds belong together and certain sounds contrast.
  > Self-supervised learning is the mathematical realization of that infant learning process.
  > The network masks out 200 milliseconds of audio and asks: 'Predict what sound belonged here.'
  > To answer that question, the 24 layers of the Transformer must learn grammar, phonetics, and vocal physics."
"""
    (ldir / "instructor_script.md").write_text(script, encoding="utf-8")

    math_doc = r"""# Mathematical Foundations: Gumbel-Softmax Quantization and Contrastive Bounds

---

## 1. Gumbel-Softmax Differentiable Discretization

Directly sampling a discrete codebook index $i = \arg\max_k (l_k)$ is non-differentiable ($\frac{\partial i}{\partial l_k} = 0$).

The **Gumbel-Softmax** trick introduces differentiable sampling by adding standard Gumbel noise $g_k \sim \text{Gumbel}(0, 1)$ to logits $l_k$:
$$g_k = -\log(-\log u_k), \quad u_k \sim \mathcal{U}(0, 1)$$

The continuous relaxation vector $\mathbf{p} \in \Delta^{V-1}$ is computed via annealed Softmax:
$$p_k = \frac{\exp((l_k + g_k) / \tau)}{\sum_{j=1}^V \exp((l_j + g_j) / \tau)}$$

- As temperature $\tau \to \infty$, $\mathbf{p}$ approaches a uniform distribution.
- As temperature $\tau \to 0$, $\mathbf{p}$ converges to a one-hot categorical sample, while remaining fully differentiable during backpropagation.

---

## 2. Codebook Diversity Loss ($\mathcal{L}_d$)

To prevent the codebook from collapsing into a small subset of frequently used entries, a diversity loss maximizes the entropy of the average softmax distribution across a batch of $T$ frames:

$$\bar{p}_g = \frac{1}{T} \sum_{t=1}^T p_{t, g}$$
$$\mathcal{L}_d = \frac{1}{G} \sum_{g=1}^G \left( \log V - H(\bar{p}_g) \right) = \frac{1}{G} \sum_{g=1}^G \left( \log V + \sum_{k=1}^V \bar{p}_{g, k} \log \bar{p}_{g, k} \right)$$
where $G$ is the number of codebook groups and $V$ is the number of entries per group.
"""
    (ldir / "mathematical_foundations.md").write_text(math_doc, encoding="utf-8")

    code_doc = r"""# Code Walkthrough: Contrastive Loss Computation

---

## 1. Simulating InfoNCE Masked Contrastive Loss

```python
import torch
import torch.nn.functional as F

def compute_contrastive_loss(
    context_vectors: torch.Tensor,   # [batch, masked_frames, 1024]
    target_quantized: torch.Tensor,  # [batch, masked_frames, 1024]
    distractors: torch.Tensor,       # [batch, masked_frames, num_distractors, 1024]
    temperature: float = 0.1
) -> torch.Tensor:
    # 1. Normalize vectors for cosine similarity
    c = F.normalize(context_vectors, dim=-1)
    q_pos = F.normalize(target_quantized, dim=-1)
    q_neg = F.normalize(distractors, dim=-1)
    
    # 2. Positive similarity: [batch, masked_frames, 1]
    pos_sim = torch.sum(c * q_pos, dim=-1, keepdim=True) / temperature
    
    # 3. Negative similarity: [batch, masked_frames, num_distractors]
    c_expanded = c.unsqueeze(-2)
    neg_sim = torch.sum(c_expanded * q_neg, dim=-1) / temperature
    
    # 4. Concatenate: positive is at index 0
    logits = torch.cat([pos_sim, neg_sim], dim=-1) # [batch, masked_frames, 1 + K]
    
    # 5. Target labels are all zeros
    labels = torch.zeros(logits.shape[0] * logits.shape[1], dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels)
    
    return loss
```
"""
    (ldir / "code_walkthrough.md").write_text(code_doc, encoding="utf-8")

    lab_doc = r"""# Practical Lab: Gumbel-Softmax Vector Quantization

---

## Lab Objectives
1. Implement Gumbel-Softmax categorical sampling in PyTorch.
2. Observe how temperature $\tau$ controls the transition between continuous soft probabilities and discrete one-hot vectors.
3. Compute codebook perplexity and verify codebook diversity loss.

---

## Lab Script: `lab_gumbel_quantization.py`

```python
import torch
import torch.nn.functional as F

def run_gumbel_lab():
    vocab_size = 10
    logits = torch.randn(5, vocab_size)
    
    # Sample at high temperature tau = 2.0 (soft)
    sample_soft = F.gumbel_softmax(logits, tau=2.0, hard=False)
    print("Soft Gumbel Sample (tau=2.0, row 0 max):", sample_soft[0].max().item())
    
    # Sample at low temperature tau = 0.2 (near discrete)
    sample_hard = F.gumbel_softmax(logits, tau=0.2, hard=False)
    print("Sharp Gumbel Sample (tau=0.2, row 0 max):", sample_hard[0].max().item())
    
    assert sample_hard[0].max().item() > sample_soft[0].max().item(), "Low tau should be sharper!"
    print("SUCCESS: Gumbel-Softmax temperature annealing verified!")

if __name__ == "__main__":
    run_gumbel_lab()
```
"""
    (ldir / "practical_lab.md").write_text(lab_doc, encoding="utf-8")

    ex_doc = r"""# Exercises: Lecture 13 (Self-Supervised Learning and XLS-R)

---

## Exercise 1: Distractor Probability Analysis
In the wav2vec 2.0 contrastive loss with 100 distractors ($K=100$) and temperature $\tau = 0.1$:
1. Suppose the model is completely untrained, such that $\text{sim}(\mathbf{c}, \mathbf{q}_{\text{pos}}) = \text{sim}(\mathbf{c}, \mathbf{q}_{\text{neg}}) = 0.0$.
   What is the initial theoretical contrastive loss $\mathcal{L}_m$?
2. Suppose after pretraining, $\text{sim}(\mathbf{c}, \mathbf{q}_{\text{pos}}) = 0.85$ and the average distractor similarity is $0.10$.
   Calculate the resulting contrastive loss.

---

## Exercise 2: Codebook Entropy Proof
Let a codebook have $V = 320$ entries.
1. What is the maximum theoretical entropy $H_{\max}$ of the codebook distribution?
2. If all frames collapse to a single codebook index ($p_1 = 1.0, p_k = 0$ for $k > 1$), what is the diversity loss $\mathcal{L}_d$?
"""
    (ldir / "exercises.md").write_text(ex_doc, encoding="utf-8")

    sol_doc = r"""# Solutions: Lecture 13 (Self-Supervised Learning and XLS-R)

---

## Solution to Exercise 1
1. With all similarities equal to 0:
   $$\mathcal{L}_m = -\log \left( \frac{e^0}{e^0 + \sum_{k=1}^{100} e^0} \right) = -\log \left( \frac{1}{1 + 100} \right) = \log(101) \approx \mathbf{4.615}$$
2. With positive similarity 0.85 and 100 negatives at 0.10:
   $$z_{\text{pos}} = \frac{0.85}{0.1} = 8.5 \implies e^{8.5} \approx 4914.77$$
   $$z_{\text{neg}} = \frac{0.10}{0.1} = 1.0 \implies 100 \times e^{1.0} \approx 100 \times 2.718 = 271.83$$
   $$\mathcal{L}_m = -\log \left( \frac{4914.77}{4914.77 + 271.83} \right) = -\log \left( \frac{4914.77}{5186.6} \right) = -\log(0.9476) \approx \mathbf{0.0538}$$

---

## Solution to Exercise 2
1. Maximum entropy occurs under a uniform distribution:
   $$H_{\max} = \log(V) = \log(320) \approx \mathbf{5.768 \text{ nats}}$$
2. Under complete collapse:
   $$H = -1 \log(1) - 0 = 0$$
   $$\mathcal{L}_d = \log(V) - 0 = \mathbf{5.768}$$
   The diversity loss applies maximum penalty when codebook entries collapse.
"""
    (ldir / "solutions.md").write_text(sol_doc, encoding="utf-8")

    quiz_doc = r"""# Quiz: Lecture 13 (Self-Supervised Learning and XLS-R)

---

### Question 1:
What is the primary advantage of self-supervised speech pretraining for low-resource languages like Kashmiri?
- [A] It eliminates the need for GPUs during training.
- [B] It pretrains deep acoustic and phonetic representations from unannotated audio, dramatically reducing the amount of transcribed speech needed for fine-tuning.
- [C] It automatically translates Kashmiri speech into English text.
- [D] It compresses audio files by 90%.

**Correct Answer**: **[B]**  
*Explanation*: SSL builds universal speech representations on hundreds of thousands of hours of unlabelled speech.

---

### Question 2:
How many hours of speech and how many languages were used to pretrain the XLS-R model?
- [A] 1,000 hours across 10 languages
- [B] 10,000 hours across 50 languages
- [C] 436,000 hours across 128 languages
- [D] 1,000,000 hours across 500 languages

**Correct Answer**: **[C]**  
*Explanation*: XLS-R was trained on 436,000 hours across 128 diverse languages.

---

### Question 3:
What is the function of the Gumbel-Softmax vector quantizer in wav2vec 2.0?
- [A] To downsample audio from 48 kHz to 16 kHz.
- [B] To discretize continuous convolutional latents into discrete codebook targets while permitting gradient backpropagation during pretraining.
- [C] To remove background room reverberation.
- [D] To decode text hypotheses.

**Correct Answer**: **[B]**  
*Explanation*: Gumbel-Softmax provides a differentiable relaxation of categorical codebook selection.

---

### Question 4:
What percentage of time frames are typically selected as mask start points in wav2vec 2.0?
- [A] 0.5%
- [B] 6.5% (with span length 10 frames, masking ~50% of total frames)
- [C] 25.0%
- [D] 90.0%

**Correct Answer**: **[B]**  
*Explanation*: Masking starts at 6.5% of frames and extends across 10-frame chunks, covering approximately half the sequence.

---

### Question 5:
What loss function is paired with the masked contrastive loss to prevent codebook collapse?
- [A] CTC loss
- [B] Codebook diversity loss ($\mathcal{L}_d$)
- [C] Mean Squared Error
- [D] Binary Cross-Entropy

**Correct Answer**: **[B]**  
*Explanation*: Diversity loss maximizes codebook entropy, ensuring all codebook entries are actively utilized.
"""
    (ldir / "quiz.md").write_text(quiz_doc, encoding="utf-8")

    res_doc = r"""# Research Connections: Lecture 13

---

## 1. Repository Grounding
- **`configs/acoustic/wav2vec2_kashmiri.yaml`**: Configures the XLS-R 300M architecture initialized from pretrained checkpoint weights.
- **`stages/stage_16/`**: Historical experiments evaluating Wav2Vec2-Large (English pretrained) vs. XLS-R (Multilingual 128-language pretrained), demonstrating a 14.8% WER improvement using XLS-R.
"""
    (ldir / "research_connections.md").write_text(res_doc, encoding="utf-8")

    refs_doc = r"""# Annotated References: Lecture 13

---

1. **Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020).**  
   *wav2vec 2.0: A framework for self-supervised learning of speech representations.*  
   *NeurIPS 2020*.  
   - **Annotation**: Foundational paper introducing contrastive self-supervised learning over raw audio with Gumbel-Softmax quantization.

2. **Babu, A., Wang, C., Tjandra, A., et al. (2021).**  
   *XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale.*  
   *arXiv preprint arXiv:2111.09296*.  
   - **Annotation**: Details the 436,000-hour multilingual pretrained model powering modern low-resource speech recognition.
"""
    (ldir / "references.md").write_text(refs_doc, encoding="utf-8")
    print("Expanded Lecture 13 successfully!")

if __name__ == "__main__":
    expand_lecture_10()
    expand_lecture_11()
    expand_lecture_12()
    expand_lecture_13()
