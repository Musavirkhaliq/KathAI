# Quiz: Lecture 10 (Deep Neural Networks and Backpropagation)

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
