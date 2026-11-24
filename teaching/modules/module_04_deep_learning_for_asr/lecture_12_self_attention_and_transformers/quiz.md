# Quiz: Lecture 12 (Self-Attention and Transformers)

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
