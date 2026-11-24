# Mathematical Foundations: Self-Attention Complexity and Positional Embeddings

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
