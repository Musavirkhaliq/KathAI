# CS-892: Automatic Speech Recognition — Midterm Examination
**Scope**: Modules 01 through 06 (Foundations, Signal Processing, Kashmiri Data, Deep Learning, CTC, Scaling)  
**Duration**: 120 Minutes | **Total Marks**: 100 Points

---

## Section A: Conceptual & Theoretical Foundations (30 Marks)
1. **(10 Marks)** Deconstruct the Fundamental Equation of Speech Recognition $\hat{\mathbf{w}} = \arg\max_{\mathbf{w}} P(\mathbf{X} \mid \mathbf{w}) P(\mathbf{w})$. Explain why end-to-end CTC models replace frame-level HMMs, and why character-level tokenization is superior to Byte-Pair Encoding (BPE) in low-resource Kashmiri speech recognition.
2. **(10 Marks)** State the Nyquist-Shannon Sampling Theorem. Using continuous and discrete Fourier equations, prove that downsampling Bolbosh audio from $22{,}050\text{ Hz}$ to $16{,}000\text{ Hz}$ without an anti-aliasing lowpass filter folds frequencies between $8\text{ kHz}$ and $11.025\text{ kHz}$ back into the $[4.975\text{ kHz}, 8\text{ kHz}]$ speech band.
3. **(10 Marks)** Explain why standard random train/test data splitting causes severe optimistic bias (speaker leakage) in speech recognition, and formulate the mathematical graph-partitioning constraint required for zero-speaker overlap.

---

## Section B: Mathematical Derivations (40 Marks)
4. **(20 Marks)** Consider the 7-layer temporal convolutional feature encoder in Wav2Vec 2.0 with kernel sizes $K = (10, 3, 3, 3, 3, 2, 2)$ and strides $S = (5, 2, 2, 2, 2, 2, 2)$.
   - Derive the cumulative downsampling stride $R$ and the resulting frame rate at $16\text{ kHz}$.
   - Using the recursive formula $RF_l = RF_{l-1} + (K_l - 1) J_{l-1}$, prove that the total receptive field is exactly $400$ audio samples ($25\text{ ms}$).
5. **(20 Marks)** In Connectionist Temporal Classification (CTC):
   - Formulate the modified target sequence $\mathbf{l}'$ of length $2U + 1$.
   - Write out the complete forward variable recursion $\alpha_t(s)$ distinguishing cases where $l'_s = \epsilon$, $l'_s = l'_{s-2}$, and $l'_s \ne l'_{s-2}$.
   - Derive the analytical gradient $\frac{\partial \mathcal{L}_\text{CTC}}{\partial u_t^k} = y_t^k - P(\pi_t = k \mid \mathbf{y}, \mathbf{X})$ with respect to unnormalized neural logits $u_t^k$.

---

## Section C: Empirical Systems & Optimization (30 Marks)
6. **(15 Marks)** In training deep models on variable-length audio, explain why Layer Normalization is preferred over Batch Normalization. Calculate the baseline GPU memory strictly required for model parameters, gradients, and AdamW optimizer states for a 300M parameter model before activations are loaded.
7. **(15 Marks)** Deconstruct the empirical scaling progression from Model B-01 ($10\text{h}$) to Model B-02 ($50\text{h}$) and Model B-02C ($100\text{h}$). Explain why acoustic scaling alone encounters diminishing returns and cannot overcome the $25\%$ WER barrier without linguistic modeling.
