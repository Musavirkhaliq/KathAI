# Quiz: Lecture 11 (Temporal Convolutions and Receptive Fields)

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
