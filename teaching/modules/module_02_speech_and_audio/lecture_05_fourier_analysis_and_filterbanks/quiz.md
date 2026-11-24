# Quiz: Lecture 05 (Fourier Analysis and Mel Filterbanks)

---

### Question 1:
What fundamental trade-off is enforced by the Gabor-Heisenberg uncertainty principle in spectral analysis?
- [A] Audio cannot be recorded in stereo without losing phase information.
- [B] High temporal resolution ($\Delta t$) and high frequency resolution ($\Delta f$) cannot be simultaneously achieved; narrowing the analysis window broadens the spectral frequency resolution.
- [C] Audio cannot be quantized to 16 bits without introducing aliasing.
- [D] Increasing the learning rate reduces the batch size.

**Correct Answer**: **[B]**  
*Explanation*: The uncertainty bound $\Delta t \cdot \Delta f \ge \frac{1}{4\pi}$ mathematically restricts simultaneous precision in both time and frequency.

---

### Question 2:
Why is a tapered window function (e.g., Hann or Hamming) applied to speech frames before computing the Discrete Fourier Transform?
- [A] To increase the volume of the audio.
- [B] To eliminate boundary discontinuities at the edges of the finite frame, suppressing high-frequency spectral leakage.
- [C] To downsample the signal from 44.1 kHz to 16 kHz.
- [D] To convert complex numbers into real floating-point numbers.

**Correct Answer**: **[B]**  
*Explanation*: Abrupt truncation creates sharp boundary steps that generate high-frequency sinc sidelobes. Tapering smoothly forces endpoints to zero.

---

### Question 3:
How does frequency spacing change across an 80-channel Mel filterbank between 0 Hz and 8,000 Hz?
- [A] Filter bandwidths remain constant at 100 Hz across the entire spectrum.
- [B] Filter bandwidths are narrow and linearly spaced below 1 kHz, but become progressively wider and logarithmic above 1 kHz.
- [C] Filter bandwidths are wide at low frequencies and narrow at high frequencies.
- [D] Bandwidths are randomized to improve neural network regularization.

**Correct Answer**: **[B]**  
*Explanation*: The Mel scale approximates human cochlear resolution: fine linear discrimination at low frequencies and coarse logarithmic discrimination at high frequencies.

---

### Question 4:
What computational complexity does the Cooley-Tukey Fast Fourier Transform achieve for an N-point sequence?
- [A] $\mathcal{O}(N)$
- [B] $\mathcal{O}(N \log N)$
- [C] $\mathcal{O}(N^2)$
- [D] $\mathcal{O}(2^N)$

**Correct Answer**: **[B]**  
*Explanation*: The divide-and-conquer radix-2 FFT reduces evaluation complexity from $\mathcal{O}(N^2)$ to $\mathcal{O}(N \log N)$.

---

### Question 5:
Why do modern self-supervised models like XLS-R process raw 1D waveforms instead of hand-crafted Log-Mel filterbanks?
- [A] Raw waveforms require less GPU memory than spectrograms.
- [B] Raw waveforms preserve complete phase information and allow learnable temporal convolutional filters to optimize frequency bands tailored directly to the task.
- [C] Log-Mel filterbanks are incompatible with PyTorch tensors.
- [D] Waveforms cannot be normalized.

**Correct Answer**: **[B]**  
*Explanation*: Hand-crafted filterbanks discard phase and impose rigid triangular filter shapes. End-to-end learnable 1D convolutions allow the model to discover optimal representation filters.
