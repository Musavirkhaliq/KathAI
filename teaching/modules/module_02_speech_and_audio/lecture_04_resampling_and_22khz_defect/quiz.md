# Quiz: Lecture 04 (Resampling Algorithms and the 22.05 kHz Defect)

---

### Question 1:
What are the simplified coprime upsampling ($L$) and downsampling ($M$) factors when converting audio from 22,050 Hz to 16,000 Hz?
- [A] $L = 16, M = 22$
- [B] $L = 320, M = 441$
- [C] $L = 160, M = 441$
- [D] $L = 441, M = 320$

**Correct Answer**: **[B]**  
*Explanation*: $\frac{16000}{22050} = \frac{1600}{2205} = \frac{320}{441}$.

---

### Question 2:
What was the primary root cause of the 22.05 kHz Bolbosh defect discovered during Stage 03?
- [A] The audio files were accidentally deleted from disk.
- [B] An overly aggressive anti-aliasing filter cutoff combined with improper Kaiser window parameters wiped out acoustic energy between 5.8 kHz and 8.0 kHz.
- [C] Bolbosh speakers spoke an unrecognized dialect of Dardic.
- [D] The audio was converted to 8-bit unsigned integer format.

**Correct Answer**: **[B]**  
*Explanation*: The resampler had a defective low-pass cutoff and excessively wide transition band that attenuated genuine high-frequency fricative energy.

---

### Question 3:
Why is polyphase decomposition preferred over naive upsampling followed by direct convolution?
- [A] Polyphase decomposition eliminates the need for a low-pass filter.
- [B] It avoids multiplying filter coefficients by the $L-1$ zeros inserted during upsampling, drastically reducing memory operations and FLOPs.
- [C] Polyphase filters only run on GPU hardware.
- [D] It increases the sampling rate to 96 kHz.

**Correct Answer**: **[B]**  
*Explanation*: Polyphase representations restructure the calculation so multiplications occur only on genuine non-zero input samples.

---

### Question 4:
Which phonetic class in Kashmiri was most severely damaged by the historical 22.05 kHz resampling bug?
- [A] Low back vowels (/aː/, /o/)
- [B] Bilabial nasals (/m/)
- [C] Coronal and palatal fricatives and affricates (/s/, /ʃ/, /t͡s/)
- [D] Voiced retroflex plosives (/ɖ/)

**Correct Answer**: **[C]**  
*Explanation*: Sibilants and affricates have distinct acoustic signatures in the $5.5 - 8.0 \text{ kHz}$ band, which fell directly into the filter attenuation notch.

---

### Question 5:
In `torchaudio.transforms.Resample`, what does the `rolloff` parameter specify?
- [A] The volume reduction in decibels.
- [B] The fraction of the Nyquist frequency at which the anti-aliasing low-pass filter cutoff is established.
- [C] The number of seconds of audio trimmed from the start of the recording.
- [D] The batch size for GPU parallelization.

**Correct Answer**: **[B]**  
*Explanation*: `rolloff=0.94` sets the lowpass filter cutoff to $0.94 \times \frac{f_s}{2}$.
