# Instructor Delivery Script: Lecture 04 (Resampling Algorithms and the 22.05 kHz Defect)

## Timetable & Pedagogical Flow (90 Minutes Total)

```
00:00 - 00:15 | The Reality of Multi-Rate Speech Corpora
00:15 - 00:40 | Mathematical Foundations: Rational Polyphase Resampling (L/M = 320/441)
00:40 - 01:05 | Case Study: The 22.05 kHz Bolbosh Disaster in EduCasheer
01:05 - 01:20 | Forensic Code Inspection: Comparing the Broken vs. Certified Resampler
01:20 - 01:30 | Empirical Verification: Spectrogram Inspection and Audio Diffing
```

---

## Detailed Delivery Script

### 00:00 - 00:15 | Opening Hook
- Show an audio waveform on the projector.
  > "In machine learning research, we love to pretend that data is clean. You download LibriSpeech, it's 16 kHz 16-bit FLAC. Life is easy.
  > But when you build ASR for an endangered or low-resource language, you don't get pristine studio data. You get Bolbosh: 18 hours recorded on early 2000s hardware at 22,050 Hz.
  > What happens if you run `audio.resample(16000)` without knowing what the math is doing?
  > In our repository, it introduced a bug that went unnoticed for 12 research stages, silently degrading our Word Error Rate by over 6% absolute.
  > Today, we conduct a forensic autopsy of that defect."

### 00:15 - 00:40 | Rational Factor Resampling
- Write the ratio on the board:

$$\frac{16000}{22050} = \frac{320}{441}$$

- Explain the three-step polyphase sequence:
  1. Upsample by $L = 320$ (insert 319 zeros between each sample).
  2. Apply an anti-imaging and anti-aliasing digital low-pass filter $h[n]$ with cutoff $\omega_c = \min\left(\frac{\pi}{L}, \frac{\pi}{M}\right) = \frac{\pi}{441}$.
  3. Downsample by $M = 441$ (keep every 441st sample).
- Show the polyphase identity: We do not compute 319 zeros! We commute the filter into $L$ polyphase sub-filters.

### 00:40 - 01:05 | The Forensic Autopsy
- Pull up the spectrogram comparison on the screen:
  - Left: Original Bolbosh 22.05 kHz audio (fricative energy up to 10 kHz).
  - Middle: Stage 03 Resampled audio (black void above 5.8 kHz).
  - Right: Stage 26 Certified Resampled audio (clean preservation up to 7.5 kHz).
- Walk through the linguistic consequence:
  - Point out the Kashmiri word **`ژور`** (/t͡sʊr/, four).
  - The affricate /t͡s/ needs high-frequency energy. Without it, the acoustic model misclassified it as **`تور`** (/tʊr/, cold) or **`دور`** (/dʊr/, far).

### 01:05 - 01:30 | Live Terminal Demo & Verification
- Execute the script comparing `scipy.signal.resample` vs `torchaudio.transforms.Resample(sinc_interp_kaiser)`.
- Show students the difference in execution speed and frequency response.
