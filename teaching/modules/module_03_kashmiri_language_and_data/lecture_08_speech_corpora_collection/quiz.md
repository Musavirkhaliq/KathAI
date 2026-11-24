# Quiz: Lecture 08 (Speech Corpora Collection)

---

### Question 1:
How many distinct speech sources were collected and audited in the EduCasheer repository?
- [A] 2 sources
- [B] 12 sources
- [C] 45 sources
- [D] 100 sources

**Correct Answer**: **[B]**  
*Explanation*: The repository audited 12 diverse sources encompassing studio, broadcast, religious, educational, and spontaneous speech.

---

### Question 2:
What is the minimum SNR threshold enforced by the clean curation pipeline?
- [A] 0 dB
- [B] 12 dB
- [C] 40 dB
- [D] 80 dB

**Correct Answer**: **[B]**  
*Explanation*: 12 dB ensures speech acoustic clarity without discarding valuable authentic dialectal recordings.

---

### Question 3:
Why is hard digital clipping detrimental to ASR training?
- [A] It decreases the sampling rate.
- [B] It generates spurious high-frequency harmonics and destroys acoustic formant trajectories.
- [C] It crashes the PyTorch dataloader.
- [D] It converts stereo audio into mono.

**Correct Answer**: **[B]**  
*Explanation*: Hard saturation creates sharp waveform edges that pollute the spectrum with non-linear distortion.

---

### Question 4:
Which dataset source contains spontaneous open-air mosque sermons with heavy acoustic reverberation?
- [A] Bolbosh
- [B] IndicVoices
- [C] WAAZ (SRC-04)
- [D] Radio Kashmir

**Correct Answer**: **[C]**  
*Explanation*: WAAZ represents challenging real-world spontaneous sermon audio.

---

### Question 5:
What is the total raw audio volume held across the 12 sources in the repository?
- [A] 20.5 hours
- [B] 75.0 hours
- [C] 214.65 hours
- [D] 10,000 hours

**Correct Answer**: **[C]**  
*Explanation*: The comprehensive audit verified 214.65 total raw hours across 106,487 audio segments.
