# Quiz: Lecture 33 (Production Streaming and Chunking)

---

### Question 1:
What are the window length and overlap duration defined by Method C2 chunking?
- [A] 1.0s window, 0.1s overlap
- [B] 8.0s window, 1.0s overlap
- [C] 60.0s window, 10.0s overlap
- [D] No overlap

**Correct Answer**: **[B]**  
*Explanation*: Method C2 uses an 8.0-second window with 1.0-second overlap (7.0s hop).

---

### Question 2:
What is the primary danger of chunking audio without an overlap region?
- [A] Audio file size doubles.
- [B] Words spoken across chunk boundaries are truncated, causing severe word deletion and substitution errors.
- [C] The GPU crashes.
- [D] Sampling rate drops to 8 kHz.

**Correct Answer**: **[B]**  
*Explanation*: Without overlap, boundary phonemes are chopped in half.

---

### Question 3:
What is Real-Time Factor (RTF)?
- [A] The speed of sound divided by wind speed.
- [B] The ratio of wall-clock processing time to audio duration ($\frac{T_{\text{proc}}}{T_{\text{audio}}}$).
- [C] The number of words recognized per minute.
- [D] The batch size.

**Correct Answer**: **[B]**  
*Explanation*: RTF measures compute efficiency relative to real-time playback.

---

### Question 4:
What empirical RTF is achieved by the EduCasheer production pipeline on an RTX 3090 GPU?
- [A] ~1.50 (slower than real-time)
- [B] ~0.082 (over 12 times faster than real-time)
- [C] 0.0001
- [D] Exactly 1.000

**Correct Answer**: **[B]**  
*Explanation*: An RTF of ~0.082 enables fast streaming and concurrent user serving.

---

### Question 5:
Why cannot a 30-minute continuous audio file be processed in a single forward pass through XLS-R?
- [A] Because Python does not allow numbers larger than 1,000.
- [B] Because quadratic self-attention memory ($\mathcal{O}(T^2)$) exceeds physical GPU VRAM, and relative positional embeddings lose calibration over long spans.
- [C] Because 30-minute files are illegal under copyright law.
- [D] Because CTC loss is limited to 10 seconds.

**Correct Answer**: **[B]**  
*Explanation*: Quadratic memory scaling and positional drift prevent monolithic long-form passes.
