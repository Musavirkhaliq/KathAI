# Solutions: Lecture 35 (Capstone Synthesis and Open Horizons)

---

## Solution to Exercise 1
1. **Low-Resource Stability**: CTC over self-supervised pretrained representations (XLS-R) trains robustly on < 100 hours of speech, whereas autoregressive decoders (AED/Whisper) rapidly overfit.
2. **Hallucination Risk**: CTC guarantees monotonic alignment, completely preventing loops and hallucinations.
3. **Streaming Latency**: CTC operates in linear time $\mathcal{O}(T)$ without autoregressive feedback loops.
4. **Shallow Fusion**: External $n$-gram and neural language models integrate seamlessly into prefix beam search via log-linear score addition.
- **Verdict**: CTC maximizes acoustic fidelity and operational stability under data scarcity.

---

## Solution to Exercise 2
1. **Frontend**: A dedicated diacritic restoration module (or central vowel predictor) must expand un-diacriticized text into fully specified phonetic sequences.
2. **Acoustic Generator**: VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech) utilizing normalizing flows and HiFi-GAN vocoder.
3. **Vocal Formants**: Incorporate explicit pitch ($F_0$) and formant predictors to ensure accurate synthesis of the central vowel /əː/ ($F_2 \approx 1550 \text{ Hz}$).
