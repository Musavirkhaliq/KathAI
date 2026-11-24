# Solutions: Lecture 34 (MLOps and Golden Logit Parity Testing)

---

## Solution to Exercise 1
1. Different GPU architectures use different CUDA math libraries, instruction scheduling, and Fused Multiply-Add (FMA) implementations. Minor non-associative additions in parallel reductions ($a + b \neq b + a$) create differences in the least significant bits of FP32 mantissas ($< 10^{-6}$).
2. $\tau = 10^{-5}$ is large enough to absorb benign non-associative GPU rounding differences, yet orders of magnitude too small to affect Softmax probabilities or alter beam search decoding decisions.

---

## Solution to Exercise 2
The 5 mandatory CI stages:
1. **Linting & Code Formatting**: `flake8`, `black`, `isort`.
2. **Text Normalization Invariant Tests**: Verify idempotence and homoglyph unification on 1,000 edge cases.
3. **Audio Processing Unit Tests**: Check Kaiser resampler preservation up to 7.5 kHz.
4. **Golden Logit Parity Test**: Verify max difference $< 10^{-5}$ against reference audio.
5. **Fast Integration Test**: Transcribe a 3-utterance toy benchmark and assert expected WER $< 5\%$.
