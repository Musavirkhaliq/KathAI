# Homework Exercises: Lecture 01

---

## Analytical & Mathematical Problems

### Problem 1: Zero-One Loss and Minimum Word Error Rate
In Lecture 01, we proved that maximizing the posterior probability $P(\mathbf{w} \mid \mathbf{X})$ minimizes the sequence-level zero-one error $L_{0-1}(\mathbf{w}, \hat{\mathbf{w}}) = \mathbb{I}(\mathbf{w} \ne \hat{\mathbf{w}})$.
1. In speech recognition, systems are evaluated using **Word Error Rate (WER)** based on Levenshtein distance:
   $$L_\text{WER}(\mathbf{w}, \hat{\mathbf{w}}) = \frac{\text{Sub}(\mathbf{w}, \hat{\mathbf{w}}) + \text{Del}(\mathbf{w}, \hat{\mathbf{w}}) + \text{Ins}(\mathbf{w}, \hat{\mathbf{w}})}{|\mathbf{w}|}$$
   Prove that the MAP decision rule $\hat{\mathbf{w}} = \arg\max_{\mathbf{w}} P(\mathbf{w} \mid \mathbf{X})$ is **NOT** the Bayes-optimal decision rule under $L_\text{WER}$.
2. Under what theoretical conditions would the MAP rule coincide with the minimum expected Word Error Rate rule?

### Problem 2: The Decision Boundary Transition
An acoustic model evaluates two candidate Kashmiri transcriptions:
- Hypothesis $A$: $\text{"کٲشُر زَبان"}$ with acoustic likelihood $P(\mathbf{X} \mid A) = 0.045$.
- Hypothesis $B$: $\text{"کاشُر زَبان"}$ with acoustic likelihood $P(\mathbf{X} \mid B) = 0.090$.

A statistical language model assigns priors:
$$P(A) = 0.015, \quad P(B) = 0.002$$

Under the shallow fusion decoding objective:
$$\mathcal{S}(\mathbf{w}) = \ln P(\mathbf{X} \mid \mathbf{w}) + \alpha \ln P(\mathbf{w})$$
1. Calculate the score of Hypothesis $A$ and Hypothesis $B$ when $\alpha = 1.0$. Which hypothesis wins?
2. Derive the exact critical value $\alpha^*$ such that for all $\alpha > \alpha^*$, Hypothesis $A$ is selected, and for all $\alpha < \alpha^*$, Hypothesis $B$ is selected.

---

## Implementation & Coding Problems

### Problem 3: Signal-to-Noise Ratio (SNR) Verification
Write a self-contained PyTorch function:
```python
def calculate_audio_snr(signal_tensor: torch.Tensor, noise_tensor: torch.Tensor) -> float:
    pass
```
Your function must:
1. Accept two 1D floating-point tensors of identical length representing isolated speech and background noise.
2. Compute the average signal power $P_s = \frac{1}{N} \sum x_i^2$ and noise power $P_n = \frac{1}{N} \sum n_i^2$.
3. Handle the edge case where noise power is zero (return `float('inf')`).
4. Return the Signal-to-Noise Ratio in decibels: $\text{SNR}_\text{dB} = 10 \log_{10}(P_s / P_n)$.

---

## Forensic Research & Auditing Problems

### Problem 4: The Stage 01 Urdu Failure Analysis
In Stage 01 of the EduCasheer repository (`stages/stage_01/`), the zero-shot Urdu Wav2Vec2 model achieved an $88.4\%$ Word Error Rate on Kashmiri test audio.
1. Inspect the phonetic inventory of Urdu versus Kashmiri. List three specific consonants and two vowels that exist in Kashmiri but have no equivalent acoustic representation in Urdu.
2. Explain why an acoustic model lacking phonetic targets for central vowels (/ɨ/, /ə/) exhibits high deletion error rates rather than substitution errors.
