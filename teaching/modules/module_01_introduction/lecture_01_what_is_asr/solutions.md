# Homework Solutions: Lecture 01

---

## Solution to Problem 1: Zero-One Loss vs. Levenshtein WER

### Part 1: Non-Equivalence Proof
The Bayes-optimal decision rule under arbitrary loss $L$ is:
$$\hat{\mathbf{w}}_L = \arg\min_{\hat{\mathbf{w}} \in \mathcal{W}} \sum_{\mathbf{w} \in \mathcal{W}} L(\mathbf{w}, \hat{\mathbf{w}}) P(\mathbf{w} \mid \mathbf{X})$$

Under zero-one loss, $L_{0-1}(\mathbf{w}, \hat{\mathbf{w}}) = 1 - \delta_{\mathbf{w}, \hat{\mathbf{w}}}$. The risk is $1 - P(\hat{\mathbf{w}} \mid \mathbf{X})$, which depends solely on the posterior probability of the single candidate $\hat{\mathbf{w}}$.

Under Levenshtein Word Error Rate, the loss $L_\text{WER}(\mathbf{w}, \hat{\mathbf{w}}) = \frac{\text{EditDist}(\mathbf{w}, \hat{\mathbf{w}})}{|\mathbf{w}|}$ is a metric taking continuous non-negative values.
The expected risk is:
$$R_\text{WER}(\hat{\mathbf{w}} \mid \mathbf{X}) = \sum_{\mathbf{w} \in \mathcal{W}} \left( \frac{\text{EditDist}(\mathbf{w}, \hat{\mathbf{w}})}{|\mathbf{w}|} \right) P(\mathbf{w} \mid \mathbf{X})$$

Notice that $R_\text{WER}$ is a weighted sum over all other hypotheses in the space $\mathcal{W}$.
A candidate $\hat{\mathbf{w}}_1$ with moderate posterior probability $P(\hat{\mathbf{w}}_1 \mid \mathbf{X}) = 0.35$ may share many common words with the remaining $65\%$ of probability mass (e.g. diverging by only 1 substitution). Its expected edit distance across the distribution is low.

Conversely, a candidate $\hat{\mathbf{w}}_2$ with the highest individual posterior $P(\hat{\mathbf{w}}_2 \mid \mathbf{X}) = 0.40$ may be an outlier sentence completely phonetically distant from the rest of the distribution.
In this case:
$$\hat{\mathbf{w}}_\text{MAP} = \hat{\mathbf{w}}_2 \ne \arg\min_{\hat{\mathbf{w}}} R_\text{WER}(\hat{\mathbf{w}} \mid \mathbf{X}) = \hat{\mathbf{w}}_1$$
Thus, MAP decoding is NOT Bayes-optimal under Word Error Rate. Minimizing expected WER requires Minimum Bayes Risk (MBR) decoding over word lattices. $\blacksquare$

### Part 2: Coincidence Conditions
The MAP rule coincides with the minimum expected WER rule if and only if the posterior distribution $P(\mathbf{w} \mid \mathbf{X})$ is degenerate (a Dirac delta function where $P(\mathbf{w}^* \mid \mathbf{X}) = 1.0$), or if the edit distance between all distinct pairs of hypotheses is uniform ($L(\mathbf{w}, \hat{\mathbf{w}}) = c \cdot \mathbb{I}(\mathbf{w} \ne \hat{\mathbf{w}})$).

---

## Solution to Problem 2: Decision Boundary Transition

### Part 1: Evaluating Scores at $\alpha = 1.0$
- **Hypothesis A**:
  $$\mathcal{S}(A) = \ln(0.045) + 1.0 \cdot \ln(0.015)$$
  $$\ln(0.045) = -3.1011, \quad \ln(0.015) = -4.1997$$
  $$\mathcal{S}(A) = -3.1011 - 4.1997 = -7.3008$$

- **Hypothesis B**:
  $$\mathcal{S}(B) = \ln(0.090) + 1.0 \cdot \ln(0.002)$$
  $$\ln(0.090) = -2.4079, \quad \ln(0.002) = -6.2146$$
  $$\mathcal{S}(B) = -2.4079 - 6.2146 = -8.6225$$

Since $\mathcal{S}(A) = -7.3008 > \mathcal{S}(B) = -8.6225$, **Hypothesis A is selected**.

### Part 2: Finding Critical Alpha $\alpha^*$
Set $\mathcal{S}(A) = \mathcal{S}(B)$:
$$\ln P(\mathbf{X} \mid A) + \alpha^* \ln P(A) = \ln P(\mathbf{X} \mid B) + \alpha^* \ln P(B)$$
$$\ln P(\mathbf{X} \mid A) - \ln P(\mathbf{X} \mid B) = \alpha^* \left[ \ln P(B) - \ln P(A) \right]$$
$$\ln\left( \frac{P(\mathbf{X} \mid A)}{P(\mathbf{X} \mid B)} \right) = \alpha^* \ln\left( \frac{P(B)}{P(A)} \right)$$
$$\alpha^* = \frac{\ln(0.045 / 0.090)}{\ln(0.002 / 0.015)} = \frac{\ln(0.50)}{\ln(0.13333)} = \frac{-0.69315}{-2.01490} \approx 0.3440$$

**Conclusion**:
- For $\alpha > 0.3440$: Language model prior dominates; Hypothesis A wins.
- For $\alpha < 0.3440$: Acoustic model evidence dominates; Hypothesis B wins.

---

## Solution to Problem 3: PyTorch SNR Implementation

```python
import torch

def calculate_audio_snr(signal_tensor: torch.Tensor, noise_tensor: torch.Tensor) -> float:
    if signal_tensor.shape != noise_tensor.shape:
        raise ValueError("Signal and noise tensors must have identical shapes.")
        
    p_signal = torch.mean(signal_tensor ** 2)
    p_noise = torch.mean(noise_tensor ** 2)
    
    if p_noise.item() == 0.0:
        return float('inf')
    if p_signal.item() == 0.0:
        return float('-inf')
        
    snr_db = 10.0 * torch.log10(p_signal / p_noise)
    return snr_db.item()
```

---

## Solution to Problem 4: Urdu vs. Kashmiri Phonetic Failure

1. **Phonetic Inventory Divergence**:
   - **Kashmiri Consonants Absent in Urdu**:
     - Dental affricate /t͡s/ (ژ)
     - Aspirated dental affricate /t͡sʰ/ (څ)
     - Palatalized alveolar consonants (e.g. /tʲ/, /dʲ/, /lʲ/)
   - **Kashmiri Vowels Absent in Urdu**:
     - High central unrounded vowel /ɨ/ (Tschapre hath: `U+0655`)
     - Mid central unrounded vowel /ə/ (Pesh with circle: `U+0654`)
2. **Why Central Vowels Cause Deletions**:
   Central vowels in Kashmiri are frequently short ($30 - 50\text{ ms}$). Because the Urdu acoustic model was never trained on audio containing central formant resonances ($F_1 \approx 500\text{ Hz}, F_2 \approx 1500\text{ Hz}$), the neural encoder outputs low activations across all its known phone heads. The highest output emission probability defaults to the blank token $\epsilon$ or silence. Under the CTC collapse operator $\mathcal{B}$, sequences of blanks are stripped, completely eliminating the vowel and producing a deletion error.
