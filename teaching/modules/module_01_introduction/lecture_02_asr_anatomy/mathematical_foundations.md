# Mathematical Foundations: Pipeline Anatomy and Decoding Formulation

---

## 1. Convolutional Receptive Field and Stride Arithmetic

The acoustic feature extractor of the `facebook/wav2vec2-xls-r-300m` model consists of a 7-layer temporal 1D convolutional network operating on raw audio waveforms sampled at $f_s = 16,000 \text{ Hz}$.

### 1.1 Temporal Convolution Dimension Equation
For a 1D convolution with input sequence length $L_{\text{in}}$, kernel size $K$, stride $S$, and zero padding $P=0$, the output sequence length $L_{\text{out}}$ is given by:

$$L_{\text{out}} = \left\lfloor \frac{L_{\text{in}} - K}{S} \right\rfloor + 1$$

### 1.2 Cumulative Stride Derivation
Let $S_\ell$ denote the stride of layer $\ell \in \{1, \dots, 7\}$. The cumulative temporal downsampling factor $S_{\text{total}}$ of the 7-layer cascade is:

$$S_{\text{total}} = \prod_{\ell=1}^7 S_\ell = 5 \times 2 \times 2 \times 2 \times 2 \times 2 \times 2 = 5 \times 2^6 = 320 \text{ samples}$$

Given $f_s = 16,000 \text{ Hz}$, the frame period $\Delta t$ between consecutive output frames is:

$$\Delta t = \frac{S_{\text{total}}}{f_s} = \frac{320}{16,000} \text{ s} = 0.020 \text{ s} = 20 \text{ ms}$$

The corresponding frame emission frequency is:

$$f_{\text{frame}} = \frac{1}{\Delta t} = 50 \text{ Hz}$$

### 1.3 Total Receptive Field Recursion
The receptive field $R_\ell$ defines the number of input audio samples in the raw waveform that influence a single output activation at layer $\ell$. It satisfies the recursive formula:

$$R_\ell = R_{\ell-1} + (K_\ell - 1) \cdot \prod_{j=1}^{\ell-1} S_j \quad \text{with } R_0 = 1$$

Evaluating step-by-step:
1. Layer 1 ($K_1=10, S_1=5$):

$$R_1 = 1 + (10 - 1) \times 1 = 10 \text{ samples}$$

2. Layer 2 ($K_2=3, S_2=2$):

$$R_2 = 10 + (3 - 1) \times 5 = 10 + 10 = 20 \text{ samples}$$

3. Layer 3 ($K_3=3, S_3=2$):

$$R_3 = 20 + (3 - 1) \times 10 = 20 + 20 = 40 \text{ samples}$$

4. Layer 4 ($K_4=3, S_4=2$):

$$R_4 = 40 + (3 - 1) \times 20 = 40 + 40 = 80 \text{ samples}$$

5. Layer 5 ($K_5=3, S_5=2$):

$$R_5 = 80 + (3 - 1) \times 40 = 80 + 80 = 160 \text{ samples}$$

6. Layer 6 ($K_6=2, S_6=2$):

$$R_6 = 160 + (2 - 1) \times 80 = 160 + 80 = 240 \text{ samples}$$

7. Layer 7 ($K_7=2, S_7=2$):

$$R_7 = 240 + (2 - 1) \times 160 = 240 + 160 = 400 \text{ samples}$$

$$\text{Total Receptive Field} = 400 \text{ samples} = \frac{400}{16,000} \text{ s} = 25 \text{ ms}$$

### 1.4 Exact Valid CTC Length
An input audio buffer with $N$ samples produces exactly $T_{\text{valid}}$ frames if and only if $N \ge 400$:

$$T_{\text{valid}} = \left\lfloor \frac{N - 400}{320} \right\rfloor + 1$$

---

## 2. The CTC Collapse Operator $\mathcal{B}$ and Alignment Combinatorics

Let $\Omega$ be the alphabet of 46 non-blank characters, and define $\Omega' = \Omega \cup \{\epsilon\}$. An alignment $\boldsymbol{\pi} = (\pi_1, \dots, \pi_T) \in (\Omega')^T$ maps to sequence $\mathbf{y} = (y_1, \dots, y_U) \in \Omega^U$ via operator $\mathcal{B}$.

### 2.1 Deterministic Collapse Definition
The operator $\mathcal{B}$ executes two sequential transformations:
1. **Consecutive Duplicate Deduplication**:

$$\mathcal{B}_{\text{dedup}}(\pi_1, \dots, \pi_T) = (\pi_{t_1}, \pi_{t_2}, \dots) \quad \text{where } \pi_{t_k} \neq \pi_{t_k-1}$$

2. **Blank Removal**:

$$\mathcal{B}(\boldsymbol{\pi}) = [c \in \mathcal{B}_{\text{dedup}}(\boldsymbol{\pi}) \mid c \neq \epsilon]$$

### 2.2 Cardinality of the CTC Preimage $|\mathcal{B}^{-1}(\mathbf{y})|$
Consider a target sequence $\mathbf{y} = (y_1, \dots, y_U)$ of length $U$ decoded across $T$ frames ($T \ge U$).

#### Case A: No Consecutive Identical Tokens ($y_u \neq y_{u+1} \forall u$)
When no adjacent characters are identical, blank tokens are optional. The number of valid alignments corresponds to distributing $T$ frames across the $U$ labels and possible intermediate blanks:

$$|\mathcal{B}^{-1}(\mathbf{y})| = \sum_{k=0}^{T-U} \binom{T}{k} \binom{T-k-1}{U-1}$$

#### Case B: Consecutive Identical Tokens ($y_u = y_{u+1}$)
If $y_u = y_{u+1}$, the alignment **must** contain at least one blank token $\epsilon$ between them:

$$\pi_{t_a} = y_u, \quad \pi_{t_b} = \epsilon, \quad \pi_{t_c} = y_{u+1} \quad (t_a < t_b < t_c)$$

Without this intermediate blank, the collapse operator would merge the identical characters: $\mathcal{B}(a, a) = a \neq (a, a)$. This constraint strictly reduces the cardinality of $\mathcal{B}^{-1}(\mathbf{y})$.

---

## 3. Acoustic Temperature Scaling and Posterior Entropy

Let $\mathbf{z}_t \in \mathbb{R}^{|V|}$ denote the raw acoustic logit vector at frame $t$. Under temperature scaling with $T_{\text{calib}} > 0$:

$$P(\pi_t = k \mid \mathbf{X}; T_{\text{calib}}) = y_k(T_{\text{calib}}) = \frac{\exp(z_{t,k} / T_{\text{calib}})}{\sum_{j=1}^{|V|} \exp(z_{t,j} / T_{\text{calib}})}$$

### 3.1 First Derivative with Respect to Temperature
To determine how temperature alters token probabilities, we differentiate $y_k$ with respect to $T_{\text{calib}}$:

$$\frac{\partial y_k}{\partial T_{\text{calib}}} = \frac{\partial}{\partial T_{\text{calib}}} \left[ \frac{\exp(z_k / T_{\text{calib}})}{\sum_j \exp(z_j / T_{\text{calib}})} \right]$$

Using the quotient rule:

$$\frac{\partial y_k}{\partial T_{\text{calib}}} = \frac{-\frac{z_k}{T_{\text{calib}}^2} \exp(z_k / T_{\text{calib}}) \sum_j \exp(z_j / T_{\text{calib}}) - \exp(z_k / T_{\text{calib}}) \sum_j \left( -\frac{z_j}{T_{\text{calib}}^2} \exp(z_j / T_{\text{calib}}) \right)}{\left( \sum_j \exp(z_j / T_{\text{calib}}) \right)^2}$$

Factoring out $y_k$ and $\frac{1}{T_{\text{calib}}^2}$:

$$\frac{\partial y_k}{\partial T_{\text{calib}}} = -\frac{y_k}{T_{\text{calib}}^2} \left[ z_k - \sum_{j=1}^{|V|} y_j z_j \right] = -\frac{y_k}{T_{\text{calib}}^2} (z_k - \bar{z})$$

where $\bar{z} = \mathbb{E}_{y}[z]$ is the expected logit value under the current distribution.

### 3.2 Implication for Peaked Distributions:
- If token $k$ is the dominant argmax prediction, $z_k > \bar{z} \implies \frac{\partial y_k}{\partial T_{\text{calib}}} < 0$. Increasing temperature **decreases** overconfident peak probabilities.
- For non-dominant tokens where $z_j < \bar{z} \implies \frac{\partial y_j}{\partial T_{\text{calib}}} > 0$. Increasing temperature **boosts** low-probability competing candidates.
- The Shannon entropy $H(Y) = -\sum_k y_k \log y_k$ is a strictly increasing function of $T_{\text{calib}}$ for $T_{\text{calib}} \ge 1.0$.

---

## 4. Log-Linear Shallow Fusion and Brevity Bias Derivation

The decoding search objective finds the word sequence $\hat{\mathbf{w}}$ maximizing:

$$\mathcal{S}(\mathbf{w}) = \log P_{\text{CTC}}(\mathbf{w} \mid \mathbf{X}) + \alpha \sum_{k=1}^K \log P_{\text{LM}}(w_k \mid w_{\lt k}) + \beta K$$

### 4.1 Proof of Brevity Bias
Let $\mathbf{w}_A$ be a hypothesis of length $K_A$ and $\mathbf{w}_B$ be an extended hypothesis of length $K_B = K_A + M$ ($M \ge 1$).

Assume the acoustic model provides neutral evidence across the extension:

$$\log P_{\text{CTC}}(\mathbf{w}_B \mid \mathbf{X}) \approx \log P_{\text{CTC}}(\mathbf{w}_A \mid \mathbf{X})$$

Evaluating the score difference without length compensation ($\beta = 0$):

$$\Delta \mathcal{S} = \mathcal{S}(\mathbf{w}_B) - \mathcal{S}(\mathbf{w}_A) = \alpha \sum_{j=1}^M \log P_{\text{LM}}(w_{K_A + j} \mid w_{\lt K_A + j})$$

Because $0 < P_{\text{LM}}(\cdot) \le 1$, every log-probability term is non-positive:

$$\log P_{\text{LM}}(w \mid \cdot) \le 0 \implies \Delta \mathcal{S} \le 0$$

Without a compensatory term $\beta > 0$, **any hypothesis that emits more words is strictly penalized by the language model**, causing beam search to systematically truncate sentences. Setting:

$$\beta > \alpha \cdot \mathbb{E}\left[ -\log P_{\text{LM}}(w \mid w_{\text{context}}) \right]$$

counterbalances the average cross-entropy of the language model, restoring unbiased sequence length generation.
