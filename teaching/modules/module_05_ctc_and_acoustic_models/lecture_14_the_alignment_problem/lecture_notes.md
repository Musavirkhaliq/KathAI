# Lecture 14: The Alignment Problem in Speech Recognition

---

## 1. The Core Temporal Asymmetry of Speech

The fundamental challenge in automatic speech recognition stems from a dimensional and temporal mismatch:
- An input acoustic feature sequence $\mathbf{X} = (\mathbf{x}_1, \dots, \mathbf{x}_T)$ has continuous duration $T$ (e.g., 500 frames for 10 seconds of audio).
- The corresponding target transcription $\mathbf{Y} = (y_1, \dots, y_U)$ is a sequence of discrete symbols of length $U$ (e.g., 45 characters).
- Crucially, $T \gg U$ (typically $T / U \approx 5 \text{ to } 15$).

The **Alignment Problem** asks: *Which acoustic frame $\mathbf{x}_t$ corresponds to which linguistic character $y_u$?*

In supervised learning with standard cross-entropy loss, each input vector $\mathbf{x}_t$ requires an explicit ground-truth target label $y_t$. In real-world speech, however, human transcribers provide only the sentence-level text, not frame-by-frame phonetic timestamps.

---

## 2. Historical Approaches: Dynamic Time Warping and HMM Forced Alignment

### A. Dynamic Time Warping (DTW)
Used in template-matching ASR: evaluates non-linear alignment between two temporal sequences by finding the path through an $N \times M$ distance matrix minimizing cumulative Euclidean or cosine distance:

$$D(i, j) = d(\mathbf{x}_i, \mathbf{y}_j) + \min \begin{cases} D(i-1, j) \\ D(i, j-1) \\ D(i-1, j-1) \end{cases}$$

- **Fatal limitation**: Requires pre-recorded exemplar templates of each word and cannot generalize across speakers or continuous vocabularies.

### B. HMM Forced Alignment (Viterbi Alignment)
Classical hybrid systems resolve alignment via an iterative Expectation-Maximization (EM) algorithm:
1. Initialize GMM-HMM acoustic models with uniform flat alignment.
2. Run Viterbi decoding over constrained phonetic graphs to compute the most likely frame-to-state alignment:

$$\hat{\mathbf{s}} = \arg\max_{\mathbf{s}} P(\mathbf{X}, \mathbf{s} \mid \mathbf{W})$$

3. Re-train the acoustic model on the newly discovered frame labels.
- **Why this fails for Kashmiri**: Forced alignment requires an accurate pronunciation dictionary (lexicon) mapping every word to phoneme sequences. In Kashmiri, non-standardized orthography and vowel elision cause forced aligners to fail, discarding up to 40% of training data as unalignable.

---

## 3. The End-to-End Solution: Marginalizing over All Valid Alignments

Rather than attempting to guess a single "correct" hard alignment, Connectionist Temporal Classification (CTC) mathematically **integrates (marginalizes) over the entire space of all permissible alignments**:

$$P(\mathbf{Y} \mid \mathbf{X}) = \sum_{\boldsymbol{\pi} \in \mathcal{B}^{-1}(\mathbf{Y})} P(\boldsymbol{\pi} \mid \mathbf{X})$$

By considering every possible way the character sequence $\mathbf{Y}$ could be stretched and interspersed with blanks across $T$ frames, the neural network learns to align speech autonomously through gradient descent.
