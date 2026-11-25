# Exercises: Pipeline Anatomy and Decoding Formulation

## Problem 1: Conv1d Feature Extractor Receptive Field & Stride
Consider an acoustic model feature extractor consisting of the 7 temporal Conv1d layers specified in the lecture:
- Kernel sizes: $\mathbf{K} = [10, 3, 3, 3, 3, 2, 2]$
- Strides: $\mathbf{S} = [5, 2, 2, 2, 2, 2, 2]$
- Padding: $P = 0$ for all layers
- Audio sampling rate: $f_s = 16,000 \text{ Hz}$

1. Derive the mathematical formula for the receptive field $R_\ell$ at any layer $\ell \in \{1, \dots, 7\}$.
2. Calculate the exact input sample index range $[n_{\text{start}}, n_{\text{end}}]$ that affects the second output frame ($t=2$).
3. Suppose an audio file has exactly $32,150$ samples ($2.009375$ seconds). Calculate the exact number of valid CTC frames produced by the encoder.

---

## Problem 2: CTC Alignment Combinatorics
Let $\Omega = \{\text{ك}, \text{ٲ}, \text{ش}\}$ and $\Omega' = \Omega \cup \{\epsilon\}$.
Consider an acoustic sequence of duration $T=4$ frames.

1. List all valid alignments $\boldsymbol{\pi} \in (\Omega')^4$ in the preimage $\mathcal{B}^{-1}(\mathbf{y})$ for the target sequence $\mathbf{y} = (\text{ك}, \text{ٲ})$.
2. Now consider the target sequence $\mathbf{y}_{\text{dup}} = (\text{ك}, \text{ك})$ with two consecutive identical tokens. List all valid alignments $\boldsymbol{\pi} \in (\Omega')^4$ in $\mathcal{B}^{-1}(\mathbf{y}_{\text{dup}})$.
3. Compare the cardinality $|\mathcal{B}^{-1}(\mathbf{y})|$ versus $|\mathcal{B}^{-1}(\mathbf{y}_{\text{dup}})|$. Explain mathematically why repeated tokens restrict alignment paths.

---

## Problem 3: Shallow Fusion Brevity Bias Calibration
A 5-gram KenLM language model has an average cross-entropy $\bar{H}_{\text{LM}} = 4.2 \text{ nats}$ per word on a Kashmiri test corpus.
A student runs prefix beam search with LM weight $\alpha = 0.50$ and word insertion bonus $\beta = 0.0$.

1. Explain why the student observes that the model transcribes long conversational utterances as only 2 or 3 words.
2. Calculate the minimum theoretical value of $\beta$ required to neutralize the average negative log-probability penalty introduced by the language model.
3. If $\alpha$ is doubled to $1.00$, what adjustment to $\beta$ is required to maintain sequence length balance?

---

## Problem 4: Speaker Leakage Graph Partitioning
An audio collection contains 100 audio files recorded by 10 speakers (10 files per speaker).
A naive researcher performs random 80/20 train/test splitting across the 100 files without considering speaker identity.

1. What is the probability that a specific speaker appears in *both* the training set (80 files) and the test set (20 files)?
2. If all 10 speakers have audio in both sets, explain how the model can achieve a low WER by memorizing speaker acoustic transfer functions rather than learning generalizable speech recognition.
3. Formulate the bipartite graph partitioning algorithm that guarantees $0.0\%$ speaker overlap between train and test.
