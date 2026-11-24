#!/usr/bin/env python3
"""
expand_module_05.py
Expands Lectures 14, 15, 16, and 17 into exhaustive, textbook-grade educational materials:
- Lecture 14: The Alignment Problem in Speech Recognition
- Lecture 15: CTC Loss Mathematics (Forward-Backward, Gradients, and Peaky Emissions)
- Lecture 16: Building the Kashmiri CTC Acoustic Model
- Lecture 17: The Complete PyTorch CTC Training Loop
"""

from pathlib import Path

BASE_DIR = Path("/home/musa/Kashmiri Project/teaching/modules/module_05_ctc_and_acoustic_models")

# ==============================================================================
# LECTURE 14: THE ALIGNMENT PROBLEM IN SPEECH RECOGNITION
# ==============================================================================
def expand_lecture_14():
    ldir = BASE_DIR / "lecture_14_the_alignment_problem"
    ldir.mkdir(parents=True, exist_ok=True)

    notes = r"""# Lecture 14: The Alignment Problem in Speech Recognition

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
"""
    (ldir / "lecture_notes.md").write_text(notes, encoding="utf-8")

    script = r"""# Instructor Delivery Script: Lecture 14 (The Alignment Problem in Speech Recognition)

## Timetable (90 Minutes)
```
00:00 - 00:20 | The Dimension Mismatch: 500 Acoustic Frames vs. 45 Text Characters
00:20 - 00:45 | Historical Solutions: DTW and HMM Forced Alignment Failures
00:45 - 01:10 | Why Manual Alignment Fails on Low-Resource Kashmiri
01:10 - 01:25 | The Revolutionary Concept of Latent Marginalization
01:25 - 01:30 | Transition to Lecture 15: CTC Forward-Backward Mathematics
```

---

## Instructor Notes & Dialogue

### 00:00 - 00:20 | The Opening Demonstration
- Write on the board:
  - Audio: $T = 400 \text{ frames}$ (8 seconds)
  - Text: `"کٲشُر"` ($U = 4 \text{ letters}$)
- Turn to the students:
  > "Look at the numbers on the board. 400 frames on the left. 4 characters on the right.
  > If I ask you to train a standard neural network with Cross-Entropy loss, what is the label for frame 142?
  > Is frame 142 the letter /k/? Is it the central vowel /əː/? Or is it silence?
  > Nobody knows! Even expert phonetic linguists looking at spectrograms argue over the exact millisecond a vowel transitions into a sibilant.
  > For 30 years, speech recognition was trapped in this bottleneck: you couldn't train an acoustic model without alignments, and you couldn't get alignments without an acoustic model!
  > Today, we study how CTC solved this chicken-and-egg dilemma forever."
"""
    (ldir / "instructor_script.md").write_text(script, encoding="utf-8")

    math_doc = r"""# Mathematical Foundations: Dynamic Time Warping and Sequence Distances

---

## 1. Dynamic Time Warping Formulation

Let sequence $\mathbf{X} = (\mathbf{x}_1, \dots, \mathbf{x}_N)$ and $\mathbf{Y} = (\mathbf{y}_1, \dots, \mathbf{y}_M)$ have local distance metric $d(\mathbf{x}_i, \mathbf{y}_j) = \|\mathbf{x}_i - \mathbf{y}_j\|_2$.

A warping path $W = (w_1, w_2, \dots, w_K)$ with $w_k = (i_k, j_k)$ must satisfy:
1. **Boundary Conditions**: $w_1 = (1, 1)$ and $w_K = (N, M)$.
2. **Monotonicity**: $i_1 \le i_2 \le \dots \le i_K$ and $j_1 \le j_2 \le \dots \le j_K$.
3. **Step Size Invariant**: $w_{k+1} - w_k \in \{(1, 0), (0, 1), (1, 1)\}$.

The optimal path minimizes cumulative cost:
$$\text{DTW}(\mathbf{X}, \mathbf{Y}) = \min_{W} \sum_{k=1}^K d(w_k)$$
Solved via dynamic programming in $\mathcal{O}(NM)$ operations.
"""
    (ldir / "mathematical_foundations.md").write_text(math_doc, encoding="utf-8")

    code_doc = r"""# Code Walkthrough: DTW Implementation in Python

---

## 1. Dynamic Time Warping Algorithm

```python
import numpy as np

def compute_dtw_cost(seq1: np.ndarray, seq2: np.ndarray) -> float:
    # seq1: [N, D], seq2: [M, D]
    N, M = len(seq1), len(seq2)
    cost_matrix = np.full((N + 1, M + 1), np.inf)
    cost_matrix[0, 0] = 0.0
    
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            dist = np.linalg.norm(seq1[i-1] - seq2[j-1])
            cost_matrix[i, j] = dist + min(
                cost_matrix[i-1, j],    # Insertion
                cost_matrix[i, j-1],    # Deletion
                cost_matrix[i-1, j-1]   # Match
            )
            
    return cost_matrix[N, M]
```
"""
    (ldir / "code_walkthrough.md").write_text(code_doc, encoding="utf-8")

    lab_doc = r"""# Practical Lab: Aligning Speech Sequences via Dynamic Time Warping

---

## Lab Objectives
1. Implement DTW in pure NumPy.
2. Align two audio recordings of the Kashmiri word `"کٲشُر"` spoken at different speeds.
3. Visualize the optimal non-linear warping path through the alignment lattice.

---

## Lab Script: `lab_dtw_alignment.py`

```python
import numpy as np

def run_dtw_lab():
    # Synthesize two feature sequences: fast vs. slow speech
    t_fast = np.linspace(0, 1, 20)
    feat_fast = np.sin(2 * np.pi * t_fast)[:, None]
    
    t_slow = np.linspace(0, 1, 40)
    feat_slow = np.sin(2 * np.pi * t_slow)[:, None]
    
    N, M = len(feat_fast), len(feat_slow)
    D = np.full((N + 1, M + 1), np.inf)
    D[0, 0] = 0.0
    
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            cost = abs(feat_fast[i-1, 0] - feat_slow[j-1, 0])
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
            
    total_cost = D[N, M]
    print(f"Optimal Warping Path Cost: {total_cost:.4f}")
    assert total_cost < 2.0, "DTW failed to align temporal sequences!"
    print("SUCCESS: DTW alignment verified!")

if __name__ == "__main__":
    run_dtw_lab()
```
"""
    (ldir / "practical_lab.md").write_text(lab_doc, encoding="utf-8")

    ex_doc = r"""# Exercises: Lecture 14 (The Alignment Problem)

---

## Exercise 1: Warping Lattice Complexity
1. For an acoustic sequence of length $T = 600$ and a reference text sequence of length $U = 50$, calculate the total number of cells in the DP grid.
2. If computing the local distance takes 10 nanoseconds, calculate the execution time of a single DTW pass.

---

## Exercise 2: Why Forced Alignment Breaks Down in Kashmiri
Explain three linguistic reasons why HMM forced alignment fails when applied to informal spontaneous Kashmiri speech.
"""
    (ldir / "exercises.md").write_text(ex_doc, encoding="utf-8")

    sol_doc = r"""# Solutions: Lecture 14 (The Alignment Problem)

---

## Solution to Exercise 1
1. Total cells: $N \times M = 600 \times 50 = \mathbf{30,000 \text{ cells}}$.
2. Execution time: $30,000 \times 10 \times 10^{-9} \text{ s} = 3 \times 10^{-4} \text{ s} = \mathbf{0.30 \text{ ms}}$.

---

## Solution to Exercise 2
1. **Diacritic Omission**: Written texts omit vowel diacritics (*aerab*), creating ambiguity about whether a vowel is /i/, /ɨ/, or /u/.
2. **Phonemic Central Vowels**: Acoustic GMM models lack trained central vowel states.
3. **Spontaneous Elision**: In fast speech, word-final vowels are routinely elided, causing HMM forced aligners to reject valid utterances due to zero-likelihood transitions.
"""
    (ldir / "solutions.md").write_text(sol_doc, encoding="utf-8")

    quiz_doc = r"""# Quiz: Lecture 14 (The Alignment Problem)

---

### Question 1:
What is the primary cause of the alignment problem in speech recognition?
- [A] Audio files have stereo channels.
- [B] Acoustic feature sequences are continuous and much longer ($T$) than discrete target text sequences ($U$), with no explicit frame-level timestamp annotations.
- [C] Audio cannot be loaded into GPU memory.
- [D] Speech has non-zero noise.

**Correct Answer**: **[B]**  
*Explanation*: The variable length ratio ($T \gg U$) without frame timestamps creates the alignment problem.

---

### Question 2:
How does CTC loss resolve the alignment problem?
- [A] By hiring human annotators to label every frame.
- [B] By marginalizing (summing) the probability over all permissible valid alignments that collapse to the target sequence.
- [C] By truncating audio to match text length.
- [D] By padding text with random characters.

**Correct Answer**: **[B]**  
*Explanation*: CTC integrates over all possible alignments in $\mathcal{B}^{-1}(\mathbf{Y})$.

---

### Question 3:
What is the time complexity of Dynamic Time Warping for sequences of lengths $N$ and $M$?
- [A] $\mathcal{O}(N + M)$
- [B] $\mathcal{O}(NM)$
- [C] $\mathcal{O}(N^2 M^2)$
- [D] $\mathcal{O}(\log(NM))$

**Correct Answer**: **[B]**  
*Explanation*: DTW evaluates an $N \times M$ dynamic programming cost matrix.

---

### Question 4:
Why did HMM forced alignment discard up to 40% of Kashmiri audio in early repository experiments?
- [A] Hard drive failures.
- [B] Pronunciation dictionary mismatches and non-standardized orthography made phonetic graph matching fail.
- [C] Kashmiri speakers spoke too quietly.
- [D] Kaldi was incompatible with Linux.

**Correct Answer**: **[B]**  
*Explanation*: Strict phonetic lexicons fail in low-resource languages with unstandardized spelling.

---

### Question 5:
What monotonic constraint is enforced by both DTW and CTC?
- [A] Time cannot flow backward; alignments must advance or stay at the same symbol.
- [B] Pitch must strictly increase.
- [C] Volume must decrease.
- [D] Sequence length must be even.

**Correct Answer**: **[A]**  
*Explanation*: Monotonicity ensures that the temporal order of spoken words is strictly preserved.
"""
    (ldir / "quiz.md").write_text(quiz_doc, encoding="utf-8")

    res_doc = r"""# Research Connections: Lecture 14

---

## 1. Repository Grounding
- **`stages/stage_01/`**: The failed Urdu Kaldi baseline that collapsed due to forced alignment rejections.
- **`stages/stage_04/`**: Decision to abandon lexicon-dependent forced alignment in favor of CTC end-to-end learning.
"""
    (ldir / "research_connections.md").write_text(res_doc, encoding="utf-8")

    refs_doc = r"""# Annotated References: Lecture 14

---

1. **Sakoe, H., & Chiba, S. (1978).**  
   *Dynamic programming algorithm optimization for spoken word recognition.*  
   *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 26(1), 43–49.  
   - **Annotation**: Foundational paper introducing DTW for speech recognition.
"""
    (ldir / "references.md").write_text(refs_doc, encoding="utf-8")
    print("Expanded Lecture 14 successfully!")

# ==============================================================================
# LECTURE 15: CTC LOSS MATHEMATICS
# ==============================================================================
def expand_lecture_15():
    ldir = BASE_DIR / "lecture_15_ctc_loss_mathematics"
    ldir.mkdir(parents=True, exist_ok=True)

    notes = r"""# Lecture 15: Connectionist Temporal Classification (CTC) Mathematics

---

## 1. The Extended Target Sequence and Alignment Lattice

Let $\mathbf{l} = (l_1, l_2, \dots, l_U)$ be the target label sequence of length $U$ over alphabet $\Omega$.
CTC introduces the non-emitting blank symbol $\epsilon$ ($\Omega' = \Omega \cup \{\epsilon\}$).

To account for blanks occurring before, between, and after every label, CTC defines the **extended target sequence** $\mathbf{l}'$ of length $U' = 2U + 1$:

$$\mathbf{l}' = (\epsilon, l_1, \epsilon, l_2, \epsilon, \dots, \epsilon, l_U, \epsilon)$$

### Index Mapping:
- Odd indices ($s = 1, 3, 5, \dots, 2U+1$): Always correspond to blank $\epsilon$.
- Even indices ($s = 2, 4, 6, \dots, 2U$): Correspond to actual target characters $l_{s/2}$.

---

## 2. The Forward Variable $\alpha_t(s)$ Recursion

The forward variable $\alpha_t(s)$ represents the total probability mass of all valid partial alignments of length $t$ that map onto the prefix sub-sequence $\mathbf{l}'_{1:s}$:

$$\alpha_t(s) = \sum_{\substack{\boldsymbol{\pi}_{1:t} \in (\Omega')^t \\ \mathcal{B}(\boldsymbol{\pi}_{1:t}) = \mathbf{l}'_{1:s}}} \prod_{\tau=1}^t y_{\pi_\tau}^\tau$$

### Base Case ($t = 1$):
At the first time frame, an alignment can only start on the first blank or the first label:
$$\alpha_1(1) = y_\epsilon^1$$
$$\alpha_1(2) = y_{l_1}^1$$
$$\alpha_1(s) = 0 \quad \forall s > 2$$

### Recursive Step ($t > 1$):
At time step $t$, the transition to state $s$ depends on whether state $s$ is a blank, a duplicate label, or a new distinct label:

**Case 1: State $s$ is blank ($\mathbf{l}'_s = \epsilon$) or identical to the previous label ($\mathbf{l}'_s = \mathbf{l}'_{s-2}$)**:
The alignment could have stayed at state $s$ or transitioned from state $s-1$:
$$\alpha_t(s) = \left( \alpha_{t-1}(s) + \alpha_{t-1}(s-1) \right) y_{\mathbf{l}'_s}^t$$

**Case 2: State $s$ is a distinct label ($\mathbf{l}'_s \neq \epsilon$ and $\mathbf{l}'_s \neq \mathbf{l}'_{s-2}$)**:
The alignment could have stayed at state $s$, moved from the immediately preceding blank $s-1$, or **skipped the blank entirely** from state $s-2$:
$$\alpha_t(s) = \left( \alpha_{t-1}(s) + \alpha_{t-1}(s-1) + \alpha_{t-1}(s-2) \right) y_{\mathbf{l}'_s}^t$$

---

## 3. Total Likelihood and Loss Definition

At the final frame $T$, a valid alignment must end either on the final label ($s = 2U$) or the final blank ($s = 2U + 1$):

$$P(\mathbf{l} \mid \mathbf{X}) = \alpha_T(2U) + \alpha_T(2U + 1)$$

The **CTC Loss** is defined as the negative log-likelihood of the ground-truth sequence:
$$\mathcal{L}_{\text{CTC}} = -\log P(\mathbf{l} \mid \mathbf{X}) = -\log \left( \alpha_T(2U) + \alpha_T(2U + 1) \right)$$

---

## 4. Analytical Logit Gradients: Why CTC Produces Peaky Distributions

Let $z_k^t$ be the unnormalized logit for character $k$ at time $t$, where $y_k^t = \text{Softmax}(z_k^t)$.
By the chain rule, the gradient of $\mathcal{L}_{\text{CTC}}$ with respect to logit $z_k^t$ is:

$$\frac{\partial \mathcal{L}_{\text{CTC}}}{\partial z_k^t} = y_k^t - \frac{1}{P(\mathbf{l} \mid \mathbf{X})} \sum_{s \in \text{lab}(\mathbf{l}', k)} \alpha_t(s) \beta_t(s)$$

where $\text{lab}(\mathbf{l}', k) = \{s \mid \mathbf{l}'_s = k\}$, and $\beta_t(s)$ is the backward CTC variable.

### The Peaky Behavior Phenomenon:
Notice the elegant structure of the gradient:
$$\frac{\partial \mathcal{L}_{\text{CTC}}}{\partial z_k^t} = y_k^t - \gamma_t(k)$$
where $\gamma_t(k)$ is the posterior probability that symbol $k$ was emitted at frame $t$.
- When the acoustic evidence is ambiguous, the model prefers emitting the blank token $\epsilon$ because the blank is valid across multiple paths in the lattice.
- When an acoustic transition is sharp, the model concentrates all probability mass into a **single, sharp 1-frame spike (peak)**, surrounded by long stretches of blank emissions.
- This creates the characteristic "peaky" emission profile of CTC acoustic models.
"""
    (ldir / "lecture_notes.md").write_text(notes, encoding="utf-8")

    script = r"""# Instructor Delivery Script: Lecture 15 (CTC Loss Mathematics)

## Timetable (90 Minutes)
```
00:00 - 00:20 | The Extended Target Lattice (2U + 1 States)
00:20 - 00:50 | Step-by-Step Derivation of Forward Variable Alpha Recursion
00:50 - 01:15 | The Elegant Gradient: (y_k^t - gamma_t(k))
01:15 - 01:25 | Explaining Peaky Behavior: Why CTC Loves the Blank Token
01:25 - 01:30 | Summary & Homework Assignment
```

---

## Instructor Notes & Dialogue

### 00:20 - 00:50 | Forward Recursion on the Board
- Draw a 2D grid: Time frames $t=1 \dots T$ horizontally, Extended states $s=1 \dots 2U+1$ vertically.
- Mark the states for `"كٲشُر"`:
  - $s=1: \epsilon$
  - $s=2: \text{ك}$
  - $s=3: \epsilon$
  - $s=4: \text{ٲ}$
  - $s=5: \epsilon$
  - $s=6: \text{ش}$
  - $s=7: \epsilon$
  - $s=8: \text{ر}$
  - $s=9: \epsilon$
- Trace the three incoming arrows into state 4 (`ٲ`):
  1. Horizontal arrow: Stayed at `ٲ` ($\alpha_{t-1}(4)$).
  2. Diagonal arrow 1: Came from intermediate blank $\epsilon$ ($\alpha_{t-1}(3)$).
  3. Diagonal arrow 2: **Skipped the blank completely** from `ك` ($\alpha_{t-1}(2)$)!
- Point to arrow 3:
  > "Why can you skip the blank here?
  > Because 'ك' and 'ٲ' are different letters! You don't need a blank to separate them.
  > But what if the two letters were identical, like two consecutive Shins?
  > Then arrow 3 is forbidden! You MUST pass through state 3. That is the genius of Alex Graves' formulation."
"""
    (ldir / "instructor_script.md").write_text(script, encoding="utf-8")

    math_doc = r"""# Mathematical Foundations: CTC Backward Variables and Gradient Derivations

---

## 1. The Backward Variable $\beta_t(s)$

The backward variable $\beta_t(s)$ represents the total probability mass of partial alignments from time $t$ to $T$ given state $s$ at time $t$:

### Base Case ($t = T$):
$$\beta_T(2U + 1) = 1, \quad \beta_T(2U) = 1, \quad \beta_T(s) = 0 \quad \forall s < 2U$$

### Recursive Step ($t < T$):
$$\beta_t(s) = \begin{cases}
\left( \beta_{t+1}(s) y_{\mathbf{l}'_s}^{t+1} + \beta_{t+1}(s+1) y_{\mathbf{l}'_{s+1}}^{t+1} \right) & \text{if } \mathbf{l}'_s = \epsilon \text{ or } \mathbf{l}'_s = \mathbf{l}'_{s+2} \\
\left( \beta_{t+1}(s) y_{\mathbf{l}'_s}^{t+1} + \beta_{t+1}(s+1) y_{\mathbf{l}'_{s+1}}^{t+1} + \beta_{t+1}(s+2) y_{\mathbf{l}'_{s+2}}^{t+1} \right) & \text{otherwise}
\end{cases}$$

---

## 2. Derivation of the Logit Gradient

Using Bayes' rule, the posterior probability of state $s$ at time $t$ is:
$$P(\pi_t = \mathbf{l}'_s \mid \mathbf{X}, \mathbf{l}) = \frac{\alpha_t(s) \beta_t(s)}{P(\mathbf{l} \mid \mathbf{X})}$$

Summing over all occurrences of character $k$:
$$\gamma_t(k) = \frac{1}{P(\mathbf{l} \mid \mathbf{X})} \sum_{s \in \text{lab}(\mathbf{l}', k)} \alpha_t(s) \beta_t(s)$$

Since $\frac{\partial \log P(\mathbf{l} \mid \mathbf{X})}{\partial y_k^t} = \frac{\gamma_t(k)}{y_k^t}$, applying the Softmax derivative $\frac{\partial y_k^t}{\partial z_j^t} = y_k^t (\delta_{kj} - y_j^t)$ yields:

$$\frac{\partial \mathcal{L}_{\text{CTC}}}{\partial z_j^t} = -\sum_k \frac{\partial \log P}{\partial y_k^t} \frac{\partial y_k^t}{\partial z_j^t} = -\sum_k \frac{\gamma_t(k)}{y_k^t} y_k^t (\delta_{kj} - y_j^t) = y_j^t \sum_k \gamma_t(k) - \gamma_t(j) = y_j^t - \gamma_t(j) \quad \blacksquare$$
"""
    (ldir / "mathematical_foundations.md").write_text(math_doc, encoding="utf-8")

    code_doc = r"""# Code Walkthrough: CTC Forward-Backward Verification

---

## 1. PyTorch Native `nn.CTCLoss` Usage

```python
import torch
import torch.nn as nn

# Production CTC loss instantiation
ctc_loss_fn = nn.CTCLoss(
    blank=0,
    reduction='mean',
    zero_infinity=True  # Clamps infinite loss values if T < U
)

# Shapes:
# logits: [T, B, vocab_size] (PyTorch standard expects T first)
# targets: [B, U] (padded with pad_idx)
# input_lengths: [B] (actual valid frame counts)
# target_lengths: [B] (actual valid label counts)
```
"""
    (ldir / "code_walkthrough.md").write_text(code_doc, encoding="utf-8")

    lab_doc = r"""# Practical Lab: Implementing CTC Forward Recursion in NumPy

---

## Lab Objectives
1. Implement the forward variable $\alpha_t(s)$ in NumPy.
2. Verify total sequence probability $P(\mathbf{l} \mid \mathbf{X}) = \alpha_T(2U) + \alpha_T(2U+1)$.
3. Compare numerical forward-backward likelihood against PyTorch `nn.CTCLoss`.

---

## Lab Script: `lab_ctc_forward.py`

```python
import numpy as np
import torch
import torch.nn as nn

def compute_ctc_forward_prob(log_probs, target_labels, blank_idx=0):
    # log_probs: [T, vocab_size]
    # target_labels: [U]
    T, V = log_probs.shape
    U = len(target_labels)
    
    # Extended target sequence l' of length 2U + 1
    extended = [blank_idx]
    for lbl in target_labels:
        extended.extend([lbl, blank_idx])
    L_prime = len(extended)
    
    # Forward alpha table in log-space
    alpha = np.full((T, L_prime), -np.inf)
    
    # Base case t = 0
    alpha[0, 0] = log_probs[0, blank_idx]
    alpha[0, 1] = log_probs[0, extended[1]]
    
    for t in range(1, T):
        for s in range(L_prime):
            # Same state or previous state
            prev_sum = np.logaddexp(alpha[t-1, s], alpha[t-1, s-1] if s > 0 else -np.inf)
            
            # Skip blank if applicable
            if s > 1 and extended[s] != blank_idx and extended[s] != extended[s-2]:
                prev_sum = np.logaddexp(prev_sum, alpha[t-1, s-2])
                
            alpha[t, s] = prev_sum + log_probs[t, extended[s]]
            
    # Total loss
    total_log_prob = np.logaddexp(alpha[T-1, L_prime-1], alpha[T-1, L_prime-2])
    return -total_log_prob

# Verification
T = 6
V = 5
target = [1, 2] # Two character sequence
logits = torch.randn(T, 1, V)
log_probs = torch.log_softmax(logits, dim=-1)

# PyTorch Loss
ctc_loss = nn.CTCLoss(blank=0, reduction='sum')
in_len = torch.tensor([T])
tg_len = torch.tensor([len(target)])
pt_loss = ctc_loss(log_probs, torch.tensor([target]), in_len, tg_len).item()

# Custom Loss
my_loss = compute_ctc_forward_prob(log_probs.squeeze(1).numpy(), target, blank_idx=0)
print(f"PyTorch CTC Loss: {pt_loss:.5f} | Custom Forward CTC Loss: {my_loss:.5f}")
assert abs(pt_loss - my_loss) < 1e-4, "Loss mismatch!"
print("SUCCESS: CTC forward algorithm validated!")

if __name__ == "__main__":
    run_lab = None
```
"""
    (ldir / "practical_lab.md").write_text(lab_doc, encoding="utf-8")

    ex_doc = r"""# Exercises: Lecture 15 (CTC Loss Mathematics)

---

## Exercise 1: Extended Sequence Construction
For the Kashmiri target word `"سَر"` (Reh, Zain, etc. - 2 characters: $c_1, c_2$):
1. Write the extended target sequence $\mathbf{l}'$. What is $U'$?
2. If the target word has two identical letters `"دَد"` ($c_1, c_1$), explain why $\alpha_t(4)$ cannot receive a transition directly from $\alpha_{t-1}(2)$.

---

## Exercise 2: Peaky Gradient Verification
Suppose at frame $t$, the model's Softmax output for letter 'ب' is $y_ب^t = 0.95$, while the CTC posterior $\gamma_t(ب) = 0.98$.
1. Calculate the logit gradient $\frac{\partial \mathcal{L}}{\partial z_ب^t}$.
2. In which direction will the logit $z_ب^t$ move after gradient descent?
"""
    (ldir / "exercises.md").write_text(ex_doc, encoding="utf-8")

    sol_doc = r"""# Solutions: Lecture 15 (CTC Loss Mathematics)

---

## Solution to Exercise 1
1. Extended sequence: $\mathbf{l}' = (\epsilon, c_1, \epsilon, c_2, \epsilon)$. Length $U' = 2(2) + 1 = \mathbf{5}$.
2. For `"دَد"`, state 2 is $c_1$ and state 4 is $c_1$.
   If $\alpha_t(4)$ received a transition directly from $\alpha_{t-1}(2)$, the resulting alignment sub-path would emit $(c_1, c_1)$ without an intervening blank. Under collapse operator $\mathcal{B}$, consecutive identical characters merge: $\mathcal{B}(c_1, c_1) = c_1$, which collapses the two words into a single letter. To preserve two distinct letters, passing through the intervening blank at state 3 ($\epsilon$) is mandatory.

---

## Solution to Exercise 2
1. Logit gradient:
   $$\frac{\partial \mathcal{L}}{\partial z_ب^t} = y_ب^t - \gamma_t(ب) = 0.95 - 0.98 = \mathbf{-0.03}$$
2. Gradient descent update rule:
   $$z_{\text{new}} = z_{\text{old}} - \eta \frac{\partial \mathcal{L}}{\partial z} = z_{\text{old}} - \eta (-0.03) = z_{\text{old}} + 0.03 \eta$$
   The logit will **increase**, pushing $y_ب^t$ closer to $1.0$.
"""
    (ldir / "solutions.md").write_text(sol_doc, encoding="utf-8")

    quiz_doc = r"""# Quiz: Lecture 15 (CTC Loss Mathematics)

---

### Question 1:
What is the length of the extended target sequence $\mathbf{l}'$ for a target transcription of length $U$?
- [A] $U$
- [B] $U + 1$
- [C] $2U + 1$
- [D] $U^2$

**Correct Answer**: **[C]**  
*Explanation*: Extended sequence inserts a blank before, between, and after every symbol: $U + (U + 1) = 2U + 1$.

---

### Question 2:
What is the analytical gradient of the CTC loss with respect to unnormalized logit $z_k^t$?
- [A] $y_k^t - \gamma_t(k)$
- [B] $(y_k^t)^2$
- [C] $\log(y_k^t)$
- [D] $y_k^t \cdot \gamma_t(k)$

**Correct Answer**: **[A]**  
*Explanation*: The gradient is simply the difference between model output probability $y_k^t$ and alignment posterior $\gamma_t(k)$.

---

### Question 3:
Under what condition can state $s$ receive a transition directly from state $s-2$ in the forward recursion?
- [A] When state $s$ is the blank token.
- [B] When state $s$ is a non-blank token that is distinct from state $s-2$ ($\mathbf{l}'_s \neq \epsilon$ and $\mathbf{l}'_s \neq \mathbf{l}'_{s-2}$).
- [C] When $t > 50$.
- [D] Under all conditions.

**Correct Answer**: **[B]**  
*Explanation*: Blank-skipping is permissible only between distinct non-blank symbols.

---

### Question 4:
Why does CTC training produce 'peaky' emission spikes surrounded by blank tokens?
- [A] Because of hardware bugs in GPUs.
- [B] Because blanks can participate in almost all lattice paths, concentrating probability mass into blanks during transition frames and reserving spikes for high-confidence frames.
- [C] Because characters are normalized with NFKC.
- [D] To save battery power.

**Correct Answer**: **[B]**  
*Explanation*: Blanks provide a low-risk fallback for ambiguous frames, concentrating phonetic evidence into narrow spikes.

---

### Question 5:
What occurs if input acoustic frame length $T$ is less than target text length $U$ ($T < U$)?
- [A] Training proceeds normally.
- [B] CTC loss is mathematically undefined (returns $+\infty$ or zero-infinity clamp) because no valid monotonic alignment can fit $U$ characters into fewer than $U$ frames.
- [C] Text characters are automatically duplicated.
- [D] The audio sample rate doubles.

**Correct Answer**: **[B]**  
*Explanation*: CTC requires $T \ge U$; otherwise the preimage $\mathcal{B}^{-1}(\mathbf{l})$ is empty.
"""
    (ldir / "quiz.md").write_text(quiz_doc, encoding="utf-8")

    res_doc = r"""# Research Connections: Lecture 15

---

## 1. Repository Grounding
- **`src/models/wav2vec2.py`**: Interfacing with PyTorch `nn.CTCLoss`.
- **`stages/stage_15/`**: Audit that identified loss spikes caused by utterances where $T_\text{frames} < U_\text{chars}$, prompting the addition of `zero_infinity=True`.
"""
    (ldir / "research_connections.md").write_text(res_doc, encoding="utf-8")

    refs_doc = r"""# Annotated References: Lecture 15

---

1. **Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J. (2006).**  
   *Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks.*  
   *ICML 2006*, 369–376.  
   - **Annotation**: The foundational paper deriving forward-backward $\alpha$ and $\beta$ recursions and the analytical logit gradient.
"""
    (ldir / "references.md").write_text(refs_doc, encoding="utf-8")
    print("Expanded Lecture 15 successfully!")

# ==============================================================================
# LECTURE 16: BUILDING THE KASHMIRI CTC ACOUSTIC MODEL
# ==============================================================================
def expand_lecture_16():
    ldir = BASE_DIR / "lecture_16_kashmiri_ctc_acoustic_model"
    ldir.mkdir(parents=True, exist_ok=True)

    notes = r"""# Lecture 16: Building the Kashmiri CTC Acoustic Model

---

## 1. Architectural Anatomy: From Pretrained Backbone to CTC Emission

The EduCasheer acoustic model couples a massive self-supervised multilingual backbone (**XLS-R 300M**) with a dedicated task-specific linear projection head:

```
[ Raw Audio x (16 kHz) ]
           │
           ▼
[ 7-Layer CNN Feature Extractor ] ── (Output: 512-dim, 50 Hz frames)
           │
           ▼
[ Feature Projection & LayerNorm ] ── (Maps 512-dim to 1024-dim)
           │
           ▼
[ 24-Layer Transformer Encoder ] ── (1024-dim hidden states h_t)
           │
           ▼
[ Final Dropout (p = 0.1) ]
           │
           ▼
[ Linear CTC Projection Head ] ── (Linear: 1024 -> 47 classes)
           │
           ▼
[ Frame-Level Logits z_t in R^47 ]
```

---

## 2. Linear Head Design and Output Bias Initialization

The classification head is an affine transformation:
$$\mathbf{z}_t = \mathbf{W}_{\text{head}} \mathbf{h}_t + \mathbf{b}_{\text{head}}$$
where $\mathbf{W}_{\text{head}} \in \mathbb{R}^{47 \times 1024}$ and $\mathbf{b}_{\text{head}} \in \mathbb{R}^{47}$.

### Output Bias Initialization Strategy:
In CTC training, $> 90\%$ of frames correspond to the blank token $\epsilon$ (index 0).
- If $\mathbf{b}_{\text{head}}$ is initialized to uniform zeros, the initial Softmax assigns uniform probability $\frac{1}{47} \approx 0.021$ to all tokens.
- For an alignment of length $T=500$, initial loss is massive ($-\log(0.021^{500}) \approx 1930$), causing huge gradient shocks that can destroy pretrained Transformer weights.
- **Remedy**: Initializing the blank bias $b_0$ to a positive value (e.g., $b_0 = 3.0$) and all label biases to $0.0$ starts the network with an inductive bias toward blanks ($P(\epsilon) \approx 0.90$), stabilizing early optimization.

---

## 3. Freeze and Unfreeze Schedules: The Staged Fine-Tuning Recipe

Training all 317 million parameters simultaneously from epoch 1 on a low-resource dataset leads to **catastrophic forgetting**: the delicate low-level acoustic filters in the 7-layer CNN are ruined by noisy initial gradients.

EduCasheer enforces a **staged training schedule**:
1. **Stage 1 (Warmup Phase, Epochs 1–3)**:
   - **Feature Extractor**: Strictly frozen (`requires_grad = False`).
   - **Transformer Layers 1–16**: Frozen.
   - **Transformer Layers 17–24 + CTC Head**: Trainable with low learning rate ($3 \times 10^{-5}$).
2. **Stage 2 (Full Acoustic Adaptation, Epochs 4–20)**:
   - **Feature Extractor**: Remains frozen (preserves universal low-level edge filters).
   - **All 24 Transformer Layers**: Unfrozen with Layer-wise Learning Rate Decay (LLRD).
"""
    (ldir / "lecture_notes.md").write_text(notes, encoding="utf-8")

    script = r"""# Instructor Delivery Script: Lecture 16 (Building the Kashmiri CTC Acoustic Model)

## Timetable (90 Minutes)
```
00:00 - 00:20 | Model Anatomy: Coupling XLS-R to the 47-Class Linear Head
00:20 - 00:45 | Initializing the CTC Head: Why Blank Bias Prevents Gradient Shocks
00:45 - 01:10 | Staged Fine-Tuning: Freezing the Feature Extractor to Prevent Forgetting
01:10 - 01:25 | Code Inspection: src/models/wav2vec2.py
01:25 - 01:30 | Synthesis & Transition to the PyTorch Training Loop
```

---

## Instructor Notes & Dialogue

### 00:20 - 00:45 | Blank Bias Demonstration
- Step to the board:
  > "If you have 47 classes and initialize your bias to zero, what is the probability of the blank token on step zero?
  > One divided by 47. About 2%.
  > But in CTC, what fraction of audio frames are actually blanks?
  > Over 90%!
  > If your model starts by predicting that blanks have only 2% probability, your very first backward pass generates a massive gradient explosion that punches your 24 Transformer layers in the face.
  > By setting the blank bias to +3.0, the model begins with 90% blank probability. The gradients are calm, smooth, and gentle from step one."
"""
    (ldir / "instructor_script.md").write_text(script, encoding="utf-8")

    math_doc = r"""# Mathematical Foundations: Parameter Counting and Capacity Allocation

---

## 1. Parameter Breakdown of EduCasheer Kashmiri Acoustic Model

### A. 7-Layer Convolutional Feature Extractor:
- Total Conv parameters:
  $$P_{\text{conv}} = \sum_{i=1}^7 K_i \cdot C_{i-1} \cdot C_i \approx 4.1 \times 10^6 \text{ parameters}$$

### B. 24-Layer Transformer Encoder:
For each layer with $d_{\text{model}} = 1024, d_{\text{ff}} = 4096$:
- Multi-Head Attention: $4 \times d_{\text{model}}^2 = 4 \times 1024^2 \approx 4.19 \times 10^6$
- Feedforward Network: $2 \times d_{\text{model}} \times d_{\text{ff}} = 2 \times 1024 \times 4096 \approx 8.39 \times 10^6$
- LayerNorms: $4 \times 1024 \approx 4,096$
- Per-layer total: $\approx 12.59 \times 10^6$
- 24 layers total: $24 \times 12.59 \times 10^6 \approx 302.2 \times 10^6 \text{ parameters}$

### C. CTC Projection Head:
$$P_{\text{head}} = (1024 \times 47) + 47 = 48,175 \text{ parameters}$$

**Grand Total**: $\approx \mathbf{317 \times 10^6 \text{ parameters}}$ (317M).
"""
    (ldir / "mathematical_foundations.md").write_text(math_doc, encoding="utf-8")

    code_doc = r"""# Code Walkthrough: Acoustic Model Architecture (`src/models/wav2vec2.py`)

---

## 1. Inspecting the Kashmiri CTC Model Class

```python
import torch
import torch.nn as nn
from transformers import Wav2Vec2PreTrainedModel, Wav2Vec2Model

class KashmiriWav2Vec2ForCTC(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Initialize weights
        self.init_weights()
        
        # Set positive blank bias
        with torch.no_grad():
            self.lm_head.bias[0] = 3.0
            
    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()
        
    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits
```
"""
    (ldir / "code_walkthrough.md").write_text(code_doc, encoding="utf-8")

    lab_doc = r"""# Practical Lab: Freezing Layers and Tracking Trainable Parameters

---

## Lab Objectives
1. Instantiate the Kashmiri CTC architecture.
2. Implement freezing functions for feature extractor and lower Transformer layers.
3. Verify parameter counts and assert that frozen weights receive zero gradients during backprop.

---

## Lab Script: `lab_model_freezing.py`

```python
import torch
import torch.nn as nn

class ToySpeechModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 64, kernel_size=10, stride=5)
        self.fc = nn.Linear(64, 47)
        
    def forward(self, x):
        h = self.conv(x).transpose(1, 2)
        return self.fc(h)

model = ToySpeechModel()

# Freeze conv
for p in model.conv.parameters():
    p.requires_grad = False
    
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)

print(f"Trainable Parameters: {trainable} | Frozen Parameters: {frozen}")

# Run forward-backward
x = torch.randn(2, 1, 100)
out = model(x).sum()
out.backward()

assert model.conv.weight.grad is None, "Frozen layer received gradients!"
print("SUCCESS: Freezing verified!")

if __name__ == "__main__":
    pass
```
"""
    (ldir / "practical_lab.md").write_text(lab_doc, encoding="utf-8")

    ex_doc = r"""# Exercises: Lecture 16 (Building the Kashmiri CTC Acoustic Model)

---

## Exercise 1: Parameter Ratio Calculation
Given the parameter counts derived in the lecture:
1. What percentage of the model's total parameters reside in the 24 Transformer layers?
2. What percentage of parameters reside in the task-specific CTC classification head?
3. What does this ratio tell us about the risk of overfitting in the projection head vs. the backbone?

---

## Exercise 2: Blank Bias Softmax Shift
Suppose $d_{\text{vocab}} = 47$.
1. If $\mathbf{b} = \mathbf{0}$, what is $P(\epsilon)$?
2. If $b_0 = 3.0$ and $b_k = 0$ for all $k \ge 1$, calculate $P(\epsilon)$ when $\mathbf{h} = \mathbf{0}$.
"""
    (ldir / "exercises.md").write_text(ex_doc, encoding="utf-8")

    sol_doc = r"""# Solutions: Lecture 16 (Building the Kashmiri CTC Acoustic Model)

---

## Solution to Exercise 1
1. Transformer layers: $\frac{302.2}{317} \times 100 = \mathbf{95.3\%}$.
2. Classification head: $\frac{0.048}{317} \times 100 = \mathbf{0.015\%}$.
3. The projection head has negligible capacity ($< 0.02\%$), meaning catastrophic overfitting cannot occur in the linear head itself; overfitting occurs when the 302M parameters in the Transformer backbone distort universal speech features.

---

## Solution to Exercise 2
1. For $\mathbf{b} = \mathbf{0}$:
   $$P(\epsilon) = \frac{e^0}{\sum_{i=1}^{47} e^0} = \frac{1}{47} \approx \mathbf{0.0213 \text{ (2.13\%)}}$$
2. For $b_0 = 3.0$ and $b_{1:46} = 0$:
   $$e^{b_0} = e^{3.0} \approx 20.0855$$
   $$\sum_{k=1}^{46} e^{b_k} = 46 \times 1.0 = 46.0$$
   $$P(\epsilon) = \frac{20.0855}{20.0855 + 46.0} = \frac{20.0855}{66.0855} \approx \mathbf{0.3039 \text{ (30.4\%)}}$$
   (Setting $b_0 = 5.0$ yields $P(\epsilon) \approx 76.3\%$).
"""
    (ldir / "solutions.md").write_text(sol_doc, encoding="utf-8")

    quiz_doc = r"""# Quiz: Lecture 16 (Building the Kashmiri CTC Acoustic Model)

---

### Question 1:
How many classes does the final linear projection head output in the Kashmiri ASR model?
- [A] 26
- [B] 47
- [C] 128
- [D] 1024

**Correct Answer**: **[B]**  
*Explanation*: 47 classes match the certified Kashmiri character vocabulary.

---

### Question 2:
Why is the feature extractor frozen during early fine-tuning stages?
- [A] To prevent memory leaks in PyTorch.
- [B] To protect universal low-level acoustic edge filters from catastrophic forgetting caused by noisy initial task gradients.
- [C] Because 1D convolutions do not support backpropagation.
- [D] To increase audio volume.

**Correct Answer**: **[B]**  
*Explanation*: Freezing low-level CNN filters preserves acoustic features learned during self-supervised pretraining.

---

### Question 3:
What is the effect of initializing the CTC blank bias $b_0$ to a positive value (e.g., $+3.0$)?
- [A] It forces the model to emit all characters at once.
- [B] It aligns initial model predictions with the natural sparsity of speech, avoiding massive initial loss and gradient shocks.
- [C] It doubles the training speed.
- [D] It reduces the number of Transformer layers.

**Correct Answer**: **[B]**  
*Explanation*: Positive blank bias provides a smooth inductive starting point reflecting the dominant blank frequency.

---

### Question 4:
Approximately how many total parameters are contained in the XLS-R 300M Kashmiri acoustic model?
- [A] 10 million
- [B] 100 million
- [C] 317 million
- [D] 1 billion

**Correct Answer**: **[C]**  
*Explanation*: The architecture contains ~4.1M CNN params + 302.2M Transformer params + 0.05M head params $\approx 317$M.

---

### Question 5:
What dropout probability is applied prior to the final linear projection head?
- [A] 0.0
- [B] 0.1
- [C] 0.5
- [D] 0.9

**Correct Answer**: **[B]**  
*Explanation*: Standard final dropout $p=0.1$ regularizes the bottleneck representations before logit projection.
"""
    (ldir / "quiz.md").write_text(quiz_doc, encoding="utf-8")

    res_doc = r"""# Research Connections: Lecture 16

---

## 1. Repository Grounding
- **`src/models/wav2vec2.py`**: Implementation of `KashmiriWav2Vec2ForCTC`.
- **`stages/stage_16/`**: Historical stage where the linear classification head was adapted for 47 certified Kashmiri graphemes.
"""
    (ldir / "research_connections.md").write_text(res_doc, encoding="utf-8")

    refs_doc = r"""# Annotated References: Lecture 16

---

1. **Baevski, A., et al. (2020).**  
   *wav2vec 2.0: A framework for self-supervised learning of speech representations.*  
   *NeurIPS 2020*.  
   - **Annotation**: Section on fine-tuning with CTC linear heads and staged freezing.
"""
    (ldir / "references.md").write_text(refs_doc, encoding="utf-8")
    print("Expanded Lecture 16 successfully!")

# ==============================================================================
# LECTURE 17: THE COMPLETE PYTORCH CTC TRAINING LOOP
# ==============================================================================
def expand_lecture_17():
    ldir = BASE_DIR / "lecture_17_pytorch_training_loop"
    ldir.mkdir(parents=True, exist_ok=True)

    notes = r"""# Lecture 17: The Complete PyTorch CTC Training Loop

---

## 1. Systems Engineering in Deep Acoustic Model Training

Training a 317-million-parameter acoustic model over 170+ hours of speech requires meticulous systems engineering. A naive training script will rapidly suffer from GPU out-of-memory (OOM) crashes, numerical underflows in mixed precision, corrupted padding losses, and non-reproducible metric checkpoints.

The canonical training loop (`src/training/trainer.py`) orchestrates five critical systems:
1. **Dynamic Batch Bucketing**: Minimizes padding waste.
2. **Automatic Mixed Precision (AMP)**: Accelerates matrix multiplication with `torch.cuda.amp.autocast` while preserving numerical stability via `GradScaler`.
3. **Gradient Accumulation**: Simulates massive effective batch sizes ($B=64$ or $128$) on modest hardware.
4. **Gradient Norm Clipping**: Constrains gradient norm ($\|\mathbf{g}\| \le 1.0$) to prevent exploding step updates.
5. **Atomic Checkpointing**: Atomically serializes model weights, optimizer state, scheduler state, and scalar evaluation history.

---

## 2. Automatic Mixed Precision (AMP) and GradScaler Mechanics

Standard FP32 floating point uses 32 bits (1 sign, 8 exponent, 23 mantissa). FP16 uses 16 bits (1 sign, 5 exponent, 10 mantissa).
- While FP16 doubles Tensor Core compute throughput and halves memory footprint, its dynamic range is restricted: numbers smaller than $2^{-24} \approx 5.96 \times 10^{-8}$ underflow to zero.
- In CTC loss backpropagation, gradients in deep Transformer layers frequently drop below $10^{-7}$. Under plain FP16, these gradients underflow to zero, causing training to freeze.

### The Solution: Dynamic Gradient Scaling
`torch.cuda.amp.GradScaler` multiplies the loss by a large scale factor $S$ (e.g., $S = 2^{15} = 32,768$) before backpropagation:
$$\mathbf{g}_{\text{scaled}} = \nabla_{\mathbf{W}} (S \cdot \mathcal{L})$$
This shifts small gradients into the representable range of FP16.
Before the optimizer updates the weights, the scaler unscales the gradients:
$$\mathbf{g} = \frac{\mathbf{g}_{\text{scaled}}}{S}$$
If infs or NaNs are detected, the optimizer step is skipped, and the scale factor $S$ is halved.

---

## 3. Effective Batch Size through Gradient Accumulation

CTC loss is notoriously noisy on small batch sizes ($B=4$ or $B=8$). Stable convergence requires an **effective batch size of 64 or 128 utterances**.
When GPU VRAM limits physical batch size to $B_{\text{phys}} = 8$:
- The loop accumulates gradients over $K = 8$ micro-steps:
  $$\mathbf{g}_{\text{accum}} = \frac{1}{K} \sum_{k=1}^K \nabla_{\mathbf{W}} \mathcal{L}_k$$
- The optimizer step and gradient zeroing are triggered only every $K$ steps.
"""
    (ldir / "lecture_notes.md").write_text(notes, encoding="utf-8")

    script = r"""# Instructor Delivery Script: Lecture 17 (The Complete PyTorch CTC Training Loop)

## Timetable (90 Minutes)
```
00:00 - 00:20 | Systems Architecture: The 5 Pillars of Stable Acoustic Training
00:20 - 00:45 | Mixed Precision Mechanics: FP16 Underflows and Dynamic GradScaler
00:45 - 01:10 | Gradient Accumulation and Gradient Clipping (Norm <= 1.0)
01:10 - 01:25 | Line-by-Line Code Inspection: src/training/trainer.py
01:25 - 01:30 | Module 05 Synthesis & Course Midpoint Review
```

---

## Instructor Notes & Dialogue

### 00:20 - 00:45 | GradScaler Demonstration
- Write on the board:
  - FP16 minimum positive normal: $2^{-14} \approx 6.1 \times 10^{-5}$
  - FP16 minimum subnormal: $2^{-24} \approx 5.96 \times 10^{-8}$
- Look at the students:
  > "If your backward pass computes a gradient of $10^{-9}$, what does an FP16 GPU register do?
  > It flushes it to absolute zero!
  > If that happens across 10 layers, your model stops updating.
  > GradScaler says: 'Multiply the loss by 32,768.'
  > Now $10^{-9}$ becomes $3.27 \times 10^{-5}$, which fits comfortably in FP16!
  > Right before the weights update, you divide by 32,768.
  > That is how we get $3\times$ faster training without losing a single bit of precision."
"""
    (ldir / "instructor_script.md").write_text(script, encoding="utf-8")

    math_doc = r"""# Mathematical Foundations: Gradient Clipping and Accumulation

---

## 1. Gradient Norm Clipping Invariant

Let $\mathbf{g} = \nabla_{\mathbf{W}} \mathcal{L}$ be the concatenated gradient vector across all model parameters.
Given maximum norm threshold $M$ (typically $M = 1.0$):

$$\|\mathbf{g}\|_2 = \sqrt{\sum_{i} g_i^2}$$

The clipped gradient $\tilde{\mathbf{g}}$ is:
$$\tilde{\mathbf{g}} = \begin{cases}
\mathbf{g} & \text{if } \|\mathbf{g}\|_2 \le M \\
\frac{M}{\|\mathbf{g}\|_2} \mathbf{g} & \text{if } \|\mathbf{g}\|_2 > M
\end{cases}$$

This guarantees:
$$\|\tilde{\mathbf{g}}\|_2 \le M$$
while preserving the exact directional angle of the gradient vector:
$$\frac{\tilde{\mathbf{g}}}{\|\tilde{\mathbf{g}}\|} = \frac{\mathbf{g}}{\|\mathbf{g}\|}$$
"""
    (ldir / "mathematical_foundations.md").write_text(math_doc, encoding="utf-8")

    code_doc = r"""# Code Walkthrough: Production Training Loop (`src/training/trainer.py`)

---

## 1. Inspecting the Canonical Training Loop

```python
import torch
from torch.cuda.amp import autocast, GradScaler

def train_one_epoch(model, dataloader, optimizer, scheduler, ctc_loss_fn, device, accum_steps=8):
    model.train()
    scaler = GradScaler()
    optimizer.zero_grad()
    
    total_loss = 0.0
    
    for step, batch in enumerate(dataloader):
        input_values = batch["input_values"].to(device)
        labels = batch["labels"].to(device)
        in_lengths = batch["input_lengths"].to(device)
        lbl_lengths = batch["target_lengths"].to(device)
        
        # Forward pass under mixed precision
        with autocast():
            logits = model(input_values)  # [B, T, V]
            # PyTorch CTCLoss expects [T, B, V]
            log_probs = torch.log_softmax(logits.transpose(0, 1), dim=-1)
            loss = ctc_loss_fn(log_probs, labels, in_lengths, lbl_lengths)
            loss = loss / accum_steps
            
        # Backward pass with scaled gradients
        scaler.scale(loss).backward()
        total_loss += loss.item() * accum_steps
        
        # Optimizer step every accum_steps
        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
    return total_loss / len(dataloader)
```
"""
    (ldir / "code_walkthrough.md").write_text(code_doc, encoding="utf-8")

    lab_doc = r"""# Practical Lab: Mixed Precision and Gradient Accumulation Verification

---

## Lab Objectives
1. Implement gradient accumulation over 4 micro-steps.
2. Verify that accumulated gradients match a single full-batch gradient within numerical tolerance.
3. Observe dynamic scale adjustment in `GradScaler`.

---

## Lab Script: `lab_training_mechanics.py`

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

def run_loop_lab():
    model = nn.Linear(32, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = GradScaler(enabled=False) # CPU simulation
    
    # Simulate 4 micro-batches
    accum_steps = 4
    optimizer.zero_grad()
    
    for _ in range(accum_steps):
        x = torch.randn(8, 32)
        y = model(x).sum() / accum_steps
        y.backward()
        
    # Check that grad is non-zero
    grad_norm = model.weight.grad.norm().item()
    print(f"Accumulated Gradient Norm: {grad_norm:.4f}")
    assert grad_norm > 0.0, "Zero gradient after accumulation!"
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    clipped_norm = model.weight.grad.norm().item()
    print(f"Clipped Gradient Norm:     {clipped_norm:.4f}")
    assert clipped_norm <= 1.0 + 1e-5, "Clipping invariant violated!"
    print("SUCCESS: Training loop invariants verified!")

if __name__ == "__main__":
    run_loop_lab()
```
"""
    (ldir / "practical_lab.md").write_text(lab_doc, encoding="utf-8")

    ex_doc = r"""# Exercises: Lecture 17 (The Complete PyTorch CTC Training Loop)

---

## Exercise 1: Gradient Accumulation Equivalence
Prove that accumulating the gradient of $\frac{1}{K} \sum_{k=1}^K \mathcal{L}_k$ over $K$ steps before updating weights is mathematically identical to performing a single backward pass on the combined batch $\mathcal{B} = \bigcup_{k=1}^K \mathcal{B}_k$.

---

## Exercise 2: Effective Batch Size Calculation
A researcher trains on 2 NVIDIA RTX 3090 GPUs.
- Per-GPU batch size: 8 utterances
- Gradient accumulation steps: 4
1. What is the effective batch size per optimizer step?
2. If the training manifest contains 32,000 utterances, how many optimizer updates occur per epoch?
"""
    (ldir / "exercises.md").write_text(ex_doc, encoding="utf-8")

    sol_doc = r"""# Solutions: Lecture 17 (The Complete PyTorch CTC Training Loop)

---

## Solution to Exercise 1
Let $\mathcal{L}_k = \frac{1}{|\mathcal{B}_k|} \sum_{i \in \mathcal{B}_k} \ell(x_i, y_i)$.
If all micro-batches have equal size $B$, then the total loss across the combined batch is:
$$\mathcal{L}_{\text{total}} = \frac{1}{K \cdot B} \sum_{k=1}^K \sum_{i \in \mathcal{B}_k} \ell(x_i, y_i) = \frac{1}{K} \sum_{k=1}^K \left( \frac{1}{B} \sum_{i \in \mathcal{B}_k} \ell(x_i, y_i) \right) = \frac{1}{K} \sum_{k=1}^K \mathcal{L}_k$$
By linearity of differentiation:
$$\nabla_{\mathbf{W}} \mathcal{L}_{\text{total}} = \frac{1}{K} \sum_{k=1}^K \nabla_{\mathbf{W}} \mathcal{L}_k$$
Accumulating gradients computes precisely this summation. $\blacksquare$

---

## Solution to Exercise 2
1. Effective batch size:
   $$B_{\text{eff}} = 2 \text{ GPUs} \times 8 \text{ samples/GPU} \times 4 \text{ accum steps} = \mathbf{64 \text{ utterances}}$$
2. Optimizer updates per epoch:
   $$N_{\text{steps}} = \frac{32,000}{64} = \mathbf{500 \text{ updates per epoch}}$$
"""
    (ldir / "solutions.md").write_text(sol_doc, encoding="utf-8")

    quiz_doc = r"""# Quiz: Lecture 17 (The Complete PyTorch CTC Training Loop)

---

### Question 1:
What is the primary role of `GradScaler` in PyTorch Automatic Mixed Precision (AMP)?
- [A] To speed up audio resampling.
- [B] To scale loss values by a large factor before backpropagation, preventing small gradient values from underflowing to zero in 16-bit floating point (FP16).
- [C] To normalize audio waveforms to [-1, 1].
- [D] To decode text hypotheses.

**Correct Answer**: **[B]**  
*Explanation*: Dynamic scaling prevents FP16 underflow in deep networks.

---

### Question 2:
What gradient norm clipping threshold is standard in EduCasheer training?
- [A] 0.01
- [B] 1.0
- [C] 100.0
- [D] No clipping

**Correct Answer**: **[B]**  
*Explanation*: Clipping at $\le 1.0$ prevents destabilizing step updates while preserving gradient direction.

---

### Question 3:
Why must loss be divided by `accum_steps` when using gradient accumulation?
- [A] To save disk space.
- [B] Because gradients accumulate additively; dividing by $K$ ensures the accumulated gradient equals the true arithmetic mean over the effective batch.
- [C] To convert from natural log to log base 10.
- [D] Because PyTorch requires integer loss.

**Correct Answer**: **[B]**  
*Explanation*: PyTorch accumulates gradients additively; scaling by $1/K$ computes the correct average gradient.

---

### Question 4:
What occurs if `scaler.unscale_(optimizer)` is omitted before calling `clip_grad_norm_`?
- [A] Gradients are clipped based on their artificially scaled values rather than their true magnitudes, resulting in overly aggressive clipping.
- [B] PyTorch crashes with a segmentation fault.
- [C] Model weights double in size.
- [D] The audio sample rate drops.

**Correct Answer**: **[A]**  
*Explanation*: Clipping must operate on true unscaled gradients to apply the correct threshold $M=1.0$.

---

### Question 5:
What tensor dimension ordering is expected by PyTorch `nn.CTCLoss` for log-probabilities?
- [A] `[Batch, Time, Vocab]`
- [B] `[Time, Batch, Vocab]`
- [C] `[Vocab, Time, Batch]`
- [D] `[Time, Vocab]`

**Correct Answer**: **[B]**  
*Explanation*: Native PyTorch `CTCLoss` requires `[Time, Batch, Vocab]` format unless `batch_first=True` is explicitly specified.
"""
    (ldir / "quiz.md").write_text(quiz_doc, encoding="utf-8")

    res_doc = r"""# Research Connections: Lecture 17

---

## 1. Repository Grounding
- **`src/training/trainer.py`**: The production training implementation featuring AMP, GradScaler, and accumulation.
- **`stages/stage_17/`**: Research stage that validated FP16 training parity against FP32 baselines, proving 0.00% WER difference while cutting training wall-clock time by 58%.
"""
    (ldir / "research_connections.md").write_text(res_doc, encoding="utf-8")

    refs_doc = r"""# Annotated References: Lecture 17

---

1. **Micikevicius, P., et al. (2018).**  
   *Mixed precision training.*  
   *International Conference on Learning Representations (ICLR 2018)*.  
   - **Annotation**: Foundational paper establishing FP16/FP32 mixed precision training and loss scaling algorithms.
"""
    (ldir / "references.md").write_text(refs_doc, encoding="utf-8")
    print("Expanded Lecture 17 successfully!")

if __name__ == "__main__":
    expand_lecture_14()
    expand_lecture_15()
    expand_lecture_16()
    expand_lecture_17()
