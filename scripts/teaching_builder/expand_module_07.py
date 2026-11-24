#!/usr/bin/env python3
"""
expand_module_07.py
Expands Lectures 21, 22, and 23 into exhaustive, textbook-grade educational materials:
- Lecture 21: Decoding Strategies: Greedy Search vs. Beam Search
- Lecture 22: N-Gram Language Modeling, Modified Kneser-Ney, and KenLM
- Lecture 23: Acoustic Temperature Calibration and Shallow Fusion Decoding
"""

from pathlib import Path

BASE_DIR = Path("/home/musa/Kashmiri Project/teaching/modules/module_07_decoding_and_language_models")

# ==============================================================================
# LECTURE 21: GREEDY AND BEAM SEARCH DECODING
# ==============================================================================
def expand_lecture_21():
    ldir = BASE_DIR / "lecture_21_greedy_and_beam_search"
    ldir.mkdir(parents=True, exist_ok=True)

    notes = r"""# Lecture 21: Decoding Strategies: Greedy Search vs. Beam Search

---

## 1. The Optimal Path Search Problem

At every acoustic frame $t \in \{1, \dots, T\}$, the acoustic model outputs a posterior probability distribution over the vocabulary $\Omega' = \Omega \cup \{\epsilon\}$:

$$\mathbf{y}_t = \text{Softmax}(\mathbf{z}_t) \in \Delta^{|\Omega'|-1}$$

The exact maximum a posteriori (MAP) decoding problem seeks the target text sequence $\mathbf{w}^*$ that maximizes the sum of probabilities over all valid alignment paths:

$$\mathbf{w}^* = \arg\max_{\mathbf{w}} P(\mathbf{w} \mid \mathbf{X}) = \arg\max_{\mathbf{w}} \sum_{\boldsymbol{\pi} \in \mathcal{B}^{-1}(\mathbf{w})} \prod_{t=1}^T y_{\pi_t}^t$$

Because the number of possible alignments grows exponentially with sequence length ($|\Omega'|^T \approx 47^{500} \approx 10^{835}$), exact global search is computationally intractable.

---

## 2. Greedy Search and the "Greedy Fallacy"

**Greedy Search (Best Path Decoding)** selects the single highest-probability token at each time step independently, and then collapses the resulting path:

$$\hat{\boldsymbol{\pi}} = (\arg\max_{k} y_k^1, \arg\max_k y_k^2, \dots, \arg\max_k y_k^T)$$
$$\mathbf{w}_{\text{greedy}} = \mathcal{B}(\hat{\boldsymbol{\pi}})$$

### The Greedy Fallacy:
Greedy search operates under the flawed assumption that:
$$\mathcal{B}\left( \arg\max_{\boldsymbol{\pi}} P(\boldsymbol{\pi} \mid \mathbf{X}) \right) \stackrel{?}{=} \arg\max_{\mathbf{w}} \sum_{\boldsymbol{\pi} \in \mathcal{B}^{-1}(\mathbf{w})} P(\boldsymbol{\pi} \mid \mathbf{X})$$

This equality fails routinely in practice.
Consider a toy example where $T = 2$:
- Path $A_1 = (\text{ك}, \text{ٲ})$ has $P = 0.40 \times 0.40 = 0.16 \implies \mathcal{B} = \text{"کٲ"}$
- Path $B_1 = (\text{ت}, \epsilon)$ has $P = 0.35 \times 0.35 = 0.1225 \implies \mathcal{B} = \text{"ت"}$
- Path $B_2 = (\epsilon, \text{ت})$ has $P = 0.35 \times 0.35 = 0.1225 \implies \mathcal{B} = \text{"ت"}$
- Path $B_3 = (\text{ت}, \text{ت})$ has $P = 0.35 \times 0.35 = 0.1225 \implies \mathcal{B} = \text{"ت"}$

Greedy search outputs $\hat{\boldsymbol{\pi}} = (\text{ك}, \text{ٲ})$ with path probability $0.16$.
However, the total marginalized probability of `"ت"` is:
$$P(\text{"ت"}) = P(B_1) + P(B_2) + P(B_3) = 0.1225 + 0.1225 + 0.1225 = \mathbf{0.3675}$$
The true optimal sequence is `"ت"` ($36.75\%$), but greedy search emitted `"کٲ"` ($16\%$)!
In Kashmiri ASR, greedy search results in an **artificially elevated WER (typically 4–6% higher than beam search)**.

---

## 3. CTC Prefix Beam Search Algorithm

**Prefix Beam Search** maintains a pruned set of $W$ highest-scoring text prefixes $\ell$, merging paths that collapse to the same prefix before pruning.

To handle consecutive identical characters correctly, each prefix $\ell$ maintains two distinct probability accumulators:
1. $P_b(\ell, t)$: Total probability mass of all paths collapsing to $\ell$ that **end in a blank token $\epsilon$** at time $t$.
2. $P_{nb}(\ell, t)$: Total probability mass of all paths collapsing to $\ell$ that **end in a non-blank character** at time $t$.

The total probability of prefix $\ell$ at time $t$ is:
$$P(\ell, t) = P_b(\ell, t) + P_{nb}(\ell, t)$$

At each time step $t$, every prefix $\ell$ in the beam is extended by all candidate characters $c \in \Omega'$:
- If extending by blank $\epsilon$:
  $$P_b(\ell, t) = P(\ell, t-1) \cdot y_\epsilon^t$$
- If extending by character $c$:
  - If $c = \text{last}(\ell)$:
    $$P_{nb}(\ell, t) = P_{nb}(\ell, t-1) \cdot y_c^t \quad (\text{repeating identical character without blank})$$
    $$P_{nb}(\ell \circ c, t) = P_b(\ell, t-1) \cdot y_c^t \quad (\text{new character emission following blank})$$
  - If $c \neq \text{last}(\ell)$:
    $$P_{nb}(\ell \circ c, t) = P(\ell, t-1) \cdot y_c^t$$

After accumulating probabilities across all extensions, only the top $W$ prefixes (typically $W = 64 \text{ to } 100$) are retained.
"""
    (ldir / "lecture_notes.md").write_text(notes, encoding="utf-8")

    script = r"""# Instructor Delivery Script: Lecture 21 (Greedy vs. Beam Search Decoding)

## Timetable (90 Minutes)
```
00:00 - 00:20 | The MAP Decoding Paradox: Why Greedy Argmax Fails
00:20 - 00:50 | Mathematical Counterexample: The Multi-Path Marginalization Trap
00:50 - 01:15 | Step-by-Step Walkthrough: Prefix Beam Search (P_b and P_nb)
01:15 - 01:25 | Code Walkthrough: src/decoding/beam_search.py
01:25 - 01:30 | Summary & Homework Assignment
```

---

## Instructor Notes & Dialogue

### 00:20 - 00:50 | The Board Counterexample
- Write the probabilities on the board:
  - Frame 1: $P(\text{ك}) = 0.40, P(\text{ت}) = 0.35, P(\epsilon) = 0.25$
  - Frame 2: $P(\text{ٲ}) = 0.40, P(\text{ت}) = 0.35, P(\epsilon) = 0.25$
- Look at the class:
  > "What does greedy search pick?
  > Frame 1: 'ك' (40%). Frame 2: 'ٲ' (40%).
  > Greedy output: 'کٲ' with probability 16%.
  > Now calculate the probability of the single letter 'ت':
  > It could be 'ت' followed by blank (12.25%).
  > It could be blank followed by 'ت' (12.25%).
  > It could be 'ت' followed by 'ت' which collapses to 'ت' (12.25%).
  > Total probability of 'ت': 36.75%!
  > 'ت' is more than twice as likely as 'کٲ'!
  > But greedy search was blinded by local argmax and missed it completely.
  > This is why beam search with prefix path merging is mathematically mandatory."
"""
    (ldir / "instructor_script.md").write_text(script, encoding="utf-8")

    math_doc = r"""# Mathematical Foundations: Prefix Beam Search and Path Compression

---

## 1. Prefix State Transition Equations

Let $\ell$ be a prefix string over alphabet $\Omega$, and $c \in \Omega$ be a character emission.
The complete dynamic update equations for time frame $t$ given frame emission probabilities $\mathbf{y}^t$ are:

### Blank Extension:
$$P_b(\ell, t) = \left( P_b(\ell, t-1) + P_{nb}(\ell, t-1) \right) y_\epsilon^t$$

### Non-Blank Extensions:
For each symbol $c \in \Omega$:

$$\ell^+ = \ell \circ c$$

If $c = \text{end}(\ell)$:
$$P_{nb}(\ell, t) \mathrel{+}= P_{nb}(\ell, t-1) \cdot y_c^t$$
$$P_{nb}(\ell^+, t) \mathrel{+}= P_b(\ell, t-1) \cdot y_c^t$$

If $c \neq \text{end}(\ell)$:
$$P_{nb}(\ell^+, t) \mathrel{+}= \left( P_b(\ell, t-1) + P_{nb}(\ell, t-1) \right) y_c^t$$

---

## 2. Pruning Complexity
For beam width $W$ and vocabulary size $|\Omega|$:
- Number of candidate prefix extensions per frame: $W \times (|\Omega| + 1)$.
- Sorting/Selection via max-heap or partial selection: $\mathcal{O}(W \cdot |\Omega| + W \log W)$.
- Overall time complexity for sequence $T$: $\mathcal{O}(T \cdot W \cdot |\Omega|)$.
"""
    (ldir / "mathematical_foundations.md").write_text(math_doc, encoding="utf-8")

    code_doc = r"""# Code Walkthrough: Prefix Beam Search (`src/decoding/beam_search.py`)

---

## 1. Pure Python Prefix Beam Search Implementation

```python
from collections import defaultdict
import numpy as np

def ctc_prefix_beam_search(probs, vocab, beam_width=64, blank_idx=0):
    # probs: [T, vocab_size]
    T, V = probs.shape
    
    # beam: map prefix -> (p_blank, p_non_blank)
    beam = {"": (1.0, 0.0)}
    
    for t in range(T):
        next_beam = defaultdict(lambda: (0.0, 0.0))
        
        for prefix, (p_b, p_nb) in beam.items():
            p_total = p_b + p_nb
            
            # 1. Extend with blank
            p_b_curr, p_nb_curr = next_beam[prefix]
            next_beam[prefix] = (p_b_curr + p_total * probs[t, blank_idx], p_nb_curr)
            
            # 2. Extend with non-blank characters
            for c_idx in range(V):
                if c_idx == blank_idx:
                    continue
                c = vocab[c_idx]
                p_c = probs[t, c_idx]
                
                if len(prefix) > 0 and prefix[-1] == c:
                    # Repeating character: p_nb extends same prefix, p_b creates new prefix
                    p_b_ex, p_nb_ex = next_beam[prefix]
                    next_beam[prefix] = (p_b_ex, p_nb_ex + p_nb * p_c)
                    
                    p_b_new, p_nb_new = next_beam[prefix + c]
                    next_beam[prefix + c] = (p_b_new, p_nb_new + p_b * p_c)
                else:
                    new_prefix = prefix + c
                    p_b_new, p_nb_new = next_beam[new_prefix]
                    next_beam[new_prefix] = (p_b_new, p_nb_new + p_total * p_c)
                    
        # Prune to top W prefixes
        sorted_prefixes = sorted(
            next_beam.keys(),
            key=lambda p: next_beam[p][0] + next_beam[p][1],
            reverse=True
        )
        beam = {p: next_beam[p] for p in sorted_prefixes[:beam_width]}
        
    best_prefix = max(beam.keys(), key=lambda p: beam[p][0] + beam[p][1])
    return best_prefix
```
"""
    (ldir / "code_walkthrough.md").write_text(code_doc, encoding="utf-8")

    lab_doc = r"""# Practical Lab: Beam Search vs. Greedy Search Benchmark

---

## Lab Objectives
1. Implement Prefix Beam Search with path merging.
2. Construct synthetic logits where greedy decoding yields an erroneous sequence while beam search finds the true mode.
3. Measure decoding runtime across beam widths $W \in \{1, 10, 50, 100\}$.

---

## Lab Script: `lab_beam_benchmark.py`

```python
import numpy as np
from src.decoding.beam_search import ctc_prefix_beam_search

def run_beam_lab():
    vocab = {0: "<blank>", 1: "ك", 2: "ٲ", 3: "ت"}
    
    # Synthetic logits where greedy fails
    # Frame 0: k=0.40, t=0.35, eps=0.25
    # Frame 1: a=0.40, t=0.35, eps=0.25
    probs = np.array([
        [0.25, 0.40, 0.00, 0.35],
        [0.25, 0.00, 0.40, 0.35]
    ])
    
    # Greedy picks [1, 2] -> "كٲ"
    greedy_tokens = np.argmax(probs, axis=-1)
    greedy_str = "".join([vocab[t] for t in greedy_tokens if t != 0])
    print(f"Greedy Transcription: '{greedy_str}'")
    
    # Beam search should pick "ت"
    beam_str = ctc_prefix_beam_search(probs, vocab, beam_width=10, blank_idx=0)
    print(f"Beam Search Transcription: '{beam_str}'")
    
    assert beam_str == "ت", f"Expected 'ت', got '{beam_str}'"
    print("SUCCESS: Beam search successfully overcame greedy fallacy!")

if __name__ == "__main__":
    run_beam_lab()
```
"""
    (ldir / "practical_lab.md").write_text(lab_doc, encoding="utf-8")

    ex_doc = r"""# Exercises: Lecture 21 (Greedy vs. Beam Search)

---

## Exercise 1: Prefix Path Merging
Given prefix $\ell = \text{"ک"}$, with $P_b(\ell) = 0.30$ and $P_{nb}(\ell) = 0.20$.
At time step $t$, the emissions are:
$y_\epsilon^t = 0.40$, $y_{\text{ک}}^t = 0.30$, $y_{\text{ٲ}}^t = 0.20$.
1. Calculate the updated probabilities $P_b(\text{"ک"})$ and $P_{nb}(\text{"ک"})$.
2. Calculate the updated probability $P_{nb}(\text{"کک"})$.
3. Calculate the updated probability $P_{nb}(\text{"کٲ"})$.

---

## Exercise 2: Beam Width Sensitivity
In an ASR benchmark across 500 Kashmiri sentences:
- Greedy ($W=1$): $\text{WER} = 29.8\%$
- Beam $W=10$: $\text{WER} = 26.4\%$
- Beam $W=64$: $\text{WER} = 24.8\%$
- Beam $W=200$: $\text{WER} = 24.7\%$
Plot the WER vs. Beam Width curve and identify the optimal operational sweet spot balancing accuracy and real-time latency.
"""
    (ldir / "exercises.md").write_text(ex_doc, encoding="utf-8")

    sol_doc = r"""# Solutions: Lecture 21 (Greedy vs. Beam Search)

---

## Solution to Exercise 1
Total prior probability: $P_{\text{total}} = P_b + P_{nb} = 0.30 + 0.20 = 0.50$.
1. For prefix $\text{"ک"}$:
   - Blank extension: $P_b(\text{"ک"}) = P_{\text{total}} \times y_\epsilon = 0.50 \times 0.40 = \mathbf{0.20}$.
   - Non-blank identical character extension:
     $$P_{nb}(\text{"ک"}) = P_{nb}(\text{"ک"}) \times y_{\text{ک}} = 0.20 \times 0.30 = \mathbf{0.06}$$
2. For new prefix $\text{"کک"}$ (identical character following blank):
   $$P_{nb}(\text{"کک"}) = P_b(\text{"ک"}) \times y_{\text{ک}} = 0.30 \times 0.30 = \mathbf{0.09}$$
3. For new prefix $\text{"کٲ"}$ (distinct character):
   $$P_{nb}(\text{"کٲ"}) = P_{\text{total}} \times y_{\text{ٲ}} = 0.50 \times 0.20 = \mathbf{0.10}$$

---

## Solution to Exercise 2
- Going from $W=1$ to $W=10$ drops WER by **3.4% absolute**.
- Going from $W=10$ to $W=64$ drops WER by **1.6% absolute**.
- Going from $W=64$ to $W=200$ drops WER by only **0.1% absolute**, while increasing CPU decoding time by over $300\%$.
- **Optimal Sweet Spot**: **$W = 64$ to $100$**, capturing 99.6% of maximum accuracy within strict real-time constraints.
"""
    (ldir / "solutions.md").write_text(sol_doc, encoding="utf-8")

    quiz_doc = r"""# Quiz: Lecture 21 (Greedy vs. Beam Search)

---

### Question 1:
What is the 'greedy fallacy' in CTC decoding?
- [A] The model uses too much RAM.
- [B] The sequence of locally argmax frame emissions does not necessarily equal the sequence with the highest total marginalized probability mass across all collapsing paths.
- [C] The model emits words in reverse order.
- [D] Greedy search is non-deterministic.

**Correct Answer**: **[B]**  
*Explanation*: Multi-path probability accumulation often favors candidate sequences whose individual paths are not the argmax.

---

### Question 2:
Why does CTC prefix beam search track two separate probabilities ($P_b$ and $P_{nb}$) for each prefix?
- [A] To compute standard deviation.
- [B] To correctly differentiate between sustaining a character vs. emitting a consecutive identical character separated by a blank.
- [C] For mixed precision compatibility.
- [D] To store audio sampling rate.

**Correct Answer**: **[B]**  
*Explanation*: Tracking $P_b$ and $P_{nb}$ prevents identical adjacent characters from erroneously collapsing.

---

### Question 3:
What is the standard beam width ($W$) used in EduCasheer production pipelines?
- [A] 1
- [B] 4
- [C] 64 to 100
- [D] 10,000

**Correct Answer**: **[C]**  
*Explanation*: $W=64-100$ provides optimal balance between error rate reduction and decoding throughput.

---

### Question 4:
What computational complexity does prefix beam search have with respect to beam width $W$ and vocabulary size $|\Omega|$ per frame?
- [A] $\mathcal{O}(1)$
- [B] $\mathcal{O}(W \cdot |\Omega|)$
- [C] $\mathcal{O}(W^2 \cdot |\Omega|^2)$
- [D] $\mathcal{O}(2^W)$

**Correct Answer**: **[B]**  
*Explanation*: Each of the $W$ prefixes generates $|\Omega|$ candidate symbol extensions.

---

### Question 5:
Approximately how much absolute WER improvement does beam search typically yield over greedy decoding in Kashmiri ASR?
- [A] 0.1%
- [B] 4% to 6% absolute
- [C] 25% absolute
- [D] Greedy is better than beam search.

**Correct Answer**: **[B]**  
*Explanation*: Beam search consistently recovers 4–6% WER by exploring multi-path marginalization and language model candidates.
"""
    (ldir / "quiz.md").write_text(quiz_doc, encoding="utf-8")

    res_doc = r"""# Research Connections: Lecture 21

---

## 1. Repository Grounding
- **`src/decoding/beam_search.py`**: Production implementation of CTC prefix beam search.
- **`stages/stage_21/`**: Research stage comparing greedy decoding against KenLM beam search across all 8 evaluation domains.
"""
    (ldir / "research_connections.md").write_text(res_doc, encoding="utf-8")

    refs_doc = r"""# Annotated References: Lecture 21

---

1. **Hannun, A. Y., Maas, A. L., Jurafsky, D., & Ng, A. Y. (2014).**  
   *First-pass large vocabulary continuous speech recognition using bi-directional recurrent DNNs.*  
   *arXiv preprint arXiv:1408.2876*.  
   - **Annotation**: The authoritative paper formalizing prefix beam search with language model integration.
"""
    (ldir / "references.md").write_text(refs_doc, encoding="utf-8")
    print("Expanded Lecture 21 successfully!")

# ==============================================================================
# LECTURE 22: N-GRAM LANGUAGE MODELS AND KENLM
# ==============================================================================
def expand_lecture_22():
    ldir = BASE_DIR / "lecture_22_ngram_language_models_kenlm"
    ldir.mkdir(parents=True, exist_ok=True)

    notes = r"""# Lecture 22: N-Gram Language Modeling, Modified Kneser-Ney, and KenLM

---

## 1. The Linguistic Function of the Language Model in ASR

While the acoustic model computes posterior emissions based purely on audio physics, speech contains inherent acoustic ambiguities:
- **Homophones**: Words that sound identical but have different spellings and meanings.
- **Acoustic Masking**: Low-energy vowels or unstressed syllables obscured by room noise.
- **Grammatical Invariants**: Agreement between subject and verb in Kashmiri's strict verb-second (V2) syntax.

The **Language Model (LM)** provides the statistical prior $P(\mathbf{w})$:
$$P(w_1, w_2, \dots, w_U) = \prod_{u=1}^U P(w_u \mid w_1, \dots, w_{u-1})$$

In an $n$-gram model, the Markov assumption restricts context history to the preceding $n-1$ words:
$$P(w_u \mid w_{<u}) \approx P(w_u \mid w_{u-n+1}, \dots, w_{u-1})$$

---

## 2. Smoothing Algorithms: From Jelinek-Mercer to Modified Kneser-Ney

In low-resource Kashmiri text corpora, the vast majority of valid word sequences never appear in training text. A naive Maximum Likelihood Estimate (MLE) assigns zero probability to unseen sequences ($P_{\text{MLE}} = 0$), causing the decoder to discard valid acoustic hypotheses.

### Modified Kneser-Ney Smoothing (The Gold Standard)
Kneser-Ney smoothing introduces a profound linguistic insight: **continuation probability**.
Instead of asking *"How often does word $w$ appear?"*, it asks:
*"In how many distinct contexts does word $w$ appear as a novel continuation?"*

The lower-order continuation probability is:
$$P_{\text{continuation}}(w) = \frac{|\{ w' : C(w', w) > 0 \}|}{\sum_{w''} |\{ w' : C(w', w'') > 0 \}|}$$

**Modified Kneser-Ney** applies **three distinct discount parameters** ($D_1, D_2, D_{3+}$) based on whether an $n$-gram count is 1, 2, or $\ge 3$:

$$P_{\text{MKN}}(w_i \mid w_{i-n+1}^{i-1}) = \frac{C(w_{i-n+1}^i) - D(C(w_{i-n+1}^i))}{C(w_{i-n+1}^{i-1})} + \gamma(w_{i-n+1}^{i-1}) P_{\text{MKN}}(w_i \mid w_{i-n+2}^{i-1})$$

where:
$$D(c) = \begin{cases} 
0 & \text{if } c = 0 \\
D_1 & \text{if } c = 1 \\
D_2 & \text{if } c = 2 \\
D_3 & \text{if } c \ge 3
\end{cases}$$
and $\gamma(\cdot)$ is the backoff normalization weight ensuring probabilities sum to $1.0$.

---

## 3. High-Performance C++ Inference with KenLM

During beam search decoding, the language model is queried hundreds of thousands of times per second. Python implementations are far too slow for real-time speech.

EduCasheer integrates **KenLM** (Heafield, 2011), an industrial C++ library utilizing:
1. **Sorted Probing / Trie Data Structures**: Minimizes memory cache misses.
2. **Quantized Probability Storage**: Compresses $n$-gram backoff weights into 8-bit quantized bins.
3. **Sub-Microsecond Query Latency**: Evaluates $P(w \mid \text{context})$ in $< 200 \text{ nanoseconds}$, allowing beam search to explore large beams without frame latency.
"""
    (ldir / "lecture_notes.md").write_text(notes, encoding="utf-8")

    script = r"""# Instructor Delivery Script: Lecture 22 (N-Gram Language Models and KenLM)

## Timetable (90 Minutes)
```
00:00 - 00:20 | The Prior: Why Acoustics Alone Cannot Disambiguate Homophones
00:20 - 00:50 | The San Francisco Example: Deriving Kneser-Ney Continuation Probabilities
00:50 - 01:15 | Modified Kneser-Ney Three-Tier Discounting Mathematics
01:15 - 01:25 | KenLM C++ Trie Architecture and Binary Probing
01:25 - 01:30 | Synthesis & Practical Commands
```

---

## Instructor Notes & Dialogue

### 00:20 - 00:50 | The Continuation Probability Story
- Write on the board:
  - `"I want to read a book about ________"`
  - Candidate 1: `"Kashmir"`
  - Candidate 2: `"Francisco"`
- Look at the students:
  > "Which word is more likely in English?
  > In a raw text corpus, the word 'Francisco' might appear 100,000 times!
  > Does that mean 'book about Francisco' is likely?
  > No! Because 'Francisco' almost exclusively appears after one single word: 'San'!
  > It has high frequency, but almost zero continuation versatility!
  > Standard MLE smoothing says: 'Francisco is frequent, so give it high backoff probability.'
  > Kneser-Ney says: 'Nonsense! Francisco only has one predecessor. It is not a versatile continuation word.'
  > In Kashmiri, this prevents proper noun fragments from polluting acoustic backoff paths."
"""
    (ldir / "instructor_script.md").write_text(script, encoding="utf-8")

    math_doc = r"""# Mathematical Foundations: Modified Kneser-Ney Discount Formulations

---

## 1. Closed-Form Estimates for Modified Kneser-Ney Discounts

Let $n_k$ denote the total number of $n$-grams with count $k$ in the training corpus:
$$Y = \frac{n_1}{n_1 + 2 n_2}$$

The three discount parameters are computed as:
$$D_1 = 1 - 2 Y \frac{n_2}{n_1}$$
$$D_2 = 2 - 3 Y \frac{n_3}{n_2}$$
$$D_3 = 3 - 4 Y \frac{n_4}{n_3}$$

The backoff weight $\gamma(w_{i-n+1}^{i-1})$ is:
$$\gamma(w_{i-n+1}^{i-1}) = \frac{D_1 N_1(w_{i-n+1}^{i-1}) + D_2 N_2(w_{i-n+1}^{i-1}) + D_3 N_{3+}(w_{i-n+1}^{i-1})}{C(w_{i-n+1}^{i-1})}$$
where $N_k(h) = |\{ w : C(h, w) = k \}|$ is the number of distinct words following context history $h$ with frequency $k$.
"""
    (ldir / "mathematical_foundations.md").write_text(math_doc, encoding="utf-8")

    code_doc = r"""# Code Walkthrough: KenLM Model Training and Querying

---

## 1. Building a 5-Gram Kashmiri KenLM Model via Shell CLI

```bash
# 1. Prepare normalized text corpus
python3 src/normalization/kashmiri_normalizer.py text_raw.txt > text_normalized.txt

# 2. Train ARPA language model using KenLM lmplz with Modified Kneser-Ney
lmplz -o 5 \
      --text text_normalized.txt \
      --arpa kashmiri_5gram.arpa \
      --discount_fallback \
      --prune 0 0 1 2 2

# 3. Compile ARPA into fast binary trie format
build_binary trie kashmiri_5gram.arpa kashmiri_5gram.binary
```

---

## 2. Python KenLM Query Interface

```python
import kenlm

class KashmiriLanguageModel:
    def __init__(self, binary_path: str):
        self.model = kenlm.Model(binary_path)
        
    def score_sentence(self, sentence: str) -> float:
        # Returns log10 probability
        return self.model.score(sentence, bos=True, eos=True)
        
    def get_word_prob(self, word: str, context: str) -> float:
        state = kenlm.State()
        self.model.BeginSentenceWrite(state)
        # Advance state with context
        for w in context.split():
            next_state = kenlm.State()
            self.model.BaseScore(state, w, next_state)
            state = next_state
            
        out_state = kenlm.State()
        log10_p = self.model.BaseScore(state, word, out_state)
        return log10_p
```
"""
    (ldir / "code_walkthrough.md").write_text(code_doc, encoding="utf-8")

    lab_doc = r"""# Practical Lab: Training and Evaluating a KenLM Language Model

---

## Lab Objectives
1. Train an n-gram language model on normalized Kashmiri text.
2. Calculate corpus perplexity ($PPL$).
3. Compare unigram continuation probabilities vs. raw frequency counts.

---

## Lab Script: `lab_lm_evaluation.py`

```python
import numpy as np

def compute_perplexity(log10_scores, total_words):
    # PPL = 10 ^ (-1/N * sum(log10 P))
    cross_entropy = -sum(log10_scores) / total_words
    ppl = 10 ** cross_entropy
    return ppl

# Test on simulated sentence scores
simulated_log10_scores = [-4.2, -3.1, -5.8, -2.4] # 4 sentences
word_counts = [5, 4, 7, 3] # 19 total words

ppl = compute_perplexity(simulated_log10_scores, sum(word_counts))
print(f"Evaluated Test Perplexity (PPL): {ppl:.2f}")
assert ppl > 1.0, "Perplexity calculation error!"
print("SUCCESS: Language model perplexity calculation validated!")

if __name__ == "__main__":
    pass
```
"""
    (ldir / "practical_lab.md").write_text(lab_doc, encoding="utf-8")

    ex_doc = r"""# Exercises: Lecture 22 (N-Gram Language Models and KenLM)

---

## Exercise 1: Continuation Probability Calculation
A toy training corpus has the following 2-gram counts:
- `("سان", "فرانسسکو")`: count = 500
- `("بٔڑ", "فرانسسکو")`: count = 0
- Total distinct contexts where `"فرانسسکو"` appears: $|\{ w' : C(w', \text{"فرانسسکو"}) > 0 \}| = 1$.
- Total distinct 2-gram pairs in the corpus: 20,000.
1. Calculate the Kneser-Ney continuation probability $P_{\text{continuation}}(\text{"فرانسسکو"})$.
2. If total words in the corpus = 100,000, calculate the standard unigram MLE probability $P_{\text{MLE}}(\text{"فرانسسکو"})$.
3. Explain why continuation probability prevents spurious backoff hallucinations.

---

## Exercise 2: Perplexity Calculation
A 5-gram language model evaluates a test corpus of 500 words and assigns a cumulative log-likelihood:
$$\sum_{i=1}^{500} \log_{10} P(w_i \mid w_{<i}) = -1,250.0$$
1. Calculate the cross-entropy per word.
2. Calculate the test perplexity (PPL).
"""
    (ldir / "exercises.md").write_text(ex_doc, encoding="utf-8")

    sol_doc = r"""# Solutions: Lecture 22 (N-Gram Language Models and KenLM)

---

## Solution to Exercise 1
1. Continuation probability:
   $$P_{\text{continuation}}(\text{"فرانسسکو"}) = \frac{1}{20,000} = \mathbf{0.00005 \text{ (5 \times 10^{-5})}}$$
2. Unigram MLE probability:
   $$P_{\text{MLE}}(\text{"فرانسسکو"}) = \frac{500}{100,000} = \mathbf{0.0050 \text{ (5 \times 10^{-3})}}$$
3. Notice that $P_{\text{MLE}}$ is **100 times higher** than $P_{\text{continuation}}$! In an unseen context (e.g., after the Kashmiri word for "mountain"), standard MLE would assign a dangerously high backoff probability to "Francisco" simply because it was frequent, whereas Kneser-Ney penalizes it because it lacks context diversity.

---

## Solution to Exercise 2
1. Cross-entropy:
   $$H = -\frac{1}{500} \sum_{i=1}^{500} \log_{10} P(w_i) = -\frac{-1,250.0}{500} = \mathbf{2.50}$$
2. Perplexity:
   $$\text{PPL} = 10^H = 10^{2.50} = \mathbf{316.23}$$
"""
    (ldir / "solutions.md").write_text(sol_doc, encoding="utf-8")

    quiz_doc = r"""# Quiz: Lecture 22 (N-Gram Language Models and KenLM)

---

### Question 1:
What fundamental linguistic question is addressed by Kneser-Ney continuation probability?
- [A] How loud the speaker's voice was.
- [B] In how many distinct linguistic contexts a word appears as a novel completion, rather than simply how frequent it is overall.
- [C] How many letters are in the alphabet.
- [D] The duration of the audio clip.

**Correct Answer**: **[B]**  
*Explanation*: Continuation probability measures contextual versatility, avoiding high backoff weights for words that only follow single specific contexts.

---

### Question 2:
Why does Modified Kneser-Ney use three separate discount parameters ($D_1, D_2, D_3$)?
- [A] To run on 3 GPUs.
- [B] Because $n$-grams with low frequency (count 1 or 2) require different discount rates than high-frequency $n$-grams to maximize likelihood on held-out text.
- [C] To support Arabic, Persian, and Urdu.
- [D] To reduce vocabulary size.

**Correct Answer**: **[B]**  
*Explanation*: Three-tier discounting provides optimal smoothing across the frequency spectrum.

---

### Question 3:
What data structure allows KenLM to evaluate language model scores in under 200 nanoseconds?
- [A] Python dictionaries
- [B] Sorted binary tries and probing hash tables optimized for CPU L3 cache lines
- [C] SQL database tables
- [D] Excel spreadsheets

**Correct Answer**: **[B]**  
*Explanation*: KenLM's binary trie minimizes memory overhead and CPU cache misses.

---

### Question 4:
What metric evaluates language model performance on unseen text independently of ASR?
- [A] Word Error Rate (WER)
- [B] Perplexity (PPL)
- [C] Signal-to-Noise Ratio (SNR)
- [D] Receptive Field

**Correct Answer**: **[B]**  
*Explanation*: Perplexity ($10^H$) measures the effective branching factor of the language model.

---

### Question 5:
What text preprocessing must be applied to the language model training corpus?
- [A] It must be converted to uppercase English.
- [B] It must undergo the exact same Certified Normalizer v3 transformation applied to the acoustic transcripts.
- [C] All vowels must be deleted.
- [D] Text must be stored in binary float32 format.

**Correct Answer**: **[B]**  
*Explanation*: Text normalization must be identical across AM and LM to prevent vocabulary mismatches during shallow fusion.
"""
    (ldir / "quiz.md").write_text(quiz_doc, encoding="utf-8")

    res_doc = r"""# Research Connections: Lecture 22

---

## 1. Repository Grounding
- **`src/decoding/beam_search.py`**: Interfacing with the KenLM binary model.
- **`stages/stage_22/`**: Training the 5-gram Kashmiri KenLM model on 1.2M sentences of crawled and normalized literature.
"""
    (ldir / "research_connections.md").write_text(res_doc, encoding="utf-8")

    refs_doc = r"""# Annotated References: Lecture 22

---

1. **Heafield, K. (2011).**  
   *KenLM: Faster and smaller language model queries.*  
   *Proceedings of the Sixth Workshop on Statistical Machine Translation*, 187–197.  
   - **Annotation**: Introduces the binary trie and probing data structures powering KenLM.

2. **Chen, S. F., & Goodman, J. (1999).**  
   *An empirical study of smoothing techniques for language modeling.*  
   *Computer Speech & Language*, 13(4), 359–394.  
   - **Annotation**: The definitive benchmark proving the mathematical superiority of Modified Kneser-Ney smoothing.
"""
    (ldir / "references.md").write_text(refs_doc, encoding="utf-8")
    print("Expanded Lecture 22 successfully!")

# ==============================================================================
# LECTURE 23: ACOUSTIC TEMPERATURE AND SHALLOW FUSION
# ==============================================================================
def expand_lecture_23():
    ldir = BASE_DIR / "lecture_23_acoustic_temperature_and_shallow_fusion"
    ldir.mkdir(parents=True, exist_ok=True)

    notes = r"""# Lecture 23: Acoustic Temperature Calibration and Shallow Fusion Decoding

---

## 1. The Log-Linear Shallow Fusion Framework

During inference, decoding searches for the hypothesis $\hat{\mathbf{w}}$ maximizing the composite log-linear shallow fusion score:

$$\hat{\mathbf{w}} = \arg\max_{\mathbf{w}} \left( \log P_{\text{CTC}}(\mathbf{w} \mid \mathbf{X}, T) + \alpha \log P_{\text{LM}}(\mathbf{w}) + \beta |\mathbf{w}| \right)$$

This objective balances three distinct mathematical forces:
1. **$\log P_{\text{CTC}}(\mathbf{w} \mid \mathbf{X}, T)$**: Acoustic evidence conditioned on calibrated temperature $T$.
2. **$\alpha \log P_{\text{LM}}(\mathbf{w})$**: Syntactic prior weighted by language model weight $\alpha \in [0.1, 1.0]$.
3. **$\beta |\mathbf{w}|$**: Word insertion penalty compensating for character-level brevity bias ($\beta \in [0.5, 2.5]$).

---

## 2. Acoustic Temperature Calibration ($T = 1.08$)

In deep neural networks trained with cross-entropy or CTC loss, output distributions suffer from **overconfidence**:
- Logits are excessively large, causing Softmax probabilities to concentrate into near $1.0$ or $0.0$ values.
- In beam search, overconfident CTC probabilities prematurely prune valid linguistic branches before the language model has an opportunity to evaluate their syntactic validity.

### Temperature Scaling Formulation:
Logits $\mathbf{z}_t$ are calibrated by scalar temperature parameter $T > 1.0$:

$$y_k^t(T) = \frac{\exp(z_k^t / T)}{\sum_{j=1}^{|\Omega'|} \exp(z_j^t / T)}$$

- In Stage 23 of EduCasheer, temperature was optimized via negative log-likelihood (NLL) minimization on the disjoint validation split, yielding optimal **$T = 1.08$**.
- Setting $T = 1.08$ softens emission distributions, preserving candidate diversity in the beam search lattice and unlocking optimal shallow fusion synergy.

---

## 3. Grid Search Optimization of $\alpha$ and $\beta$

The parameters $\alpha$ (LM weight) and $\beta$ (Word insertion penalty) cannot be tuned on the test set without severe contamination. They must be calibrated via a **2D Grid Search on the held-out validation set**:

```
Alpha Sweep: [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
Beta Sweep:  [0.50, 0.80, 1.00, 1.20, 1.50, 2.00]
```

### The Two Historic Operating Regimes in the Repository:
1. **Configuration A (Balanced)**: $\alpha = 0.30, \beta = 1.00, T = 1.08, \text{beam} = 64$.
   - Yields balanced performance across clean and conversational audio.
2. **Configuration B (Aggressive LM)**: $\alpha = 0.50, \beta = 1.50, T = 1.08, \text{beam} = 100$.
   - Optimizes formal broadcast news and Quranic commentary, but slightly over-corrects colloquial names in spontaneous dialects.
   - Used by the certified `E22-01` benchmark to achieve **24.12% disjoint WER**.
"""
    (ldir / "lecture_notes.md").write_text(notes, encoding="utf-8")

    script = r"""# Instructor Delivery Script: Lecture 23 (Acoustic Temperature and Shallow Fusion)

## Timetable (90 Minutes)
```
00:00 - 00:20 | The Tug-of-War: Acoustic Likelihood vs. Language Prior
00:20 - 00:45 | Overconfidence and Temperature Scaling Calibration (T = 1.08)
00:45 - 01:10 | The Brevity Bias and the Mathematical Role of Beta
01:10 - 01:25 | 2D Grid Search Execution on Validation Sets
01:25 - 01:30 | Module 07 Synthesis & Wrap-up
```

---

## Instructor Notes & Dialogue

### 00:20 - 00:45 | The Overconfidence Demonstration
- Plot on the board: Softmax output without temperature vs. with temperature ($T=1.08$).
- Address the students:
  > "Why does an acoustic model act like an arrogant know-it-all?
  > Because modern deep neural networks have millions of parameters and zero incentive to admit uncertainty.
  > If a model is 99.9% confident that it heard 'cat' when the true word was 'cot', beam search prunes 'cot' immediately.
  > The language model never even gets to see 'cot'!
  > Dividing by $T = 1.08$ is a dose of humility for your neural network.
  > It softens 0.999 down to 0.92, keeping alternate phonetic paths alive in the beam so the language model can save the transcription."
"""
    (ldir / "instructor_script.md").write_text(script, encoding="utf-8")

    math_doc = r"""# Mathematical Foundations: Temperature Scaling and Brevity Bias Derivations

---

## 1. Temperature Calibration Optimization

Let validation set contain $N$ frames with ground-truth alignment labels $y_t^*$.
Optimal temperature $T^*$ minimizes negative log-likelihood (NLL):

$$T^* = \arg\min_T -\sum_{t=1}^N \log \left( \frac{\exp(z_{y_t^*}^t / T)}{\sum_k \exp(z_k^t / T)} \right)$$

Solved via 1D line search (Brent's method) or gradient descent over scalar $T$.

---

## 2. Derivation of the Brevity Bias

In character-level decoding, a word of length $L$ characters requires $L$ emission transitions.
Because each transition probability $P(c) < 1.0$, log-probabilities are negative ($\log P < 0$).
Consequently, longer words accumulate larger negative penalties:
$$\log P_{\text{AM}}(\text{long word}) \ll \log P_{\text{AM}}(\text{short word})$$

Without compensation, the beam search systematically favors deleting words or choosing shorter erroneous synonyms.
Adding $\beta |\mathbf{w}|$ adds a positive reward for each completed word, mathematically counteracting acoustic deletion bias.
"""
    (ldir / "mathematical_foundations.md").write_text(math_doc, encoding="utf-8")

    code_doc = r"""# Code Walkthrough: 2D Grid Search for Alpha and Beta

---

## 1. Grid Search Implementation (`src/decoding/tune_hyperparams.py`)

```python
import numpy as np

def run_grid_search(val_samples, decoder, alpha_range, beta_range):
    best_wer = float('inf')
    best_params = (None, None)
    
    results = {}
    for alpha in alpha_range:
        for beta in beta_range:
            decoder.set_hyperparams(alpha=alpha, beta=beta)
            total_wer = 0.0
            
            for audio, ref_text in val_samples:
                hyp = decoder.decode(audio)
                total_wer += compute_wer(ref_text, hyp)
                
            avg_wer = total_wer / len(val_samples)
            results[(alpha, beta)] = avg_wer
            
            if avg_wer < best_wer:
                best_wer = avg_wer
                best_params = (alpha, beta)
                
    return best_params, best_wer
```
"""
    (ldir / "code_walkthrough.md").write_text(code_doc, encoding="utf-8")

    lab_doc = r"""# Practical Lab: Grid Searching Alpha, Beta, and Temperature

---

## Lab Objectives
1. Implement temperature scaling over raw CTC logits.
2. Execute a 2D grid search over $\alpha \in [0.1, 0.6]$ and $\beta \in [0.5, 2.0]$.
3. Identify the global minimum WER configuration on a toy validation split.

---

## Lab Script: `lab_tune_decoding.py`

```python
import numpy as np

def run_tuning_lab():
    alphas = [0.1, 0.3, 0.5]
    betas = [0.5, 1.0, 1.5]
    
    # Simulate a convex WER surface
    # Minimum at alpha = 0.3, beta = 1.0
    best_wer = 999.0
    best_cfg = None
    
    for a in alphas:
        for b in betas:
            # Simulated parabolic response surface
            wer = 25.0 + 15.0 * (a - 0.3)**2 + 10.0 * (b - 1.0)**2
            print(f"Alpha: {a:.2f} | Beta: {b:.2f} | Simulated WER: {wer:.2f}%")
            if wer < best_wer:
                best_wer = wer
                best_cfg = (a, b)
                
    print(f"\nOptimal Configuration: Alpha={best_cfg[0]}, Beta={best_cfg[1]} (WER = {best_wer:.2f}%)")
    assert best_cfg == (0.3, 1.0), "Grid search failed to find optimal point!"
    print("SUCCESS: Decoding hyperparameter tuning validated!")

if __name__ == "__main__":
    run_tuning_lab()
```
"""
    (ldir / "practical_lab.md").write_text(lab_doc, encoding="utf-8")

    ex_doc = r"""# Exercises: Lecture 23 (Acoustic Temperature and Shallow Fusion)

---

## Exercise 1: Temperature Calibration Derivative
Let $y_k(T) = \frac{e^{z_k / T}}{\sum_j e^{z_j / T}}$.
1. Derive the derivative of probability $y_k$ with respect to temperature $T$: $\frac{\partial y_k}{\partial T}$.
2. Show that if $z_k > \bar{z}$ (where $\bar{z} = \sum_j y_j z_j$), increasing temperature decreases the probability of token $k$.

---

## Exercise 2: Word Insertion Penalty Threshold
Given two hypotheses for an ambiguous utterance:
- Hypothesis 1: 2 words, $\log P_{\text{AM}} = -8.0, \log P_{\text{LM}} = -4.0$
- Hypothesis 2: 3 words, $\log P_{\text{AM}} = -10.5, \log P_{\text{LM}} = -4.5$
Let $\alpha = 0.40$.
Calculate the critical threshold $\beta^*$ above which Hypothesis 2 (the longer hypothesis) is selected over Hypothesis 1.
"""
    (ldir / "exercises.md").write_text(ex_doc, encoding="utf-8")

    sol_doc = r"""# Solutions: Lecture 23 (Acoustic Temperature and Shallow Fusion)

---

## Solution to Exercise 1
1. Let $S = \sum_j e^{z_j / T}$.
   $$\frac{\partial y_k}{\partial T} = \frac{\frac{\partial}{\partial T}(e^{z_k / T}) S - e^{z_k / T} \frac{\partial S}{\partial T}}{S^2}$$
   Using $\frac{\partial}{\partial T} e^{z_k / T} = e^{z_k / T} \left( -\frac{z_k}{T^2} \right)$:
   $$\frac{\partial y_k}{\partial T} = \frac{-z_k e^{z_k / T} S - e^{z_k / T} \sum_j (-z_j e^{z_j / T})}{T^2 S^2} = -\frac{y_k}{T^2} \left( z_k - \sum_j y_j z_j \right) = \mathbf{-\frac{y_k}{T^2} (z_k - \bar{z})}$$
2. If $z_k > \bar{z}$, then $(z_k - \bar{z}) > 0$. Since $y_k > 0$ and $T^2 > 0$, the derivative $\frac{\partial y_k}{\partial T} < 0$. Thus, increasing temperature systematically reduces the probability of above-average (overconfident) logits. $\blacksquare$

---

## Solution to Exercise 2
Scores:
$$\mathcal{S}(H_1) = -8.0 + 0.40(-4.0) + 2\beta = -8.0 - 1.6 + 2\beta = -9.6 + 2\beta$$
$$\mathcal{S}(H_2) = -10.5 + 0.40(-4.5) + 3\beta = -10.5 - 1.8 + 3\beta = -12.3 + 3\beta$$
For $H_2$ to win:
$$\mathcal{S}(H_2) > \mathcal{S}(H_1) \implies -12.3 + 3\beta > -9.6 + 2\beta$$
$$\beta > -9.6 + 12.3 \implies \mathbf{\beta^* = 2.70}$$
For $\beta > 2.70$, the insertion bonus overcomes the acoustic penalty and selects the longer word sequence.
"""
    (ldir / "solutions.md").write_text(sol_doc, encoding="utf-8")

    quiz_doc = r"""# Quiz: Lecture 23 (Acoustic Temperature and Shallow Fusion)

---

### Question 1:
What optimal acoustic temperature $T$ was certified for EduCasheer models?
- [A] 0.50
- [B] 1.00
- [C] 1.08
- [D] 5.00

**Correct Answer**: **[C]**  
*Explanation*: $T = 1.08$ was experimentally calibrated via validation NLL minimization.

---

### Question 2:
What is the function of the word insertion penalty ($\beta$)?
- [A] To increase the volume of the audio.
- [B] To compensate for the brevity bias of character-level models by adding a positive score for each completed word, preventing excessive word deletions.
- [C] To penalize bad grammar.
- [D] To slow down decoding.

**Correct Answer**: **[B]**  
*Explanation*: $\beta$ rewards completed words, counteracting acoustic probability decay over long strings.

---

### Question 3:
Why must $\alpha$ and $\beta$ be tuned on the validation set rather than the test set?
- [A] Because the test set is read-only on disk.
- [B] To prevent data contamination and preserve the statistical validity of the final benchmark evaluation.
- [C] Because PyTorch crashes if test sets are used.
- [D] To reduce training loss.

**Correct Answer**: **[B]**  
*Explanation*: Tuning hyperparameters on test data is benchmark contamination.

---

### Question 4:
What are the decoder hyperparameter settings for Configuration B (Aggressive LM)?
- [A] $\alpha = 0.0, \beta = 0.0$
- [B] $\alpha = 0.30, \beta = 1.00$
- [C] $\alpha = 0.50, \beta = 1.50$
- [D] $\alpha = 2.0, \beta = 5.0$

**Correct Answer**: **[C]**  
*Explanation*: Configuration B uses $\alpha = 0.50, \beta = 1.50, \text{beam} = 100$.

---

### Question 5:
What happens during shallow fusion decoding if $\alpha$ is set to zero ($\alpha = 0.0$)?
- [A] The model uses only the acoustic model probabilities and length penalty, operating without external language model guidance.
- [B] The decoder emits only spaces.
- [C] The acoustic temperature resets to 0.
- [D] Audio duration doubles.

**Correct Answer**: **[A]**  
*Explanation*: $\alpha = 0.0$ disables language model scoring completely.
"""
    (ldir / "quiz.md").write_text(quiz_doc, encoding="utf-8")

    res_doc = r"""# Research Connections: Lecture 23

---

## 1. Repository Grounding
- **`src/decoding/beam_search.py`**: Implementation of shallow fusion with calibrated $T$, $\alpha$, and $\beta$.
- **`stages/stage_23/`**: Stage auditing the conflicting WER numbers (24.74% vs. 24.12%), proving that the difference was caused entirely by Config A ($\alpha=0.3, \beta=1.0$) vs. Config B ($\alpha=0.5, \beta=1.5$).
"""
    (ldir / "research_connections.md").write_text(res_doc, encoding="utf-8")

    refs_doc = r"""# Annotated References: Lecture 23

---

1. **Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).**  
   *On calibration of modern neural networks.*  
   *ICML 2017*, 1321–1330.  
   - **Annotation**: Foundational paper introducing temperature scaling for neural network calibration.
"""
    (ldir / "references.md").write_text(refs_doc, encoding="utf-8")
    print("Expanded Lecture 23 successfully!")

if __name__ == "__main__":
    expand_lecture_21()
    expand_lecture_22()
    expand_lecture_23()
