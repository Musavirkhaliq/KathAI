#!/usr/bin/env python3
"""
expand_module_08.py
Expands Lectures 24, 25, and 26 into exhaustive, textbook-grade educational materials:
- Lecture 24: Levenshtein Distance, Word Error Rate (WER), and Character Error Rate (CER)
- Lecture 25: Scientific Benchmarking: The 8-Domain Kashmiri Evaluation Protocol
- Lecture 26: Statistical Significance, Confidence Intervals, and Bootstrap Resampling
"""

from pathlib import Path

BASE_DIR = Path("/home/musa/Kashmiri Project/teaching/modules/module_08_benchmarking_and_evaluation")

# ==============================================================================
# LECTURE 24: LEVENSHTEIN DISTANCE, WER, AND CER
# ==============================================================================
def expand_lecture_24():
    ldir = BASE_DIR / "lecture_24_levenshtein_distance_wer_cer"
    ldir.mkdir(parents=True, exist_ok=True)

    notes = r"""# Lecture 24: Levenshtein Distance, Word Error Rate (WER), and Character Error Rate (CER)

---

## 1. The Mathematical Definition of Edit Distance

Evaluating speech recognition requires measuring the minimum string editing operations required to transform a model hypothesis sequence $\mathbf{H} = (h_1, \dots, h_M)$ into a ground-truth reference sequence $\mathbf{R} = (r_1, \dots, r_N)$.

The **Levenshtein Distance** $D(N, M)$ defines an optimal alignment using three primitive atomic operations:
1. **Substitution ($S$)**: Replacing reference token $r_i$ with distinct hypothesis token $h_j$.
2. **Deletion ($D$)**: Deleting reference token $r_i$ (omitted in hypothesis).
3. **Insertion ($I$)**: Inserting extra hypothesis token $h_j$ (absent in reference).

### Dynamic Programming Matrix Recursion:
Let $d(i, j)$ be the minimum edit operations to transform prefix $\mathbf{R}_{1:i}$ to $\mathbf{H}_{1:j}$:
$$d(i, 0) = i \quad \forall i \in \{0, \dots, N\} \quad (\text{all deletions})$$
$$d(0, j) = j \quad \forall j \in \{0, \dots, M\} \quad (\text{all insertions})$$

For $i \ge 1, j \ge 1$:
$$d(i, j) = \min \begin{cases}
d(i-1, j) + 1 & (\text{Deletion}) \\
d(i, j-1) + 1 & (\text{Insertion}) \\
d(i-1, j-1) + \mathbb{I}(r_i \neq h_j) & (\text{Match / Substitution})
\end{cases}$$

where $\mathbb{I}(\cdot)$ is the indicator function returning $1$ if $r_i \neq h_j$ and $0$ if $r_i = h_j$.
The execution complexity for sequence lengths $N$ and $M$ is $\mathcal{O}(NM)$.

---

## 2. Word Error Rate (WER) and Character Error Rate (CER)

Given cumulative counts across an evaluation dataset:
- Total Substitutions: $S$
- Total Deletions: $D$
- Total Insertions: $I$
- Total Reference Tokens: $N = S + D + C$ (where $C$ is correct matches)

The standard error rate metric is:
$$\text{Error Rate} = \frac{S + D + I}{N} = \frac{S + D + I}{S + D + C}$$

### Why WER Can Exceed 100%:
Unlike standard classification accuracy bounded in $[0, 1]$, WER has **no upper bound**:
$$\text{WER} \in [0, +\infty)$$
If a model hallucinates an excessively verbose hypothesis (e.g., $N = 5$ reference words, but hypothesis contains $25$ insertions), then:
$$\text{WER} = \frac{0 + 0 + 25}{5} = \frac{25}{5} = \mathbf{500\%}$$

### Micro-Averaged vs. Macro-Averaged WER:
- **Micro-Averaged WER (Scientific Standard)**: Sums all edit operations across the entire test corpus before dividing by total reference words:
  $$\text{WER}_{\text{micro}} = \frac{\sum_{k=1}^K (S_k + D_k + I_k)}{\sum_{k=1}^K N_k}$$
- **Macro-Averaged WER (Flawed Metric)**: Computes WER independently for each utterance and averages the percentages:
  $$\text{WER}_{\text{macro}} = \frac{1}{K} \sum_{k=1}^K \frac{S_k + D_k + I_k}{N_k}$$
  Macro-averaging creates catastrophic distortion: a 1-word utterance with 1 insertion scores $200\%$, skewing the entire benchmark mean. EduCasheer strictly mandates **micro-averaged WER**.
"""
    (ldir / "lecture_notes.md").write_text(notes, encoding="utf-8")

    script = r"""# Instructor Delivery Script: Lecture 24 (Levenshtein Distance, WER, and CER)

## Timetable (90 Minutes)
```
00:00 - 00:20 | The Core Metric: Substitutions, Deletions, and Insertions
00:20 - 00:45 | Step-by-Step Walkthrough of the DP Edit Matrix
00:45 - 01:10 | The Mathematical Danger: Micro vs. Macro Averaging
01:10 - 01:25 | Line-by-Line Code Inspection: src/evaluation/metrics.py
01:25 - 01:30 | Summary & Homework Assignment
```

---

## Instructor Notes & Dialogue

### 00:45 - 01:10 | Micro vs. Macro Averaging on the Board
- Write two test sentences on the board:
  - Sentence 1: Reference = 1 word (`"سلام"`). Hypothesis = `"سلام سلام سلام"` (2 insertions).
    $$\text{WER}_1 = \frac{2}{1} = 200\%$$
  - Sentence 2: Reference = 99 words. Hypothesis = 99 correct words.
    $$\text{WER}_2 = \frac{0}{99} = 0\%$$
- Ask the students:
  > "What is the macro-averaged WER?
  > Average of 200% and 0% is 100%!
  > A paper reports: 'Our model achieved 100% error rate.'
  > But look at the total words: out of 100 words spoken, the model got 99 words completely right and hallucinated two words!
  > The true micro-averaged WER is:
  > $$\frac{2}{1 + 99} = \frac{2}{100} = 2.0\%!$$
  > In this course, if an engineer reports macro-averaged WER, their submission is rejected immediately.
  > Always micro-average across the entire corpus!"
"""
    (ldir / "instructor_script.md").write_text(script, encoding="utf-8")

    math_doc = r"""# Mathematical Foundations: Levenshtein Backtracking and Alignment Paths

---

## 1. Backtracking the Optimal Edit Sequence

The dynamic programming matrix $d(i, j)$ computes the minimal distance. To identify the specific operations, an optimal path is backtracked from cell $(N, M)$ to $(0, 0)$:

$$\text{Next Cell} = \arg\min \begin{cases}
d(i-1, j-1) & (\text{Diagonal: Match or Sub}) \\
d(i-1, j)   & (\text{Up: Deletion}) \\
d(i, j-1)   & (\text{Left: Insertion})
\end{cases}$$

---

## 2. Alignment Matrix Invariants

Let $C, S, D, I$ be the counts extracted from the optimal path:
$$N_{\text{ref}} = C + S + D$$
$$N_{\text{hyp}} = C + S + I$$
$$\text{Path Length } K \le N_{\text{ref}} + N_{\text{hyp}}$$
$$\max(N_{\text{ref}}, N_{\text{hyp}}) \le K \le N_{\text{ref}} + N_{\text{hyp}}$$
"""
    (ldir / "mathematical_foundations.md").write_text(math_doc, encoding="utf-8")

    code_doc = r"""# Code Walkthrough: Production Metric Computation (`src/evaluation/metrics.py`)

---

## 1. Inspecting the Micro-Averaged WER Engine

```python
import numpy as np

def compute_levenshtein_ops(ref_tokens, hyp_tokens):
    # Returns (S, D, I, C)
    N, M = len(ref_tokens), len(hyp_tokens)
    dp = np.zeros((N + 1, M + 1), dtype=int)
    
    for i in range(N + 1):
        dp[i, 0] = i
    for j in range(M + 1):
        dp[0, j] = j
        
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            if ref_tokens[i-1] == hyp_tokens[j-1]:
                dp[i, j] = dp[i-1, j-1]
            else:
                dp[i, j] = 1 + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
                
    # Backtrack
    i, j = N, M
    S, D, I, C = 0, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_tokens[i-1] == hyp_tokens[j-1]:
            C += 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i, j] == dp[i-1, j-1] + 1:
            S += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i, j] == dp[i-1, j] + 1:
            D += 1
            i -= 1
        else:
            I += 1
            j -= 1
            
    return S, D, I, C

def compute_corpus_wer(references, hypotheses):
    tot_S, tot_D, tot_I, tot_N = 0, 0, 0, 0
    for r, h in zip(references, hypotheses):
        r_words = r.split()
        h_words = h.split()
        S, D, I, C = compute_levenshtein_ops(r_words, h_words)
        tot_S += S
        tot_D += D
        tot_I += I
        tot_N += len(r_words)
        
    wer = (tot_S + tot_D + tot_I) / max(1, tot_N)
    return wer, tot_S, tot_D, tot_I, tot_N
```
"""
    (ldir / "code_walkthrough.md").write_text(code_doc, encoding="utf-8")

    lab_doc = r"""# Practical Lab: Dynamic Programming Levenshtein Alignment

---

## Lab Objectives
1. Implement the Levenshtein DP matrix with full backtracking in Python.
2. Align two Kashmiri sentences and visualize the alignment tokens: `Match`, `Sub`, `Del`, `Ins`.
3. Compute micro-averaged and macro-averaged WER to observe skew on synthetic outlier sentences.

---

## Lab Script: `lab_levenshtein.py`

```python
from src.evaluation.metrics import compute_levenshtein_ops, compute_corpus_wer

def run_metric_lab():
    ref = "کٲشُر چھُ اَکھ قٔدیٖم زَبان"
    hyp = "کٲشُر چھُ زَبان" # Deleted 'اَکھ' and 'قٔدیٖم'
    
    S, D, I, C = compute_levenshtein_ops(ref.split(), hyp.split())
    print(f"Substitutions: {S} | Deletions: {D} | Insertions: {I} | Matches: {C}")
    assert D == 2 and S == 0 and I == 0 and C == 3, "Alignment calculation error!"
    
    wer = (S + D + I) / (S + D + C)
    print(f"Calculated Sentence WER: {wer*100:.2f}% (Expected 40.00%)")
    print("SUCCESS: Levenshtein alignment engine validated!")

if __name__ == "__main__":
    run_metric_lab()
```
"""
    (ldir / "practical_lab.md").write_text(lab_doc, encoding="utf-8")

    ex_doc = r"""# Exercises: Lecture 24 (Levenshtein Distance, WER, and CER)

---

## Exercise 1: DP Matrix Evaluation
Given:
- Reference: `R = ("ک", "ٲ", "ش", "ر")`
- Hypothesis: `H = ("ک", "ا", "ش", "ر", "ی")`
1. Construct the complete $5 \times 6$ Levenshtein dynamic programming table.
2. Backtrack from cell $(4, 5)$ to find the optimal sequence of edit operations.
3. Calculate the Character Error Rate (CER).

---

## Exercise 2: Micro vs. Macro Inversion
An evaluation corpus contains two test sentences:
- Utterance A: Reference has 2 words. Hypothesis has 2 substitutions.
- Utterance B: Reference has 48 words. Hypothesis has 48 correct words.
1. Calculate the sentence-level WER for Utterance A and Utterance B.
2. Calculate the Macro-averaged WER across the two utterances.
3. Calculate the Micro-averaged WER.
4. Explain why Macro-averaged WER gives an absurdly misleading impression of model accuracy.
"""
    (ldir / "exercises.md").write_text(ex_doc, encoding="utf-8")

    sol_doc = r"""# Solutions: Lecture 24 (Levenshtein Distance, WER, and CER)

---

## Solution to Exercise 1
1. DP Table:
```
       eps   ک   ا   ش   ر   ی
eps     0    1   2   3   4   5
 ک      1    0   1   2   3   4
 ٲ      2    1   1   2   3   4
 ش      3    2   2   1   2   3
 ر      4    3   3   2   1   2
```
2. Backtracking path:
   - $(4, 5) \to (4, 4)$ with Insertion of `'ی'`
   - $(4, 4) \to (3, 3)$ with Match of `'ر'`
   - $(3, 3) \to (2, 2)$ with Match of `'ش'`
   - $(2, 2) \to (1, 1)$ with Substitution of `'ٲ'` with `'ا'`
   - $(1, 1) \to (0, 0)$ with Match of `'ک'`
   - Total operations: 1 Substitution, 0 Deletions, 1 Insertion, 3 Matches.
3. Character Error Rate:
   $$\text{CER} = \frac{S + D + I}{N} = \frac{1 + 0 + 1}{4} = \frac{2}{4} = \mathbf{50.0\%}$$

---

## Solution to Exercise 2
1. Sentence WERs:
   $$\text{WER}_A = \frac{2}{2} = 100.0\%, \quad \text{WER}_B = \frac{0}{48} = 0.0\%$$
2. Macro-averaged WER:
   $$\text{WER}_{\text{macro}} = \frac{100.0\% + 0.0\%}{2} = \mathbf{50.0\%}$$
3. Micro-averaged WER:
   $$\text{WER}_{\text{micro}} = \frac{2 + 0}{2 + 48} = \frac{2}{50} = \mathbf{4.0\%}$$
4. Explanation: Macro-averaging treats an utterance with 2 words with equal weight to an utterance with 48 words. The model transcribed 48 out of 50 total spoken words correctly ($96\%$ word accuracy), yet macro-averaging reports an erroneous $50\%$ error rate.
"""
    (ldir / "solutions.md").write_text(sol_doc, encoding="utf-8")

    quiz_doc = r"""# Quiz: Lecture 24 (Levenshtein Distance, WER, and CER)

---

### Question 1:
What are the three atomic editing operations measured by the Levenshtein distance?
- [A] Fourier, Wavelet, Mel
- [B] Substitution, Deletion, Insertion
- [C] Convolution, Attention, Pooling
- [D] Normalization, Quantization, Scaling

**Correct Answer**: **[B]**  
*Explanation*: The edit distance counts substitutions, deletions, and insertions.

---

### Question 2:
Can Word Error Rate (WER) exceed 100%?
- [A] No, it is mathematically bounded between 0% and 100%.
- [B] Yes, if the hypothesis contains more insertion errors than reference words ($I > N$).
- [C] Only on GPU hardware.
- [D] Only when using subwords.

**Correct Answer**: **[B]**  
*Explanation*: Unconstrained insertions can make the numerator arbitrarily larger than $N$.

---

### Question 3:
Why is micro-averaged WER universally required in scientific speech benchmarks?
- [A] It runs faster in Python.
- [B] It aggregates total errors across total words, preventing short outlier sentences from skewing the benchmark metric.
- [C] It requires fewer test files.
- [D] It eliminates punctuation automatically.

**Correct Answer**: **[B]**  
*Explanation*: Micro-averaging weights every spoken word equally.

---

### Question 4:
What is the computational complexity of the dynamic programming Levenshtein algorithm for sequences of length $N$ and $M$?
- [A] $\mathcal{O}(N + M)$
- [B] $\mathcal{O}(NM)$
- [C] $\mathcal{O}(2^{N+M})$
- [D] $\mathcal{O}(1)$

**Correct Answer**: **[B]**  
*Explanation*: Filling the $N \times M$ matrix takes $\mathcal{O}(NM)$ operations.

---

### Question 5:
In the formula $\text{WER} = \frac{S + D + I}{N}$, what does $N$ represent?
- [A] The number of words in the model hypothesis.
- [B] The total number of words in the ground-truth reference transcription ($C + S + D$).
- [C] The duration of the audio in seconds.
- [D] The sample rate.

**Correct Answer**: **[B]**  
*Explanation*: $N$ is the total reference word count.
"""
    (ldir / "quiz.md").write_text(quiz_doc, encoding="utf-8")

    res_doc = r"""# Research Connections: Lecture 24

---

## 1. Repository Grounding
- **`src/evaluation/metrics.py`**: Production implementation of micro-averaged WER, CER, and Levenshtein alignment.
- **`stages/stage_24/`**: Verification that all 37 historical experiment reports adhered to strict micro-averaged calculations.
"""
    (ldir / "research_connections.md").write_text(res_doc, encoding="utf-8")

    refs_doc = r"""# Annotated References: Lecture 24

---

1. **Levenshtein, V. I. (1966).**  
   *Binary codes capable of correcting deletions, insertions, and reversals.*  
   *Soviet Physics Doklady*, 10(8), 707–710.  
   - **Annotation**: The classic paper introducing edit distance and dynamic programming sequence alignment.
"""
    (ldir / "references.md").write_text(refs_doc, encoding="utf-8")
    print("Expanded Lecture 24 successfully!")

# ==============================================================================
# LECTURE 25: SCIENTIFIC BENCHMARKING METHODOLOGY
# ==============================================================================
def expand_lecture_25():
    ldir = BASE_DIR / "lecture_25_scientific_benchmarking_methodology"
    ldir.mkdir(parents=True, exist_ok=True)

    notes = r"""# Lecture 25: Scientific Benchmarking: The 8-Domain Kashmiri Evaluation Protocol

---

## 1. The Reproducibility Crisis in AI Benchmarks

In automatic speech recognition research, benchmark figures are frequently non-reproducible across laboratories:
- Different researchers evaluate on unstandardized text normalizers.
- Test audio contains silent speaker leakage from training partitions.
- Audio resampling introduces high-frequency roll-off.
- Evaluations are reported only on clean studio speech, concealing failure on natural spontaneous conversations.

To establish an authoritative, unassailable scientific standard for Kashmiri ASR, EduCasheer instituted the **8-Domain Frozen Evaluation Protocol** (`benchmarks/`).

---

## 2. The 8 Evaluation Domains and Acoustic Profiles

| Domain ID | Domain Name | Audio Profile | Primary Acoustic / Linguistic Challenge | Reference Hours |
| :---: | :--- | :--- | :--- | :---: |
| **DOM-01** | Studio Clean Read | Studio condenser mic, zero reverberation | Baseline phonetic clarity, vowel length contrasts | 2.50h |
| **DOM-02** | Prompted Conversational | Mobile headsets, diverse regional accents | Natural tempo variation, unstressed central vowels | 3.10h |
| **DOM-03** | Formal Broadcast News | Radio studio, standard literary Kashmiri | Complex Persianized vocabulary, formal syntax | 3.50h |
| **DOM-04** | Quranic & Expository | Professional vocal delivery, high register | Dense Arabic loanwords, pharyngealized consonants | 2.20h |
| **DOM-05** | Spontaneous Mosque (WAAZ) | Large hall reverberation, crowd coughs, echo | Severe non-stationary acoustics, spontaneous elisions | 4.20h |
| **DOM-06** | Community Smartphone | Near-field Android/iOS mics, ambient home noise | Variable gain, background traffic, room reverberation | 2.80h |
| **DOM-07** | Dialectal Folk Narratives | Elderly speakers, Southern/Northern rural accents | Archaic morphology, regional phonetic shifts | 2.40h |
| **DOM-08** | Educational Lectures | Podium dynamic mics, chalk writing, room echo | Code-switching with English terms, whiteboard pauses | 2.00h |

**Total Frozen Benchmark**: **22.70 hours** across 12,840 audio utterances with 0.00% speaker overlap against all training manifests.

---

## 3. The Protocol Invariants

To certify any model checkpoint in EduCasheer, an evaluation run must satisfy **five non-negotiable invariants**:
1. **Zero Contamination**: Absolute speaker disjointness verified by graph bipartite audit.
2. **Deterministic Text Normalization**: Transcripts and predictions must both pass through Certified Normalizer v3 before metric calculation.
3. **Certified Polyphase Resampling**: Audio must be resampled using Kaiser sinc interpolation ($\beta = 8.5555$, rolloff = 0.94).
4. **Frozen Decoder Hyperparameters**: Decoder parameters ($T=1.08, \alpha=0.50, \beta=1.50$) must remain constant across all 8 domains; per-domain test tuning is disqualified as hyperparameter leakage.
5. **Micro-Averaged Reporting**: Results must report both per-domain micro-averaged WER and the composite multi-domain micro-averaged WER.
"""
    (ldir / "lecture_notes.md").write_text(notes, encoding="utf-8")

    script = r"""# Instructor Delivery Script: Lecture 25 (Scientific Benchmarking Methodology)

## Timetable (90 Minutes)
```
00:00 - 00:20 | The Reproducibility Crisis in Low-Resource Speech
00:20 - 00:45 | Deconstructing the 8 Frozen Domains of EduCasheer
00:45 - 01:10 | The Five Mandatory Invariants of Certified Benchmarking
01:10 - 01:25 | Code Inspection: benchmarks/run_certified_eval.py
01:25 - 01:30 | Synthesis & Discussion
```

---

## Instructor Notes & Dialogue

### 00:00 - 00:20 | The Benchmark Illusion
- Step to the board:
  > "If a research paper claims 12% WER on Kashmiri, what does that number mean?
  > Did they test on studio recordings read by professional radio announcers?
  > Did the speaker in the test set appear 50 times in their training data?
  > Did they tune their language model weight $\alpha$ separately on the test set until the number looked pretty?
  > In real life, 12% on a clean read test set is 60% when you deploy it in an open-air marketplace in Srinagar.
  > Today, we formalize the 8-Domain Protocol.
  > A benchmark is not a vanity metric—it is a scientific stress test."
"""
    (ldir / "instructor_script.md").write_text(script, encoding="utf-8")

    math_doc = r"""# Mathematical Foundations: Domain-Stratified Composite Metrics

---

## 1. Multi-Domain Micro-Averaged WER Formulation

Let there be $D = 8$ evaluation domains.
Each domain $d$ contains $U_d$ utterances with reference words $N_d$ and total edit operations $E_d = S_d + D_d + I_d$.

The **Domain-Specific WER** is:
$$\text{WER}_d = \frac{E_d}{N_d}$$

The **Composite Multi-Domain Micro-Averaged WER** is:
$$\text{WER}_{\text{composite}} = \frac{\sum_{d=1}^D E_d}{\sum_{d=1}^D N_d} = \sum_{d=1}^D \left( \frac{N_d}{\sum_{k=1}^D N_k} \right) \text{WER}_d$$

Notice that composite micro-averaged WER is a **strictly weighted convex combination** of domain WERs, weighted proportional to domain word volume.
"""
    (ldir / "mathematical_foundations.md").write_text(math_doc, encoding="utf-8")

    code_doc = r"""# Code Walkthrough: Certified Benchmark Runner (`benchmarks/run_certified_eval.py`)

---

## 1. Inspecting the 8-Domain Benchmark Harness

```python
import json
from pathlib import Path
from src.evaluation.metrics import compute_corpus_wer
from src.normalization.kashmiri_normalizer import KashmiriNormalizerV3

DOMAINS = [
    "dom01_studio_clean",
    "dom02_prompted_conv",
    "dom03_broadcast_news",
    "dom04_quranic_expository",
    "dom05_waaz_reverberant",
    "dom06_community_mobile",
    "dom07_dialectal_folk",
    "dom08_educational_lecture"
]

def evaluate_certified_benchmark(pipeline, benchmark_dir="benchmarks"):
    domain_results = {}
    tot_E, tot_N = 0, 0
    
    for dom in DOMAINS:
        manifest_path = Path(benchmark_dir) / f"{dom}.jsonl"
        refs, hyps = [], []
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                norm_ref = KashmiriNormalizerV3.normalize(item["text"])
                pred_hyp = pipeline.transcribe(item["audio_filepath"])
                norm_hyp = KashmiriNormalizerV3.normalize(pred_hyp)
                
                refs.append(norm_ref)
                hyps.append(norm_hyp)
                
        wer, S, D, I, N = compute_corpus_wer(refs, hyps)
        domain_results[dom] = {"wer": wer, "S": S, "D": D, "I": I, "N": N}
        tot_E += (S + D + I)
        tot_N += N
        
    composite_wer = tot_E / max(1, tot_N)
    return composite_wer, domain_results
```
"""
    (ldir / "code_walkthrough.md").write_text(code_doc, encoding="utf-8")

    lab_doc = r"""# Practical Lab: Executing the 8-Domain Benchmark Evaluation

---

## Lab Objectives
1. Load evaluation manifests across multiple simulated acoustic domains.
2. Execute automated transcription and normalization.
3. Generate the official 8-domain evaluation scorecard and composite micro-averaged WER.

---

## Lab Script: `lab_certified_eval.py`

```python
def run_benchmark_lab():
    # Simulated domain evaluation
    domain_data = {
        "Studio Clean": (120, 1000),      # E=120, N=1000 -> 12.0%
        "Broadcast News": (180, 1000),    # E=180, N=1000 -> 18.0%
        "WAAZ Sermons": (520, 1000),      # E=520, N=1000 -> 52.0%
        "Community Mobile": (280, 1000),  # E=280, N=1000 -> 28.0%
    }
    
    tot_E = sum(v[0] for v in domain_data.values())
    tot_N = sum(v[1] for v in domain_data.values())
    
    print("=== Official Benchmark Scorecard ===")
    for dom, (E, N) in domain_data.items():
        print(f"Domain: {dom:20s} | WER: {E/N*100:5.2f}%")
        
    composite_wer = tot_E / tot_N
    print("---------------------------------------------")
    print(f"Composite Multi-Domain WER: {composite_wer*100:5.2f}%")
    assert abs(composite_wer - 0.275) < 1e-4, "Composite calculation error!"
    print("SUCCESS: Benchmark protocol executed cleanly!")

if __name__ == "__main__":
    run_benchmark_lab()
```
"""
    (ldir / "practical_lab.md").write_text(lab_doc, encoding="utf-8")

    ex_doc = r"""# Exercises: Lecture 25 (Scientific Benchmarking Methodology)

---

## Exercise 1: Composite Metric Calculation
A certified model evaluation produces the following domain metrics:
- Domain 1: $N_1 = 4,000 \text{ words}, \text{WER}_1 = 12.5\%$
- Domain 2: $N_2 = 6,000 \text{ words}, \text{WER}_2 = 22.0\%$
- Domain 3: $N_3 = 10,000 \text{ words}, \text{WER}_3 = 38.0\%$
1. Calculate the total edit errors $E_1, E_2, E_3$ in each domain.
2. Calculate the composite micro-averaged WER across the entire 20,000-word benchmark.
3. What is the macro-averaged WER across the 3 domains?

---

## Exercise 2: Auditing Invariant Violations
An external lab claims to reproduce EduCasheer's 24.12% WER using a smaller model, but their report states:
*"Audio was resampled using bilinear interpolation, and decoder alpha was tuned separately for each domain to achieve lowest error."*
Identify which protocol invariants were violated and explain why their claim is invalid.
"""
    (ldir / "exercises.md").write_text(ex_doc, encoding="utf-8")

    sol_doc = r"""# Solutions: Lecture 25 (Scientific Benchmarking Methodology)

---

## Solution to Exercise 1
1. Errors per domain:
   $$E_1 = 4000 \times 0.125 = \mathbf{500 \text{ errors}}$$
   $$E_2 = 6000 \times 0.220 = \mathbf{1320 \text{ errors}}$$
   $$E_3 = 10000 \times 0.380 = \mathbf{3800 \text{ errors}}$$
2. Composite micro-averaged WER:
   $$\text{WER}_{\text{composite}} = \frac{500 + 1320 + 3800}{4000 + 6000 + 10000} = \frac{5620}{20000} = \mathbf{28.10\%}$$
3. Macro-averaged WER:
   $$\text{WER}_{\text{macro}} = \frac{12.5\% + 22.0\% + 38.0\%}{3} = \frac{72.5\%}{3} = \mathbf{24.17\%}$$

---

## Solution to Exercise 2
Two critical invariants were violated:
1. **Invariant 3 (Certified Resampling)**: Bilinear interpolation introduces severe high-frequency roll-off and passband distortion compared to certified Kaiser sinc filtering.
2. **Invariant 4 (Frozen Decoder Hyperparameters)**: Tuning $\alpha$ separately for each test domain constitutes **test-set hyperparameter contamination (leakage)**. In real production deployments, an ASR system cannot tune hyperparameters on unseen user speech before transcribing it.
"""
    (ldir / "solutions.md").write_text(sol_doc, encoding="utf-8")

    quiz_doc = r"""# Quiz: Lecture 25 (Scientific Benchmarking Methodology)

---

### Question 1:
How many frozen evaluation domains are defined in the EduCasheer benchmarking protocol?
- [A] 2
- [B] 4
- [C] 8
- [D] 16

**Correct Answer**: **[C]**  
*Explanation*: 8 frozen domains cover studio, conversational, broadcast, religious, and reverberant acoustic profiles.

---

### Question 2:
What total duration of frozen disjoint audio is held in the 8 evaluation domains?
- [A] 1.5 hours
- [B] 5.0 hours
- [C] 22.70 hours
- [D] 100 hours

**Correct Answer**: **[C]**  
*Explanation*: The 8 domains comprise 22.70 hours across 12,840 utterances.

---

### Question 3:
Why is tuning language model weight $\alpha$ independently for each test domain considered a protocol violation?
- [A] It consumes too much CPU.
- [B] It constitutes test-set contamination (data leakage), producing artificially optimistic results that cannot be replicated in real-world deployment.
- [C] It crashes the KenLM library.
- [D] It changes the text font.

**Correct Answer**: **[B]**  
*Explanation*: Hyperparameters must be fixed prior to evaluating test sets.

---

### Question 4:
What text normalization must be executed prior to calculating benchmark Levenshtein metrics?
- [A] No normalization
- [B] Certified Normalizer v3 applied identically to both references and hypotheses
- [C] English spell-checker
- [D] Lowercase ASCII conversion

**Correct Answer**: **[B]**  
*Explanation*: Applying Normalizer v3 guarantees that typographical homoglyphs do not distort error rates.

---

### Question 5:
Which evaluation domain represents the most acoustically challenging condition in the benchmark?
- [A] DOM-01: Studio Clean
- [B] DOM-03: Broadcast News
- [C] DOM-05: Spontaneous Mosque Sermons (WAAZ)
- [D] DOM-04: Quranic Commentary

**Correct Answer**: **[C]**  
*Explanation*: WAAZ combines open-air acoustic reverberation, loudspeaker distortion, and spontaneous crowd noise.
"""
    (ldir / "quiz.md").write_text(quiz_doc, encoding="utf-8")

    res_doc = r"""# Research Connections: Lecture 25

---

## 1. Repository Grounding
- **`benchmarks/`**: Contains the 8 canonical evaluation manifests (`dom01.jsonl` to `dom08.jsonl`).
- **`stages/stage_25/`**: Independent verification audit that reconciled historical metrics against the frozen 8-domain protocol.
"""
    (ldir / "research_connections.md").write_text(res_doc, encoding="utf-8")

    refs_doc = r"""# Annotated References: Lecture 25

---

1. **Pratap, V., et al. (2020).**  
   *MLS: A large-scale multilingual dataset for speech research.*  
   *Interspeech 2020*.  
   - **Annotation**: Discusses multi-domain evaluation protocols, disjoint speaker splitting, and cross-domain robustness.
"""
    (ldir / "references.md").write_text(refs_doc, encoding="utf-8")
    print("Expanded Lecture 25 successfully!")

# ==============================================================================
# LECTURE 26: STATISTICAL SIGNIFICANCE AND BOOTSTRAP RESAMPLING
# ==============================================================================
def expand_lecture_26():
    ldir = BASE_DIR / "lecture_26_statistical_significance_bootstrap"
    ldir.mkdir(parents=True, exist_ok=True)

    notes = r"""# Lecture 26: Statistical Significance, Confidence Intervals, and Bootstrap Resampling

---

## 1. The Myth of the 0.5% WER Improvement

In machine learning research literature, papers frequently claim state-of-the-art superiority based on tiny margin improvements:
*"Model A achieved 24.12% WER, outperforming Model B's 24.40% WER by 0.28%."*

Without statistical testing, this claim is scientifically meaningless:
- Is a 0.28% difference a genuine structural algorithmic breakthrough?
- Or is it merely random statistical noise arising from the specific sample of test speakers?

In low-resource speech recognition with finite test sets, **hypothesis testing and confidence interval estimation are scientifically mandatory**.

---

## 2. Why Parametric Tests (t-tests) Fail for Speech Recognition

Standard statistical hypothesis tests (such as Student's paired $t$-test) make strict distributional assumptions:
1. Data points must be normally distributed: $X \sim \mathcal{N}(\mu, \sigma^2)$.
2. Observations must have constant variance (homoscedasticity).

Word error rates on speech sentences violate every parametric assumption:
- Error distributions are **heavily right-skewed** (most clean sentences have 0 errors, while noisy sentences have high error spikes).
- Sentence lengths vary from 1 word to 40 words, causing variance to fluctuate wildly.

Consequently, speech researchers must rely on **non-parametric statistical methods**:
1. **Paired Bootstrap Resampling**
2. **Paired Wilcoxon Signed-Rank Test**

---

## 3. The Non-Parametric Paired Bootstrap Test

Proposed by Efron (1979) and adapted to speech by Bisani and Ney (2004), the paired bootstrap estimates the empirical sampling distribution without making parametric assumptions.

### Algorithm:
Given test corpus of $M$ utterances and two competing models $A$ and $B$:
1. For bootstrap iteration $b = 1, \dots, B$ (typically $B = 10,000$):
   - Sample $M$ utterances **uniformly at random with replacement** from the test corpus: $\mathcal{S}^{(b)} = (u_1^*, \dots, u_M^*)$.
   - Evaluate micro-averaged WER for both models on sample $\mathcal{S}^{(b)}$: $\text{WER}_A^{(b)}$ and $\text{WER}_B^{(b)}$.
   - Calculate the delta: $\Delta^{(b)} = \text{WER}_B^{(b)} - \text{WER}_A^{(b)}$.
2. Compute the **empirical two-tailed $p$-value**:
   $$p = 2 \cdot \min\left( \frac{1}{B} \sum_{b=1}^B \mathbb{I}(\Delta^{(b)} \le 0), \frac{1}{B} \sum_{b=1}^B \mathbb{I}(\Delta^{(b)} \ge 0) \right)$$
3. If $p < 0.05$, the difference is statistically significant at the $95\%$ confidence level.

---

## 4. Bootstrap Confidence Intervals (Percentile Method)

To construct a $95\%$ confidence interval for Model A's true error rate:
1. Sort the $B$ bootstrap estimates in ascending order:
   $$\text{WER}_{(1)} \le \text{WER}_{(2)} \le \dots \le \text{WER}_{(B)}$$
2. The $95\%$ confidence interval is given by the $2.5\text{th}$ and $97.5\text{th}$ percentiles:
   $$\text{CI}_{95\%} = \left[ \text{WER}_{(\lfloor 0.025 B \rfloor)}, \text{WER}_{(\lfloor 0.975 B \rfloor)} \right]$$

For the certified EduCasheer model `E22-01`:
$$\text{WER}_{\text{composite}} = 24.12\% \quad (\text{95\% CI: } [23.45\%, 24.81\%])$$
"""
    (ldir / "lecture_notes.md").write_text(notes, encoding="utf-8")

    script = r"""# Instructor Delivery Script: Lecture 26 (Statistical Significance and Bootstrap Resampling)

## Timetable (90 Minutes)
```
00:00 - 00:20 | The 0.28% Question: Scientific Breakthrough or Statistical Mirage?
00:20 - 00:45 | Why Student's t-test Fails on Non-Normal Speech Errors
00:45 - 01:10 | Non-Parametric Paired Bootstrap Resampling (B = 10,000)
01:10 - 01:25 | Line-by-Line Code Inspection: src/evaluation/significance.py
01:25 - 01:30 | Synthesis & Transition to Module 09 (Research Archaeology)
```

---

## Instructor Notes & Dialogue

### 00:00 - 00:20 | The 0.28% Demonstration
- Write on the board:
  - Model 1: `WER = 24.12%`
  - Model 2: `WER = 24.40%`
  - Difference: `0.28%`
- Look at the students:
  > "If you publish a paper saying Model 1 is better than Model 2 because of 0.28%, a competent reviewer will ask:
  > 'What is your p-value?'
  > If your test set has 500 sentences, a 0.28% difference is literally **three words**!
  > If one speaker coughs during recording, your 0.28% advantage disappears!
  > Today, we learn how to prove whether a result is statistically solid or an artifact of chance.
  > In science, if you cannot prove statistical significance, you have not discovered anything."
"""
    (ldir / "instructor_script.md").write_text(script, encoding="utf-8")

    math_doc = r"""# Mathematical Foundations: Bootstrap Consistency and P-Value Bounds

---

## 1. Asymptotic Consistency of Empirical Bootstrap

Let $\hat{\theta}_n$ be an estimator based on sample size $n$ with true parameter $\theta_0$.
According to Bickel and Freedman (1981), if the underlying statistics possess finite variance:

$$\sup_{x} \left| P^*\left( \sqrt{n}(\theta^* - \hat{\theta}_n) \le x \right) - P\left( \sqrt{n}(\hat{\theta}_n - \theta_0) \le x \right) \right| \xrightarrow{P} 0 \quad \text{as } n \to \infty$$

The bootstrap distribution $P^*$ asymptotically mirrors the true sampling distribution of the test statistic.

---

## 2. Paired Wilcoxon Signed-Rank Test

For paired utterance differences $d_i = \text{WER}_A(u_i) - \text{WER}_B(u_i)$:
1. Exclude ties ($d_i = 0$).
2. Rank absolute differences $|d_i|$ from smallest to largest with ranks $R_i$.
3. Compute sum of positive ranks $W^+$ and negative ranks $W^-$:
   $$W = \min(W^+, W^-)$$
For $N > 20$, the test statistic $z$ is approximately standard normal:
$$z = \frac{W - \frac{N(N+1)}{4}}{\sqrt{\frac{N(N+1)(2N+1)}{24}}} \sim \mathcal{N}(0, 1)$$
"""
    (ldir / "mathematical_foundations.md").write_text(math_doc, encoding="utf-8")

    code_doc = r"""# Code Walkthrough: Paired Bootstrap Resampling (`src/evaluation/significance.py`)

---

## 1. Inspecting the Bootstrap Test Harness

```python
import numpy as np

def paired_bootstrap_test(refs, hyps_a, hyps_b, num_bootstrap=10000, alpha=0.05):
    # refs, hyps_a, hyps_b: lists of sentence strings of equal length M
    M = len(refs)
    deltas = np.zeros(num_bootstrap)
    
    # Pre-calculate utterance edit operations
    errors_a = []
    errors_b = []
    word_counts = []
    
    for r, ha, hb in zip(refs, hyps_a, hyps_b):
        words = len(r.split())
        ea = compute_errors(r, ha)
        eb = compute_errors(r, hb)
        errors_a.append(ea)
        errors_b.append(eb)
        word_counts.append(words)
        
    err_a = np.array(errors_a)
    err_b = np.array(errors_b)
    words = np.array(word_counts)
    
    for b in range(num_bootstrap):
        # Sample with replacement
        idx = np.random.randint(0, M, size=M)
        tot_w = np.sum(words[idx])
        wer_a = np.sum(err_a[idx]) / tot_w
        wer_b = np.sum(err_b[idx]) / tot_w
        deltas[b] = wer_b - wer_a
        
    # Two-tailed p-value
    p_val = 2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))
    is_significant = (p_val < alpha)
    
    ci_lower = np.percentile(deltas, 2.5)
    ci_upper = np.percentile(deltas, 97.5)
    
    return {
        "p_value": p_val,
        "is_significant": is_significant,
        "mean_delta": np.mean(deltas),
        "ci_95": (ci_lower, ci_upper)
    }
```
"""
    (ldir / "code_walkthrough.md").write_text(code_doc, encoding="utf-8")

    lab_doc = r"""# Practical Lab: Executing Bootstrap Significance Testing

---

## Lab Objectives
1. Implement non-parametric bootstrap resampling in NumPy.
2. Test whether an observed 0.40% WER improvement is statistically significant across 500 test sentences.
3. Compute 95% confidence intervals and plot the bootstrap delta histogram.

---

## Lab Script: `lab_significance_test.py`

```python
import numpy as np

def run_significance_lab():
    np.random.seed(42)
    M = 500 # 500 utterances
    
    # Simulate errors per utterance
    words_per_utt = np.random.randint(5, 25, size=M)
    errors_base = np.random.binomial(words_per_utt, 0.25)
    # Model candidate improves slightly (prob 0.245)
    errors_cand = np.random.binomial(words_per_utt, 0.245)
    
    B = 2000
    deltas = []
    
    for _ in range(B):
        idx = np.random.choice(M, size=M, replace=True)
        tot_w = np.sum(words_per_utt[idx])
        wer_base = np.sum(errors_base[idx]) / tot_w
        wer_cand = np.sum(errors_cand[idx]) / tot_w
        deltas.append(wer_base - wer_cand)
        
    deltas = np.array(deltas)
    p_val = 2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))
    ci = (np.percentile(deltas, 2.5), np.percentile(deltas, 97.5))
    
    print(f"Observed Mean Delta WER: {np.mean(deltas)*100:.3f}%")
    print(f"95% Confidence Interval: [{ci[0]*100:.3f}%, {ci[1]*100:.3f}%]")
    print(f"Calculated p-value:      {p_val:.4f}")
    print(f"Significant at p < 0.05: {p_val < 0.05}")
    print("SUCCESS: Bootstrap test framework executed cleanly!")

if __name__ == "__main__":
    run_significance_lab()
```
"""
    (ldir / "practical_lab.md").write_text(lab_doc, encoding="utf-8")

    ex_doc = r"""# Exercises: Lecture 26 (Statistical Significance and Bootstrap Resampling)

---

## Exercise 1: Confidence Interval Interpretation
A bootstrap test on Model `E22-01` produces $B = 10,000$ resampled WER estimates.
The sorted estimates yield:
- Index 250 (2.5%): $\text{WER} = 23.45\%$
- Index 5,000 (50.0%): $\text{WER} = 24.12\%$
- Index 9,750 (97.5%): $\text{WER} = 24.81\%$
1. State the 95% confidence interval for the true model WER.
2. If a competing laboratory claims their model achieves 23.90% WER, can we claim our model is statistically inferior to theirs at $p < 0.05$ based solely on this interval? Explain.

---

## Exercise 2: The Bonferroni Correction for Multiple Comparisons
Suppose a researcher evaluates 8 different model variants against a baseline model on the same test set.
To maintain an overall false positive rate of $\alpha = 0.05$ across all 8 tests:
1. What adjusted significance threshold $\alpha^*$ must be used for each individual test under the Bonferroni correction?
2. If a model variant achieves $p = 0.012$, is it statistically significant after Bonferroni correction?
"""
    (ldir / "exercises.md").write_text(ex_doc, encoding="utf-8")

    sol_doc = r"""# Solutions: Lecture 26 (Statistical Significance and Bootstrap Resampling)

---

## Solution to Exercise 1
1. The 95% confidence interval is: **$[23.45\%, 24.81\%]$**.
2. No! The competitor's score of $23.90\%$ falls squarely inside our 95% confidence interval $[23.45\%, 24.81\%]$. Because the point estimate lies within the range of sampling variation, there is no statistically significant difference without conducting a direct paired bootstrap comparison.

---

## Solution to Exercise 2
1. Under Bonferroni correction for $m = 8$ hypotheses:
   $$\alpha^* = \frac{\alpha}{m} = \frac{0.05}{8} = \mathbf{0.00625}$$
2. For $p = 0.012$:
   Since $0.012 > 0.00625$, the result is **not statistically significant** after Bonferroni correction. It could readily arise as a false positive due to multiple hypothesis testing.
"""
    (ldir / "solutions.md").write_text(sol_doc, encoding="utf-8")

    quiz_doc = r"""# Quiz: Lecture 26 (Statistical Significance and Bootstrap Resampling)

---

### Question 1:
Why are parametric statistical tests (like Student's t-test) inappropriate for comparing ASR Word Error Rates?
- [A] Speech error distributions are heavily non-normal, right-skewed, and exhibit variable variance across sentence lengths.
- [B] They take too long to run.
- [C] Speech cannot be stored as numbers.
- [D] They only work for images.

**Correct Answer**: **[A]**  
*Explanation*: Speech errors violate the normality and homoscedasticity assumptions of parametric tests.

---

### Question 2:
How is a sample constructed in paired bootstrap resampling?
- [A] By recording new speech files in a studio.
- [B] By sampling $M$ utterances uniformly at random with replacement from the existing test corpus.
- [C] By deleting all short sentences.
- [D] By averaging the audio waveforms.

**Correct Answer**: **[B]**  
*Explanation*: Random sampling with replacement generates the empirical bootstrap distribution.

---

### Question 3:
How many bootstrap resamples ($B$) are standardly evaluated to achieve stable p-value estimation?
- [A] 10
- [B] 100
- [C] 10,000
- [D] 1,000,000

**Correct Answer**: **[C]**  
*Explanation*: $B = 10,000$ provides tight resolution for confidence intervals and p-values down to $\alpha = 0.01$.

---

### Question 4:
What does a 95% confidence interval of $[23.45\%, 24.81\%]$ for model WER signify?
- [A] The model will fail 5% of the time.
- [B] Under repeated sampling from the population of speakers, 95% of estimated intervals will contain the true population WER.
- [C] The model has 23 layers.
- [D] The audio sample rate is 24 kHz.

**Correct Answer**: **[B]**  
*Explanation*: Confidence intervals quantify the precision of empirical point estimates.

---

### Question 5:
What correction must be applied to the significance threshold $\alpha$ when conducting multiple pairwise model comparisons?
- [A] Temperature scaling
- [B] Bonferroni correction ($\alpha^* = \alpha / m$)
- [C] Sinc interpolation
- [D] Levenshtein penalty

**Correct Answer**: **[B]**  
*Explanation*: Bonferroni correction controls the family-wise error rate across multiple hypothesis tests.
"""
    (ldir / "quiz.md").write_text(quiz_doc, encoding="utf-8")

    res_doc = r"""# Research Connections: Lecture 26

---

## 1. Repository Grounding
- **`src/evaluation/significance.py`**: Implementation of paired bootstrap resampling and Wilcoxon tests.
- **`stages/stage_26/`**: Re-evaluated the B-02C vs. E22-01 difference, confirming that the 1.70% improvement was statistically significant ($p = 0.0014 < 0.05$).
"""
    (ldir / "research_connections.md").write_text(res_doc, encoding="utf-8")

    refs_doc = r"""# Annotated References: Lecture 26

---

1. **Bisani, M., & Ney, H. (2004).**  
   *Bootstrap estimates for confidence intervals in ASR evaluation.*  
   *Interspeech 2004*, 409–412.  
   - **Annotation**: Foundational paper establishing paired bootstrap resampling for speech recognition benchmarks.

2. **Efron, B. (1979).**  
   *Bootstrap methods: Another look at the jackknife.*  
   *The Annals of Statistics*, 7(1), 1–26.  
   - **Annotation**: The seminal classic formulating the non-parametric bootstrap resampling methodology.
"""
    (ldir / "references.md").write_text(refs_doc, encoding="utf-8")
    print("Expanded Lecture 26 successfully!")

if __name__ == "__main__":
    expand_lecture_24()
    expand_lecture_25()
    expand_lecture_26()
