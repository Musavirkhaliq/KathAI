# CS-892: Automatic Speech Recognition — Comprehensive Final Examination
**Scope**: Synoptic Comprehensive (All Modules 01 through 12)  
**Duration**: 180 Minutes | **Total Marks**: 100 Points

---

## Section A: Advanced Decoding & Language Modeling (30 Marks)
1. **(15 Marks)** Deconstruct CTC Prefix Beam Search. Explain mathematically why tracking separate blank $P_b$ and non-blank $P_{nb}$ probabilities is mandatory when extending candidate prefixes by identical characters.
2. **(15 Marks)** Formulate the Calibrated Shallow Fusion objective:

$$\mathcal{S}(\mathbf{w}) = \log P_\text{CTC}(\mathbf{y} \mid \mathbf{X}; T) + \alpha \log_{10} P_\text{LM}(\mathbf{w}) + \beta |\mathbf{w}|$$

   Explain the physical role of acoustic temperature $T=1.08$, language weight $\alpha$, and word bonus $\beta$.

---

## Section B: Scientific Benchmarking & Statistical Rigor (35 Marks)
3. **(15 Marks)** Formulate the Levenshtein dynamic programming algorithm for minimum edit distance. Prove that micro-averaged Word Error Rate is strictly equivalent to the word-count-weighted average of individual sentence WERs.
4. **(20 Marks)** Explain why Student's $t$-test is theoretically invalid for evaluating ASR differences across utterances. Detail the step-by-step algorithms for:
   - Non-Parametric Bootstrap $95\%$ Confidence Intervals.
   - Paired Wilcoxon Signed-Rank Test for statistical significance.

---

## Section C: Production Systems & Research Autopsy (35 Marks)
5. **(15 Marks)** Deconstruct the Stage 15 negative result: why did Minimum Word Error Rate (MWER) sequence training reduce training WER (18% -> 11%) while degrading out-of-domain evaluation on WAAZ speech by $+6.8\%$ absolute?
6. **(20 Marks)** In transcribing a 45-minute audio recording, explain why monolithic Transformer forward passes fail. Detail the Method C2 Overlap-Add Chunking algorithm: show how 8.0s core windows, 1.0s context margins, and snapping to $R=320$ sample multiples eliminate boundary seam errors.
