# Exercises: Lecture 01 (What is Automatic Speech Recognition?)

---

## Exercise 1: Theoretical Proof of Bayesian Evidence Invariance
Let $\mathbf{X}$ be an observed acoustic tensor and $\mathbf{w} \in \mathcal{W}$ be a candidate word sequence.
1. Formally write the expression for the marginal likelihood $P(\mathbf{X})$.
2. Show that $\frac{\partial P(\mathbf{X})}{\partial \mathbf{w}} = 0$.
3. Prove that for any scalar function $g(\mathbf{X}) > 0$, the decision rule:

$$\hat{\mathbf{w}} = \arg\max_{\mathbf{w} \in \mathcal{W}} [g(\mathbf{X}) P(\mathbf{w} \mid \mathbf{X})]$$

   is strictly invariant to the choice of $g(\mathbf{X})$.

---

## Exercise 2: MBR vs. MAP Under Asymmetric Loss
Consider an emergency voice dispatch system where failing to recognize the word `"خَطَر"` (*danger*) incurs a catastrophic penalty of $C = 100$, while normal word errors incur a penalty of $1$.
1. Formulate the asymmetric loss function $L_{\text{emergency}}(\mathbf{w}, \hat{\mathbf{w}})$.
2. Suppose posterior probabilities are `P("خَطَر" \mid \mathbf{X}) = 0.05` and `P("خَبَر" \mid \mathbf{X}) = 0.95`.
   Which word does MAP decoding select?
3. Calculate the conditional Bayes risk for both hypotheses and determine which word MBR decoding selects.

---

## Exercise 3: Vowel Formant Acoustic Analysis
A speech scientist records three sustained Kashmiri vowel sounds. Spectral analysis reveals the following peak frequencies:
- Vowel A: $F_1 = 300\text{ Hz}, F_2 = 2200\text{ Hz}$
- Vowel B: $F_1 = 340\text{ Hz}, F_2 = 1450\text{ Hz}$
- Vowel C: $F_1 = 580\text{ Hz}, F_2 = 1550\text{ Hz}$
1. Map each vowel to its correct International Phonetic Alphabet (IPA) symbol (/i/, /ɨ/, /əː/).
2. Which of these vowels corresponds to the letter `ٲ` in the word `"کٲشُر"`?
3. Why does an Urdu acoustic model misclassify Vowel B and Vowel C?

---

## Exercise 4: Forensic Analysis of the Stage 01 Urdu Failure
Review the Stage 01 execution log (`stages/stage_01/logs/urdu_baseline_eval.log`):
- Total utterances: 500
- Successfully aligned: 309 utterances
- Discarded / Rejected: 191 utterances
- Reference words: 5,000
- Substitutions: 2,118; Deletions: 1,890; Insertions: 412
1. Calculate the percentage of utterances rejected by forced alignment.
2. Calculate the Word Error Rate (WER) using the standard Levenshtein formula.
3. If an engineer claims: *"The model achieved 88.4% WER, so it worked on 11.6% of words"*, explain why this statement is technically flawed.
