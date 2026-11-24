# Exercises: Lecture 32 (Code-Switching and Dialectal Adaptation)

---

## Exercise 1: LoRA Parameter Count Calculation
For an XLS-R Transformer layer with $d_{\text{model}} = 1024$:
1. How many parameters exist in the Query and Value projection matrices ($\mathbf{W}_Q, \mathbf{W}_V$)?
2. If LoRA with rank $r = 8$ is added to both $\mathbf{W}_Q$ and $\mathbf{W}_V$, how many trainable parameters are added per layer?
3. What is the percentage parameter reduction compared to full fine-tuning of those matrices?

---

## Exercise 2: Code-Switching Tokenizer Challenge
A Kashmiri sentence contains the English word `"computer"` transliterated into Perso-Arabic: `"کَمپیوٗٹَر"`.
1. Explain how character-level tokenization handles this word without creating out-of-vocabulary (OOV) errors.
2. What language model prior $P_{\text{LM}}$ penalty might this word face if the LM was trained solely on classical 19th-century Kashmiri literature?
