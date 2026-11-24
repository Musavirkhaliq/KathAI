# Comprehension Quiz: Lecture 01

---

### Question 1: The Marginal Evidence Term
In the Bayesian derivation of the speech recognition objective $\hat{\mathbf{w}} = \arg\max_{\mathbf{w}} P(\mathbf{w} \mid \mathbf{X})$, why is the evidence denominator $P(\mathbf{X})$ omitted during inference?
- [ ] A) Because $P(\mathbf{X})$ is always equal to $1.0$ for normalized audio tensors.
- [x] B) Because $P(\mathbf{X}) = \sum_{\mathbf{w}'} P(\mathbf{X} \mid \mathbf{w}') P(\mathbf{w}')$ depends strictly on the acoustic features $\mathbf{X}$ and is invariant across all competing word hypotheses $\mathbf{w}$.
- [ ] C) Because $P(\mathbf{X})$ cancels out when applying the CTC collapse operator.
- [ ] D) Because $P(\mathbf{X})$ underflows in 32-bit floating point arithmetic.

**Explanation**: The argmax operator seeks the hypothesis $\mathbf{w}$ that maximizes the posterior. Since $P(\mathbf{X})$ has the exact same value for every hypothesis $\mathbf{w}$, dividing by it does not alter their relative rankings.

---

### Question 2: Zero-One Loss Optimality
The Maximum A Posteriori (MAP) decision rule $\hat{\mathbf{w}} = \arg\max_{\mathbf{w}} P(\mathbf{w} \mid \mathbf{X})$ is Bayes-optimal strictly under which of the following loss functions?
- [ ] A) Word Error Rate (Levenshtein Distance)
- [ ] B) Mean Squared Error (MSE)
- [x] C) Zero-One Sequence Loss ($L(\mathbf{w}, \hat{\mathbf{w}}) = \mathbb{I}(\mathbf{w} \ne \hat{\mathbf{w}})$)
- [ ] D) Connectionist Temporal Classification Loss

**Explanation**: Under zero-one loss, the risk is $R(\hat{\mathbf{w}} \mid \mathbf{X}) = 1 - P(\hat{\mathbf{w}} \mid \mathbf{X})$. Minimizing risk is mathematically identical to maximizing posterior $P(\hat{\mathbf{w}} \mid \mathbf{X})$. Under edit distance, expected risk depends on all neighboring hypotheses.

---

### Question 3: The Abjad Dilemma in Kashmiri ASR
Why does the common practice of omitting vowel diacritics (*aerab*) in informal Kashmiri text create severe difficulties for training an acoustic model?
- [ ] A) It makes the text file size smaller than the audio file.
- [x] B) Multiple distinct acoustic phonetic words map to the exact same undiacritized written representation, forcing the neural network to learn an unnatural multimodal distribution with high conditional entropy.
- [ ] C) PyTorch cannot compute cross-entropy on strings with fewer than 3 characters.
- [ ] D) It causes the GPU to throw a CUDA Out-of-Memory exception.

**Explanation**: When distinct sounds (e.g. /kar/, /kər/, /kɨr/) are all labeled with the identical target `کر`, the network cannot find a consistent acoustic mapping, resulting in high loss entropy and poor convergence.

---

### Question 4: Word Insertion Bonus
In the shallow fusion objective $\mathcal{S}(\mathbf{w}) = \ln P_\text{AM}(\mathbf{w} \mid \mathbf{X}) + \alpha \ln P_\text{LM}(\mathbf{w}) + \beta |\mathbf{w}|$, what is the primary role of the hyperparameter $\beta > 0$?
- [ ] A) It increases the learning rate during gradient descent.
- [ ] B) It multiplies the audio sample rate by factor $\beta$.
- [x] C) It provides a positive reward for each emitted word to counteract the negative log-probability accumulation of the language model, preventing the decoder from favoring unrealistically short hypotheses.
- [ ] D) It scales the CTC blank token probability.

**Explanation**: Because $\ln P_\text{LM}(w) \le 0$, adding words decreases total score. $\beta > 0$ compensates for this penalty to prevent deletion bias.

---

### Question 5: Lessons from EduCasheer Stage 01
What was the primary reason an off-the-shelf Urdu Wav2Vec2 model failed on Kashmiri speech with an $88.4\%$ Word Error Rate?
- [ ] A) The audio was recorded at 44.1 kHz instead of 16 kHz.
- [x] B) Kashmiri possesses central vowels (/ɨ/, /ə/) and dental affricates absent in Urdu phonology, causing the Urdu model to misclassify or delete these sounds.
- [ ] C) The Urdu model was trained on CPU rather than GPU.
- [ ] D) The Kashmiri language cannot be represented in Unicode.

**Explanation**: Shared Perso-Arabic writing between Urdu and Kashmiri masks fundamental phonetic divergence, especially in central vowels and Dardic affricates.
