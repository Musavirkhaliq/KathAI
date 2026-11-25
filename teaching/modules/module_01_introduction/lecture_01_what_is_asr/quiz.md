# Quiz: Lecture 01 (What is Automatic Speech Recognition?)

---

### Question 1:
Under statistical decision theory, what loss function makes Maximum A Posteriori (MAP) sequence decoding strictly optimal?
- [A] Word Error Rate (Levenshtein distance)
- [B] Mean Squared Error (MSE)
- [C] Zero-One Sentence Error Loss ($L(\mathbf{w}, \hat{\mathbf{w}}) = \mathbb{I}(\mathbf{w} \ne \hat{\mathbf{w}})$)
- [D] Cross-Entropy Loss

**Correct Answer**: **[C]**  
*In-Depth Explanation*:
- **Why [C] is correct**: As proved in `mathematical_foundations.md`, the conditional risk under zero-one loss reduces directly to $R(\hat{\mathbf{w}} \mid \mathbf{X}) = 1 - P(\hat{\mathbf{w}} \mid \mathbf{X})$. Minimizing risk is therefore identical to maximizing posterior probability $P(\hat{\mathbf{w}} \mid \mathbf{X})$ (MAP).
- **Why [A] is incorrect**: Word Error Rate penalizes sequences proportionally to edit operations, leading to Minimum Bayes Risk (MBR) decoding, not MAP.
- **Why [B] is incorrect**: MSE is defined for continuous regression, not discrete sequence classification.
- **Why [D] is incorrect**: Cross-entropy is a surrogate training loss, not an evaluation decision loss.

---

### Question 2:
Why can the marginal likelihood (evidence) $P(\mathbf{X})$ be omitted during MAP sequence decoding?
- [A] Because $P(\mathbf{X}) = 1.0$ for all audio files.
- [B] Because $P(\mathbf{X})$ is an integral over all candidate sequences that depends solely on acoustic observation $\mathbf{X}$ and does not vary across competing hypotheses $\mathbf{w}$.
- [C] Because PyTorch automatically normalizes tensors.
- [D] Because acoustic models only operate in the frequency domain.

**Correct Answer**: **[B]**  
*In-Depth Explanation*:
- **Why [B] is correct**: By the Law of Total Probability, $P(\mathbf{X}) = \sum_{\mathbf{w}'} P(\mathbf{X} \mid \mathbf{w}') P(\mathbf{w}')$. It is constant with respect to candidate hypothesis $\mathbf{w}$. Since multiplying an objective by a positive constant preserves argmax, it has zero impact on search.
- **Why [A] is incorrect**: $P(\mathbf{X})$ is a continuous probability density, not 1.0.
- **Why [C] and [D] are incorrect**: These are irrelevant implementation and signal details.

---

### Question 3:
What acoustic property primarily distinguishes the Kashmiri central vowel /əː/ (represented by `ٲ` in `کٲشُر`) from high front /i/ and high back /u/?
- [A] It has an intermediate second formant ($F_2 \approx 1550\text{ Hz}$), whereas /i/ has high $F_2$ (~2200 Hz) and /u/ has low $F_2$ (~800 Hz).
- [B] It has zero fundamental frequency ($F_0 = 0$).
- [C] It has a duration exceeding 5 seconds.
- [D] It can only be spoken by male speakers.

**Correct Answer**: **[A]**  
*In-Depth Explanation*:
- **Why [A] is correct**: Second formant frequency ($F_2$) directly correlates with tongue advancement (backness). Front vowels have high $F_2$, back vowels have low $F_2$, and central vowels occupy the intermediate 1400–1600 Hz space.
- **Why [B], [C], and [D] are incorrect**: Vowels are voiced ($F_0 > 0$), have normal phonetic durations (100–300 ms), and are spoken across genders.

---

### Question 4:
What was the primary technical root cause of the 88.40% WER collapse in the Stage 01 Urdu Kaldi baseline?
- [A] The audio files were corrupted by MP3 compression.
- [B] The Urdu pronunciation lexicon contained zero phonetic entries for Kashmiri central vowels and palatalization, causing forced alignment to reject 38.2% of utterances.
- [C] The computer ran out of GPU memory.
- [D] The sample rate was too high (96 kHz).

**Correct Answer**: **[B]**  
*In-Depth Explanation*:
- **Why [B] is correct**: As proven by `stages/stage_01/logs/urdu_baseline_eval.log`, Kaldi's WFST composition $H \circ C \circ L \circ G$ had no valid paths for Kashmiri phonetics, rejecting 191 out of 500 utterances and producing catastrophic substitution rates.
- **Why [A], [C], and [D] are incorrect**: The audio was standard WAV format, Kaldi ran on CPU, and sampling was 16 kHz.

---

### Question 5:
Why does unconstrained character-level CTC decoding require a positive word insertion penalty ($\beta |\mathbf{w}|$) during shallow fusion?
- [A] To increase decoding latency.
- [B] To compensate for brevity bias caused by negative log-probabilities decaying with sequence length, preventing excessive word deletions.
- [C] To convert Perso-Arabic text to Latin script.
- [D] To satisfy Unicode NFKC standardization.

**Correct Answer**: **[B]**  
*In-Depth Explanation*:
- **Why [B] is correct**: Since frame log-probabilities are negative ($\log P < 0$), longer strings accumulate more negative penalties. Adding $\beta |\mathbf{w}|$ adds a positive reward for each completed word, counterbalancing deletion bias.
- **Why [A], [C], and [D] are incorrect**: Length penalties are designed for accuracy balancing, not latency, script transliteration, or Unicode formatting.
