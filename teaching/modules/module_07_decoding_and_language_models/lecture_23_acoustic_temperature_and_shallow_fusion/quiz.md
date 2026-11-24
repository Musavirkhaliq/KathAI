# Quiz: Lecture 23 (Acoustic Temperature and Shallow Fusion)

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
