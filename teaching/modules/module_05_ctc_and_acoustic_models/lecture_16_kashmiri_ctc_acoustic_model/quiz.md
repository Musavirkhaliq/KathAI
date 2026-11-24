# Quiz: Lecture 16 (Building the Kashmiri CTC Acoustic Model)

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
