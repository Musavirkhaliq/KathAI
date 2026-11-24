# Quiz: Lecture 32 (Code-Switching and Dialectal Adaptation)

---

### Question 1:
What is Low-Rank Adaptation (LoRA)?
- [A] A method for compressing audio waveforms.
- [B] A parameter-efficient fine-tuning technique that freezes base model weights and trains low-rank decomposition matrices ($r \ll d$) for specific tasks or dialects.
- [C] A new dialect spoken in North Kashmir.
- [D] An audio noise filter.

**Correct Answer**: **[B]**  
*Explanation*: LoRA factorizes weight updates into low-rank matrices, reducing trainable parameters by $>95\%$.

---

### Question 2:
Why is matrix $\mathbf{B}$ initialized to zero in LoRA ($\mathbf{B} = \mathbf{0}$)?
- [A] To save memory.
- [B] To ensure the adaptation term $\Delta \mathbf{W} = \mathbf{B} \mathbf{A}$ starts at zero, preventing initial disturbance to pretrained weights.
- [C] Because PyTorch does not support random initialization.
- [D] To make the model run on CPU.

**Correct Answer**: **[B]**  
*Explanation*: Zero initialization guarantees the model begins exactly at base pretrained performance.

---

### Question 3:
What are the two major regional dialect groups of Kashmiri?
- [A] Punjabi and Pashto
- [B] Maraz (Southern) and Kamraz (Northern)
- [C] Sindhi and Balochi
- [D] Shina and Burushaski

**Correct Answer**: **[B]**  
*Explanation*: Maraz (South) and Kamraz (North) are the primary regional dialect divisions.

---

### Question 4:
Why is character-level modeling robust to English code-switching in Kashmiri speech?
- [A] English letters are ignored.
- [B] English words transliterated into Perso-Arabic can be spelled character-by-character without triggering out-of-vocabulary (OOV) tokens.
- [C] The model translates English into Kashmiri automatically.
- [D] English words have zero WER.

**Correct Answer**: **[B]**  
*Explanation*: Character vocabularies can spell any phonetic transliteration.

---

### Question 5:
What rank $r$ is standardly used for LoRA adaptation in speech Transformers?
- [A] $r = 1$
- [B] $r = 8$
- [C] $r = 512$
- [D] $r = 1024$

**Correct Answer**: **[B]**  
*Explanation*: $r = 8$ provides optimal balance between expressivity and parameter efficiency.
