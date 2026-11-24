# Quiz: Lecture 13 (Self-Supervised Learning and XLS-R)

---

### Question 1:
What is the primary advantage of self-supervised speech pretraining for low-resource languages like Kashmiri?
- [A] It eliminates the need for GPUs during training.
- [B] It pretrains deep acoustic and phonetic representations from unannotated audio, dramatically reducing the amount of transcribed speech needed for fine-tuning.
- [C] It automatically translates Kashmiri speech into English text.
- [D] It compresses audio files by 90%.

**Correct Answer**: **[B]**  
*Explanation*: SSL builds universal speech representations on hundreds of thousands of hours of unlabelled speech.

---

### Question 2:
How many hours of speech and how many languages were used to pretrain the XLS-R model?
- [A] 1,000 hours across 10 languages
- [B] 10,000 hours across 50 languages
- [C] 436,000 hours across 128 languages
- [D] 1,000,000 hours across 500 languages

**Correct Answer**: **[C]**  
*Explanation*: XLS-R was trained on 436,000 hours across 128 diverse languages.

---

### Question 3:
What is the function of the Gumbel-Softmax vector quantizer in wav2vec 2.0?
- [A] To downsample audio from 48 kHz to 16 kHz.
- [B] To discretize continuous convolutional latents into discrete codebook targets while permitting gradient backpropagation during pretraining.
- [C] To remove background room reverberation.
- [D] To decode text hypotheses.

**Correct Answer**: **[B]**  
*Explanation*: Gumbel-Softmax provides a differentiable relaxation of categorical codebook selection.

---

### Question 4:
What percentage of time frames are typically selected as mask start points in wav2vec 2.0?
- [A] 0.5%
- [B] 6.5% (with span length 10 frames, masking ~50% of total frames)
- [C] 25.0%
- [D] 90.0%

**Correct Answer**: **[B]**  
*Explanation*: Masking starts at 6.5% of frames and extends across 10-frame chunks, covering approximately half the sequence.

---

### Question 5:
What loss function is paired with the masked contrastive loss to prevent codebook collapse?
- [A] CTC loss
- [B] Codebook diversity loss ($\mathcal{L}_d$)
- [C] Mean Squared Error
- [D] Binary Cross-Entropy

**Correct Answer**: **[B]**  
*Explanation*: Diversity loss maximizes codebook entropy, ensuring all codebook entries are actively utilized.
