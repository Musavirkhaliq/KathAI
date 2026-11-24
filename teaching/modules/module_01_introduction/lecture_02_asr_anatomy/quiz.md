# Quiz: Lecture 02 (The Anatomy of Modern ASR Pipelines)

---

### Question 1:
Why is a character-level vocabulary preferred over a subword (BPE) vocabulary for Kashmiri ASR in low-resource regimes?
- [A] Subwords require GPU acceleration during inference, whereas characters run on CPU.
- [B] Kashmiri Perso-Arabic orthography contains non-standardized spellings and central vowel diacritics that cause BPE to fragment words into arbitrary OOV sub-sequences without sufficient text training corpora.
- [C] CTC loss mathematically forbids any output vocabulary with cardinality $|V| > 100$.
- [D] Subwords are patented under proprietary licenses.

**Correct Answer**: **[B]**  
*Explanation*: BPE tokenizers rely on extensive text corpora to calculate stable merge frequencies. In low-resource Kashmiri with non-standardized diacritics, BPE creates brittle fragments that compound acoustic errors. Character-level tokenization provides zero OOV and perfectly captures Perso-Arabic morphology.

---

### Question 2:
What is the primary role of the CTC blank token ($\epsilon$)?
- [A] To indicate physical silence or acoustic background noise in the room.
- [B] To separate consecutive identical characters and allow the model to emit nothing when acoustic evidence is ambiguous.
- [C] To signal the end of the spoken sentence.
- [D] To trigger beam pruning in the language model decoder.

**Correct Answer**: **[B]**  
*Explanation*: The blank token is a non-emitting alignment construct that allows consecutive identical characters (e.g., /ʃ/ and /ʃ/) to be distinguished from a single sustained character, while allowing frames with transition uncertainty to emit no linguistic token.

---

### Question 3:
If the acoustic feature extractor has a stride of 320 at 16 kHz, what is the temporal duration represented by each acoustic frame?
- [A] 10 milliseconds
- [B] 20 milliseconds
- [C] 32 milliseconds
- [D] 50 milliseconds

**Correct Answer**: **[B]**  
*Explanation*: $\Delta t = \frac{320 \text{ samples}}{16,000 \text{ samples/sec}} = 0.020 \text{ seconds} = 20 \text{ ms}$.

---

### Question 4:
What occurs during inference if the language model weight $\alpha$ is tuned excessively high (e.g., $\alpha = 2.5$)?
- [A] The model experiences out-of-memory errors due to CTC tensor explosion.
- [B] The decoder ignores genuine acoustic evidence and replaces phonetically accurate proper nouns with common high-frequency n-grams.
- [C] All acoustic frames collapse to the blank token.
- [D] The audio sample rate automatically downsamples to 8 kHz.

**Correct Answer**: **[B]**  
*Explanation*: Overweighting the language model shifts the objective away from acoustic fidelity toward grammatical likelihood, causing hallucinations and substituting unusual acoustic tokens with common words.

---

### Question 5:
Why must speech evaluation strictly enforce disjoint speaker splits?
- [A] To prevent CTC loss from diverging during backpropagation.
- [B] Because deep neural acoustic models readily memorize speaker vocal tract characteristics, producing artificially low WERs that fail to generalize to unseen speakers.
- [C] Because torchaudio cannot load audio files from the same speaker simultaneously.
- [D] To satisfy Unicode standard RFC-4648.

**Correct Answer**: **[B]**  
*Explanation*: Neural encoders readily overfit to speaker-specific acoustic features ($F_0$, formants, room acoustics). Disjoint speaker splits ensure that reported benchmarks reflect true generalizability.
