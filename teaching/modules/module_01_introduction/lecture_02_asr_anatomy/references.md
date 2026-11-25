# Academic References: Lecture 02 (The Anatomy of Modern ASR Pipelines)

---

## 1. Primary Foundational Literature

1. **Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J.** (2006). *Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks*. In Proceedings of the 23rd International Conference on Machine Learning (ICML '06), pp. 369–376.
   - *Significance*: Introduced the CTC loss function, the collapse operator $\mathcal{B}$, and the forward-backward dynamic programming algorithm for monotonic sequence alignment without pre-segmented phonetic boundaries.

2. **Baevski, A., Zhou, Y., Mohamed, A., & Auli, M.** (2020). *wav2vec 2.0: A framework for self-supervised learning of speech representations*. Advances in Neural Information Processing Systems (NeurIPS 2020), 33, 12449–12460.
   - *Significance*: Established the 7-layer temporal Conv1d feature extractor architecture and the Transformer-based contrastive self-supervised pretraining framework used as the acoustic backbone in EduCasheer.

3. **Babu, A., Wang, C., Tjandra, A., Lakhotia, K., Xu, Q., Goyal, N., Singh, K., von Platen, P., Saraf, Y., Pino, J., Baevski, A., & Auli, M.** (2021). *XLS-R: Self-supervised cross-lingual speech representation learning at scale*. arXiv preprint arXiv:2111.09296.
   - *Significance*: Detailed the 128-language multilingual pretraining of XLS-R (including Indo-Aryan and Dardic speech representations), providing the transfer learning base for Kashmiri acoustic modeling.

4. **Hannun, A. Y., Annapureddy, C., Xu, G., Maas, A. L., & Ng, A. Y.** (2014). *First-pass large vocabulary continuous speech recognition using bi-directional recurrent DNNs*. arXiv preprint arXiv:1408.2868.
   - *Significance*: Introduced the prefix beam search algorithm for CTC decoding with shallow $n$-gram language model integration and the word insertion bonus $\beta$.

5. **Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I.** (2023). *Robust speech recognition via large-scale weak supervision*. In International Conference on Machine Learning (ICML 2023), pp. 28492–28518.
   - *Significance*: Documented the Attention-based Encoder-Decoder architecture of Whisper, providing empirical baselines and highlighting hallucination failure modes on low-resource and noisy speech.

---

## 2. Kashmiri Linguistics and Orthography

6. **Koul, O. N.** (2005). *A Kashmiri Dictionary*. Central Institute of Indian Languages (CIIL), Mysore.
   - *Significance*: Standard reference for Kashmiri Perso-Arabic orthography, documenting the phonemic status of central vowels and palatalization markers.

7. **Wali, K., & Koul, O. N.** (1997). *Kashmiri: A Cognitive-Descriptive Grammar*. Routledge, London.
   - *Significance*: Exhaustive morphological and syntactic analysis of Kashmiri, providing grammatical foundations for language model design and text normalization rules.

8. **Grierson, G. A.** (1915). *A Dictionary of the Kashmiri Language*. Royal Asiatic Society of Bengal, Calcutta.
   - *Significance*: Historical linguistic documentation of Dardic vowel systems and phonetic length distinctions in the Kashmir Valley.
