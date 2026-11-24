# Annotated References: Lecture 02

---

1. **Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J. (2006).**  
   *Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks.*  
   *Proceedings of the 23rd International Conference on Machine Learning (ICML)*, 369–376.  
   - **Annotation**: Foundational paper establishing CTC loss, alignment collapse operator $\mathcal{B}$, and prefix beam search decoding.

2. **Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020).**  
   *wav2vec 2.0: A framework for self-supervised learning of speech representations.*  
   *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 12449–12460.  
   - **Annotation**: Introduces raw waveform self-supervised contrastive learning that forms the acoustic backbone of modern low-resource ASR.

3. **Babu, A., Wang, C., Tjandra, A., et al. (2021).**  
   *XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale.*  
   *arXiv preprint arXiv:2111.09296*.  
   - **Annotation**: Describes the 128-language, 436,000-hour pretrained model utilized as the initialization checkpoint for EduCasheer Kashmiri ASR.

4. **Hannun, A. Y., Maas, A. L., Jurafsky, D., & Ng, A. Y. (2014).**  
   *First-pass large vocabulary continuous speech recognition using bi-directional recurrent DNNs.*  
   *arXiv preprint arXiv:1408.2876*.  
   - **Annotation**: Details the practical implementation of CTC beam search with language model shallow fusion and word insertion penalties.
