# Course Prerequisites & Technical Roadmap
===========================================

This course is designed to be self-contained, deriving core speech processing and machine learning principles from scratch. However, students will gain the most from the lectures and labs with the following foundational background.

---

## 1. Mathematical Foundations

### 1.1 Linear Algebra
- **Vectors, Matrices, and Tensors**: Vector spaces, inner products, matrix multiplication, transpose, rank, and multidimensional array manipulation.
- **Eigenvalues & Decompositions**: Singular Value Decomposition (SVD), Principal Component Analysis (PCA) intuition.
- **Norms & Projections**: $L_1$, $L_2$, Frobenius norms, cosine similarity, orthogonal projections.

### 1.2 Multivariate Calculus & Optimization
- **Partial Derivatives & Gradients**: Scalar-by-vector gradients, Jacobians, Hessians.
- **The Chain Rule**: Reverse-mode automatic differentiation (backpropagation).
- **Optimization Algorithms**: Gradient Descent, Stochastic Gradient Descent (SGD) with momentum, AdamW optimizer dynamics.

### 1.3 Probability & Statistics
- **Discrete & Continuous Random Variables**: Probability Density Functions (PDFs), Cumulative Distribution Functions (CDFs).
- **Conditional Probability & Bayes' Theorem**: $P(Y|X) = rac{P(X|Y)P(Y)}{P(X)}$.
- **Maximum Likelihood Estimation (MLE)**: Log-likelihood maximization, cross-entropy, negative log-likelihood loss.
- **Statistical Significance**: Hypothesis testing, $p$-values, confidence intervals, paired bootstrap resampling, Wilcoxon signed-rank test.

### 1.4 Digital Signal Processing (Helpful but taught in Module 2)
- Continuous vs discrete signals, Nyquist-Shannon sampling theorem, sinusoidal decomposition, Fourier analysis.

---

## 2. Software & Programming Prerequisites

### 2.1 Python 3.10+
- Object-oriented programming, classes, inheritance, dataclasses.
- Typing hints (`typing.List`, `typing.Dict`, `typing.Optional`, `typing.Tuple`).
- Generators, iterators, context managers (`with` statements).
- File I/O (`pathlib`, `json`, `csv`, `yaml`).

### 2.2 Numerical & Deep Learning Libraries
- **NumPy**: Array creation, broadcasting rules, matrix operations, indexing and slicing.
- **PyTorch**: Tensors, computational graphs, autograd, `nn.Module`, `DataLoader`, `Dataset`, training loops, CUDA device management.
- **Hugging Face Transformers**: Model configurations, pretrained weight loading (`Wav2Vec2ForCTC`, `Wav2Vec2Processor`), tokenizer dictionaries.
- **SoundFile & SciPy**: Audio loading, WAV parsing, windowed sinc resampling (`scipy.signal.resample`).
- **pyctcdecode & KenLM**: Prefix beam search decoding, n-gram language model loading, language model shallow fusion.

---

## 3. Recommended Review Reading

1. **Deep Learning**: Goodfellow, Bengio, Courville, *Deep Learning* (MIT Press, 2016). Chapters 2 (Linear Algebra), 3 (Probability), 6 (Deep Feedforward Networks).
2. **Speech Recognition**: Jurafsky & Martin, *Speech and Language Processing* (3rd ed. draft, 2024). Chapter 16 (Automatic Speech Recognition).
3. **Signal Processing**: Oppenheim & Schafer, *Discrete-Time Signal Processing* (Prentice Hall). Chapters 2 & 4 (Sampling).
