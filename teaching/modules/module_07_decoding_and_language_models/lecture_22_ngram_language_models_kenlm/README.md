# Lecture 22: N-gram Language Modeling & Kneser-Ney Smoothing
## Kashmiri Text Corpora, Modified Kneser-Ney Math, Continuation Probabilities, and KenLM Binary Formats

### Overview
This lecture is part of **Module 07**. It addresses both first-principles speech recognition science and the authentic empirical research findings of the EduCasheer Kashmiri ASR project.

### Learning Objectives
By the conclusion of this lecture, students will be able to:
- Formalize statistical language modeling as autoregressive conditional probability estimation.
- Derive Modified Kneser-Ney smoothing and understand continuation probability.
- Analyze the linguistic sparsity of Kashmiri text and the challenge of high morphological inflection.
- Train and query a 5-gram Kashmiri language model using the KenLM library.

### Document Directory
Every lecture folder contains the 11 standard educational documents:
1. [`lecture_notes.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_07_decoding_and_language_models/lecture_22_ngram_language_models_kenlm/lecture_notes.md): Comprehensive textbook chapter.
2. [`instructor_script.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_07_decoding_and_language_models/lecture_22_ngram_language_models_kenlm/instructor_script.md): Minute-by-minute teaching delivery script and blackboard plan.
3. [`mathematical_foundations.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_07_decoding_and_language_models/lecture_22_ngram_language_models_kenlm/mathematical_foundations.md): Complete mathematical derivations, proofs, and tensor transformations.
4. [`code_walkthrough.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_07_decoding_and_language_models/lecture_22_ngram_language_models_kenlm/code_walkthrough.md): Line-by-line inspection of production code from `src/`.
5. [`practical_lab.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_07_decoding_and_language_models/lecture_22_ngram_language_models_kenlm/practical_lab.md): Runnable laboratory exercise with verification scripts.
6. [`exercises.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_07_decoding_and_language_models/lecture_22_ngram_language_models_kenlm/exercises.md): Analytical, implementation, and forensic homework problems.
7. [`solutions.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_07_decoding_and_language_models/lecture_22_ngram_language_models_kenlm/solutions.md): Fully worked mathematical solutions and reference implementations.
8. [`quiz.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_07_decoding_and_language_models/lecture_22_ngram_language_models_kenlm/quiz.md): 5-question comprehension assessment with explanations.
9. [`research_connections.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_07_decoding_and_language_models/lecture_22_ngram_language_models_kenlm/research_connections.md): Direct mapping to repository stages, checkpoints, and historical bugs.
10. [`references.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_07_decoding_and_language_models/lecture_22_ngram_language_models_kenlm/references.md): Academic bibliography and reading list.
11. [`README.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_07_decoding_and_language_models/lecture_22_ngram_language_models_kenlm/README.md): This navigational overview.
