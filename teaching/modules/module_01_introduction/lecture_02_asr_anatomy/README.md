# Lecture 02: Anatomy of Modern ASR Pipelines
## Modular Architecture, Acoustic Encoders, Prediction Networks, and Vocabulary Construction

### Overview
This lecture is part of **Module 01**. It addresses both first-principles speech recognition science and the authentic empirical research findings of the EduCasheer Kashmiri ASR project.

### Learning Objectives
By the conclusion of this lecture, students will be able to:
- Deconstruct the complete end-to-end neural ASR pipeline into feature extraction, encoder, decoder, and search heads.
- Compare the three dominant paradigm families: CTC, Recurrent Neural Network Transducer (RNN-T), and Attention-based Encoder-Decoder (AED).
- Understand why CTC on self-supervised representations was selected for the EduCasheer Kashmiri production architecture.
- Formulate the tokenization and vocabulary design choices for Kashmiri Perso-Arabic script.

### Document Directory
Every lecture folder contains the 11 standard educational documents:
1. [`lecture_notes.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_01_introduction/lecture_02_asr_anatomy/lecture_notes.md): Comprehensive textbook chapter.
2. [`instructor_script.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_01_introduction/lecture_02_asr_anatomy/instructor_script.md): Minute-by-minute teaching delivery script and blackboard plan.
3. [`mathematical_foundations.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_01_introduction/lecture_02_asr_anatomy/mathematical_foundations.md): Complete mathematical derivations, proofs, and tensor transformations.
4. [`code_walkthrough.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_01_introduction/lecture_02_asr_anatomy/code_walkthrough.md): Line-by-line inspection of production code from `src/`.
5. [`practical_lab.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_01_introduction/lecture_02_asr_anatomy/practical_lab.md): Runnable laboratory exercise with verification scripts.
6. [`exercises.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_01_introduction/lecture_02_asr_anatomy/exercises.md): Analytical, implementation, and forensic homework problems.
7. [`solutions.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_01_introduction/lecture_02_asr_anatomy/solutions.md): Fully worked mathematical solutions and reference implementations.
8. [`quiz.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_01_introduction/lecture_02_asr_anatomy/quiz.md): 5-question comprehension assessment with explanations.
9. [`research_connections.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_01_introduction/lecture_02_asr_anatomy/research_connections.md): Direct mapping to repository stages, checkpoints, and historical bugs.
10. [`references.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_01_introduction/lecture_02_asr_anatomy/references.md): Academic bibliography and reading list.
11. [`README.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_01_introduction/lecture_02_asr_anatomy/README.md): This navigational overview.
