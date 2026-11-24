# Lecture 01: What is ASR? Speech Recognition Formulation
## Probabilistic Foundations, Acoustic-Language Decomposition, and Low-Resource Frontiers

### Overview
This lecture is part of **Module 01**. It addresses both first-principles speech recognition science and the authentic empirical research findings of the EduCasheer Kashmiri ASR project.

### Learning Objectives
By the conclusion of this lecture, students will be able to:
- Formalize Automatic Speech Recognition as maximum a posteriori (MAP) sequence decoding.
- Deconstruct the Fundamental Equation of Speech Recognition into acoustic and language modeling components.
- Analyze the unique mathematical and phonetic hurdles of Kashmiri as an Indo-Aryan language with Perso-Arabic orthography.
- Inspect the fundamental taxonomy of speech systems: HMM-GMM, hybrid DNN, and end-to-end neural pipelines.

### Document Directory
Every lecture folder contains the 11 standard educational documents:
1. [`lecture_notes.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_01_introduction/lecture_01_what_is_asr/lecture_notes.md): Comprehensive textbook chapter.
2. [`instructor_script.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_01_introduction/lecture_01_what_is_asr/instructor_script.md): Minute-by-minute teaching delivery script and blackboard plan.
3. [`mathematical_foundations.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_01_introduction/lecture_01_what_is_asr/mathematical_foundations.md): Complete mathematical derivations, proofs, and tensor transformations.
4. [`code_walkthrough.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_01_introduction/lecture_01_what_is_asr/code_walkthrough.md): Line-by-line inspection of production code from `src/`.
5. [`practical_lab.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_01_introduction/lecture_01_what_is_asr/practical_lab.md): Runnable laboratory exercise with verification scripts.
6. [`exercises.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_01_introduction/lecture_01_what_is_asr/exercises.md): Analytical, implementation, and forensic homework problems.
7. [`solutions.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_01_introduction/lecture_01_what_is_asr/solutions.md): Fully worked mathematical solutions and reference implementations.
8. [`quiz.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_01_introduction/lecture_01_what_is_asr/quiz.md): 5-question comprehension assessment with explanations.
9. [`research_connections.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_01_introduction/lecture_01_what_is_asr/research_connections.md): Direct mapping to repository stages, checkpoints, and historical bugs.
10. [`references.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_01_introduction/lecture_01_what_is_asr/references.md): Academic bibliography and reading list.
11. [`README.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_01_introduction/lecture_01_what_is_asr/README.md): This navigational overview.
