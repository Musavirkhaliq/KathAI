# Lecture 07: Text Normalization & Unicode Standardization
## NFD vs. NFC Normalization, Zero-Width Characters, Ligature Stripping, and The Certified Normalizer v3

### Overview
This lecture is part of **Module 03**. It addresses both first-principles speech recognition science and the authentic empirical research findings of the EduCasheer Kashmiri ASR project.

### Learning Objectives
By the conclusion of this lecture, students will be able to:
- Deconstruct Unicode normalization standards (NFC, NFD, NFKC, NFKD) and their impact on text processing.
- Identify and neutralize non-printing control characters (ZWNJ `U+200C`, ZWJ `U+200D`, LTR/RTL marks).
- Trace the evolution of the EduCasheer Kashmiri normalizer from Stage 01 heuristics to Certified Normalizer v3.
- Implement a production-grade normalizer pipeline with automated regression unit tests.

### Document Directory
Every lecture folder contains the 11 standard educational documents:
1. [`lecture_notes.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_03_kashmiri_language_and_data/lecture_07_text_normalization/lecture_notes.md): Comprehensive textbook chapter.
2. [`instructor_script.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_03_kashmiri_language_and_data/lecture_07_text_normalization/instructor_script.md): Minute-by-minute teaching delivery script and blackboard plan.
3. [`mathematical_foundations.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_03_kashmiri_language_and_data/lecture_07_text_normalization/mathematical_foundations.md): Complete mathematical derivations, proofs, and tensor transformations.
4. [`code_walkthrough.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_03_kashmiri_language_and_data/lecture_07_text_normalization/code_walkthrough.md): Line-by-line inspection of production code from `src/`.
5. [`practical_lab.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_03_kashmiri_language_and_data/lecture_07_text_normalization/practical_lab.md): Runnable laboratory exercise with verification scripts.
6. [`exercises.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_03_kashmiri_language_and_data/lecture_07_text_normalization/exercises.md): Analytical, implementation, and forensic homework problems.
7. [`solutions.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_03_kashmiri_language_and_data/lecture_07_text_normalization/solutions.md): Fully worked mathematical solutions and reference implementations.
8. [`quiz.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_03_kashmiri_language_and_data/lecture_07_text_normalization/quiz.md): 5-question comprehension assessment with explanations.
9. [`research_connections.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_03_kashmiri_language_and_data/lecture_07_text_normalization/research_connections.md): Direct mapping to repository stages, checkpoints, and historical bugs.
10. [`references.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_03_kashmiri_language_and_data/lecture_07_text_normalization/references.md): Academic bibliography and reading list.
11. [`README.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_03_kashmiri_language_and_data/lecture_07_text_normalization/README.md): This navigational overview.
