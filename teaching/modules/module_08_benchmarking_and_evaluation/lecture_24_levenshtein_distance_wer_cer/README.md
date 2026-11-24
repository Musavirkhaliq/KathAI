# Lecture 24: Levenshtein Distance, Word & Character Error Rate
## Dynamic Programming Edit Distance, Substitution/Deletion/Insertion Accounting, and Certified Evaluation Engines

### Overview
This lecture is part of **Module 08**. It addresses both first-principles speech recognition science and the authentic empirical research findings of the EduCasheer Kashmiri ASR project.

### Learning Objectives
By the conclusion of this lecture, students will be able to:
- Derive the Levenshtein dynamic programming algorithm for minimum edit distance.
- Formulate the exact definitions of Word Error Rate (WER) and Character Error Rate (CER).
- Analyze the pathological case where WER exceeds $100\%$ due to insertion errors ($I > N$).
- Implement a high-performance Levenshtein evaluation engine in Python.

### Document Directory
Every lecture folder contains the 11 standard educational documents:
1. [`lecture_notes.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_08_benchmarking_and_evaluation/lecture_24_levenshtein_distance_wer_cer/lecture_notes.md): Comprehensive textbook chapter.
2. [`instructor_script.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_08_benchmarking_and_evaluation/lecture_24_levenshtein_distance_wer_cer/instructor_script.md): Minute-by-minute teaching delivery script and blackboard plan.
3. [`mathematical_foundations.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_08_benchmarking_and_evaluation/lecture_24_levenshtein_distance_wer_cer/mathematical_foundations.md): Complete mathematical derivations, proofs, and tensor transformations.
4. [`code_walkthrough.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_08_benchmarking_and_evaluation/lecture_24_levenshtein_distance_wer_cer/code_walkthrough.md): Line-by-line inspection of production code from `src/`.
5. [`practical_lab.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_08_benchmarking_and_evaluation/lecture_24_levenshtein_distance_wer_cer/practical_lab.md): Runnable laboratory exercise with verification scripts.
6. [`exercises.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_08_benchmarking_and_evaluation/lecture_24_levenshtein_distance_wer_cer/exercises.md): Analytical, implementation, and forensic homework problems.
7. [`solutions.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_08_benchmarking_and_evaluation/lecture_24_levenshtein_distance_wer_cer/solutions.md): Fully worked mathematical solutions and reference implementations.
8. [`quiz.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_08_benchmarking_and_evaluation/lecture_24_levenshtein_distance_wer_cer/quiz.md): 5-question comprehension assessment with explanations.
9. [`research_connections.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_08_benchmarking_and_evaluation/lecture_24_levenshtein_distance_wer_cer/research_connections.md): Direct mapping to repository stages, checkpoints, and historical bugs.
10. [`references.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_08_benchmarking_and_evaluation/lecture_24_levenshtein_distance_wer_cer/references.md): Academic bibliography and reading list.
11. [`README.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_08_benchmarking_and_evaluation/lecture_24_levenshtein_distance_wer_cer/README.md): This navigational overview.
