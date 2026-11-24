# Lecture 21: Greedy Decoding & CTC Prefix Beam Search
## Argmax Collapse, The Prefix Merging Algorithm, Blank vs. Non-Blank Extensions, and Pruning Heuristics

### Overview
This lecture is part of **Module 07**. It addresses both first-principles speech recognition science and the authentic empirical research findings of the EduCasheer Kashmiri ASR project.

### Learning Objectives
By the conclusion of this lecture, students will be able to:
- Deconstruct greedy decoding (argmax path collapse) and explain why $\mathcal{B}(\arg\max \pi) \ne \arg\max P(\mathbf{y} \mid \mathbf{X})$.
- Derive the CTC Prefix Beam Search dynamic programming algorithm over prefix trees.
- Formulate the mathematical distinction between blank probability $P_b$ and non-blank probability $P_{nb}$ extensions.
- Implement a standalone CTC Prefix Beam Search decoder in Python.

### Document Directory
Every lecture folder contains the 11 standard educational documents:
1. [`lecture_notes.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_07_decoding_and_language_models/lecture_21_greedy_and_beam_search/lecture_notes.md): Comprehensive textbook chapter.
2. [`instructor_script.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_07_decoding_and_language_models/lecture_21_greedy_and_beam_search/instructor_script.md): Minute-by-minute teaching delivery script and blackboard plan.
3. [`mathematical_foundations.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_07_decoding_and_language_models/lecture_21_greedy_and_beam_search/mathematical_foundations.md): Complete mathematical derivations, proofs, and tensor transformations.
4. [`code_walkthrough.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_07_decoding_and_language_models/lecture_21_greedy_and_beam_search/code_walkthrough.md): Line-by-line inspection of production code from `src/`.
5. [`practical_lab.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_07_decoding_and_language_models/lecture_21_greedy_and_beam_search/practical_lab.md): Runnable laboratory exercise with verification scripts.
6. [`exercises.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_07_decoding_and_language_models/lecture_21_greedy_and_beam_search/exercises.md): Analytical, implementation, and forensic homework problems.
7. [`solutions.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_07_decoding_and_language_models/lecture_21_greedy_and_beam_search/solutions.md): Fully worked mathematical solutions and reference implementations.
8. [`quiz.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_07_decoding_and_language_models/lecture_21_greedy_and_beam_search/quiz.md): 5-question comprehension assessment with explanations.
9. [`research_connections.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_07_decoding_and_language_models/lecture_21_greedy_and_beam_search/research_connections.md): Direct mapping to repository stages, checkpoints, and historical bugs.
10. [`references.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_07_decoding_and_language_models/lecture_21_greedy_and_beam_search/references.md): Academic bibliography and reading list.
11. [`README.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_07_decoding_and_language_models/lecture_21_greedy_and_beam_search/README.md): This navigational overview.
