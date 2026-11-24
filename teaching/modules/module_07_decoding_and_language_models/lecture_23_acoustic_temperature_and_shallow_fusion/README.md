# Lecture 23: Acoustic Temperature & Shallow Fusion
## Softmax Calibration ($T=1.08$), Grid Search Optimization for $\alpha$ and $\beta$, and Certified Decoder Tuning

### Overview
This lecture is part of **Module 07**. It addresses both first-principles speech recognition science and the authentic empirical research findings of the EduCasheer Kashmiri ASR project.

### Learning Objectives
By the conclusion of this lecture, students will be able to:
- Analyze neural network overconfidence in CTC models and the need for acoustic temperature scaling $T$.
- Formulate Shallow Fusion decoding combining acoustic CTC log-probabilities, language model scores, and word bonus.
- Conduct a systematic 2D grid search over $(\alpha, \beta)$ to minimize Word Error Rate.
- Implement a calibrated Shallow Fusion beam search decoder in Python.

### Document Directory
Every lecture folder contains the 11 standard educational documents:
1. [`lecture_notes.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_07_decoding_and_language_models/lecture_23_acoustic_temperature_and_shallow_fusion/lecture_notes.md): Comprehensive textbook chapter.
2. [`instructor_script.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_07_decoding_and_language_models/lecture_23_acoustic_temperature_and_shallow_fusion/instructor_script.md): Minute-by-minute teaching delivery script and blackboard plan.
3. [`mathematical_foundations.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_07_decoding_and_language_models/lecture_23_acoustic_temperature_and_shallow_fusion/mathematical_foundations.md): Complete mathematical derivations, proofs, and tensor transformations.
4. [`code_walkthrough.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_07_decoding_and_language_models/lecture_23_acoustic_temperature_and_shallow_fusion/code_walkthrough.md): Line-by-line inspection of production code from `src/`.
5. [`practical_lab.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_07_decoding_and_language_models/lecture_23_acoustic_temperature_and_shallow_fusion/practical_lab.md): Runnable laboratory exercise with verification scripts.
6. [`exercises.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_07_decoding_and_language_models/lecture_23_acoustic_temperature_and_shallow_fusion/exercises.md): Analytical, implementation, and forensic homework problems.
7. [`solutions.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_07_decoding_and_language_models/lecture_23_acoustic_temperature_and_shallow_fusion/solutions.md): Fully worked mathematical solutions and reference implementations.
8. [`quiz.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_07_decoding_and_language_models/lecture_23_acoustic_temperature_and_shallow_fusion/quiz.md): 5-question comprehension assessment with explanations.
9. [`research_connections.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_07_decoding_and_language_models/lecture_23_acoustic_temperature_and_shallow_fusion/research_connections.md): Direct mapping to repository stages, checkpoints, and historical bugs.
10. [`references.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_07_decoding_and_language_models/lecture_23_acoustic_temperature_and_shallow_fusion/references.md): Academic bibliography and reading list.
11. [`README.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_07_decoding_and_language_models/lecture_23_acoustic_temperature_and_shallow_fusion/README.md): This navigational overview.
