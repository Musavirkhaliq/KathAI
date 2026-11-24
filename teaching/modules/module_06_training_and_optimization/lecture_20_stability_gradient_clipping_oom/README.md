# Lecture 20: Training Stability, Gradient Clamping & OOMs
## Debugging NaN Losses, CUDA Out-Of-Memory Recovery, Gradient Norm Clamping, and Memory Profiling

### Overview
This lecture is part of **Module 06**. It addresses both first-principles speech recognition science and the authentic empirical research findings of the EduCasheer Kashmiri ASR project.

### Learning Objectives
By the conclusion of this lecture, students will be able to:
- Diagnose and eliminate the three major failure modes of speech training: NaN loss, gradient explosion, and CUDA OOM.
- Formulate global gradient norm clipping and adaptive gradient clamping.
- Analyze CUDA memory fragmentation and implement PyTorch memory profiling.
- Construct an automated recovery wrapper for interrupted training runs.

### Document Directory
Every lecture folder contains the 11 standard educational documents:
1. [`lecture_notes.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_06_training_and_optimization/lecture_20_stability_gradient_clipping_oom/lecture_notes.md): Comprehensive textbook chapter.
2. [`instructor_script.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_06_training_and_optimization/lecture_20_stability_gradient_clipping_oom/instructor_script.md): Minute-by-minute teaching delivery script and blackboard plan.
3. [`mathematical_foundations.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_06_training_and_optimization/lecture_20_stability_gradient_clipping_oom/mathematical_foundations.md): Complete mathematical derivations, proofs, and tensor transformations.
4. [`code_walkthrough.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_06_training_and_optimization/lecture_20_stability_gradient_clipping_oom/code_walkthrough.md): Line-by-line inspection of production code from `src/`.
5. [`practical_lab.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_06_training_and_optimization/lecture_20_stability_gradient_clipping_oom/practical_lab.md): Runnable laboratory exercise with verification scripts.
6. [`exercises.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_06_training_and_optimization/lecture_20_stability_gradient_clipping_oom/exercises.md): Analytical, implementation, and forensic homework problems.
7. [`solutions.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_06_training_and_optimization/lecture_20_stability_gradient_clipping_oom/solutions.md): Fully worked mathematical solutions and reference implementations.
8. [`quiz.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_06_training_and_optimization/lecture_20_stability_gradient_clipping_oom/quiz.md): 5-question comprehension assessment with explanations.
9. [`research_connections.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_06_training_and_optimization/lecture_20_stability_gradient_clipping_oom/research_connections.md): Direct mapping to repository stages, checkpoints, and historical bugs.
10. [`references.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_06_training_and_optimization/lecture_20_stability_gradient_clipping_oom/references.md): Academic bibliography and reading list.
11. [`README.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_06_training_and_optimization/lecture_20_stability_gradient_clipping_oom/README.md): This navigational overview.
