# Lecture 18: Fine-Tuning Dynamics, Schedules & Warmup
## Learning Rate Schedules, Linear Warmup, Cosine Annealing, and Discriminative Layer Fine-Tuning

### Overview
This lecture is part of **Module 06**. It addresses both first-principles speech recognition science and the authentic empirical research findings of the EduCasheer Kashmiri ASR project.

### Learning Objectives
By the conclusion of this lecture, students will be able to:
- Deconstruct the optimization dynamics of fine-tuning large self-supervised models on small target corpora.
- Formulate linear learning rate warmup and cosine decay scheduling.
- Analyze the risks of catastrophic forgetting when fine-tuning lower versus upper Transformer layers.
- Implement a customized PyTorch learning rate scheduler with layer-wise decay.

### Document Directory
Every lecture folder contains the 11 standard educational documents:
1. [`lecture_notes.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_06_training_and_optimization/lecture_18_finetuning_dynamics_and_schedules/lecture_notes.md): Comprehensive textbook chapter.
2. [`instructor_script.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_06_training_and_optimization/lecture_18_finetuning_dynamics_and_schedules/instructor_script.md): Minute-by-minute teaching delivery script and blackboard plan.
3. [`mathematical_foundations.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_06_training_and_optimization/lecture_18_finetuning_dynamics_and_schedules/mathematical_foundations.md): Complete mathematical derivations, proofs, and tensor transformations.
4. [`code_walkthrough.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_06_training_and_optimization/lecture_18_finetuning_dynamics_and_schedules/code_walkthrough.md): Line-by-line inspection of production code from `src/`.
5. [`practical_lab.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_06_training_and_optimization/lecture_18_finetuning_dynamics_and_schedules/practical_lab.md): Runnable laboratory exercise with verification scripts.
6. [`exercises.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_06_training_and_optimization/lecture_18_finetuning_dynamics_and_schedules/exercises.md): Analytical, implementation, and forensic homework problems.
7. [`solutions.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_06_training_and_optimization/lecture_18_finetuning_dynamics_and_schedules/solutions.md): Fully worked mathematical solutions and reference implementations.
8. [`quiz.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_06_training_and_optimization/lecture_18_finetuning_dynamics_and_schedules/quiz.md): 5-question comprehension assessment with explanations.
9. [`research_connections.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_06_training_and_optimization/lecture_18_finetuning_dynamics_and_schedules/research_connections.md): Direct mapping to repository stages, checkpoints, and historical bugs.
10. [`references.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_06_training_and_optimization/lecture_18_finetuning_dynamics_and_schedules/references.md): Academic bibliography and reading list.
11. [`README.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_06_training_and_optimization/lecture_18_finetuning_dynamics_and_schedules/README.md): This navigational overview.
