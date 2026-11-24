# Lecture 15: Connectionist Temporal Classification: Math & DP
## Forward-Backward Dynamic Programming, Exact CTC Loss Derivation, and Analytical Softmax Gradients

### Overview
This lecture is part of **Module 05**. It addresses both first-principles speech recognition science and the authentic empirical research findings of the EduCasheer Kashmiri ASR project.

### Learning Objectives
By the conclusion of this lecture, students will be able to:
- Derive the complete mathematical formulation of CTC conditional sequence probability $P(\mathbf{y} \mid \mathbf{X})$.
- Formulate the Forward-Backward dynamic programming algorithm over modified target sequence $\mathbf{l}'$.
- Derive the analytical gradient of the CTC loss with respect to unnormalized neural network logits $u_t^k$.
- Implement a numerically stable log-domain CTC loss function in PyTorch.

### Document Directory
Every lecture folder contains the 11 standard educational documents:
1. [`lecture_notes.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_05_ctc_and_acoustic_models/lecture_15_ctc_loss_mathematics/lecture_notes.md): Comprehensive textbook chapter.
2. [`instructor_script.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_05_ctc_and_acoustic_models/lecture_15_ctc_loss_mathematics/instructor_script.md): Minute-by-minute teaching delivery script and blackboard plan.
3. [`mathematical_foundations.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_05_ctc_and_acoustic_models/lecture_15_ctc_loss_mathematics/mathematical_foundations.md): Complete mathematical derivations, proofs, and tensor transformations.
4. [`code_walkthrough.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_05_ctc_and_acoustic_models/lecture_15_ctc_loss_mathematics/code_walkthrough.md): Line-by-line inspection of production code from `src/`.
5. [`practical_lab.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_05_ctc_and_acoustic_models/lecture_15_ctc_loss_mathematics/practical_lab.md): Runnable laboratory exercise with verification scripts.
6. [`exercises.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_05_ctc_and_acoustic_models/lecture_15_ctc_loss_mathematics/exercises.md): Analytical, implementation, and forensic homework problems.
7. [`solutions.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_05_ctc_and_acoustic_models/lecture_15_ctc_loss_mathematics/solutions.md): Fully worked mathematical solutions and reference implementations.
8. [`quiz.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_05_ctc_and_acoustic_models/lecture_15_ctc_loss_mathematics/quiz.md): 5-question comprehension assessment with explanations.
9. [`research_connections.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_05_ctc_and_acoustic_models/lecture_15_ctc_loss_mathematics/research_connections.md): Direct mapping to repository stages, checkpoints, and historical bugs.
10. [`references.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_05_ctc_and_acoustic_models/lecture_15_ctc_loss_mathematics/references.md): Academic bibliography and reading list.
11. [`README.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_05_ctc_and_acoustic_models/lecture_15_ctc_loss_mathematics/README.md): This navigational overview.
