# Lecture 12: Self-Attention & Multi-Head Transformers
## Scaled Dot-Product Attention, Relative Positional Embeddings, and Contextual Speech Modeling

### Overview
This lecture is part of **Module 04**. It addresses both first-principles speech recognition science and the authentic empirical research findings of the EduCasheer Kashmiri ASR project.

### Learning Objectives
By the conclusion of this lecture, students will be able to:
- Derive the mathematical formulation of Scaled Dot-Product Attention and Multi-Head Attention (MHA).
- Contrast absolute sinusoidal positional embeddings against convolutional relative positional encodings for speech.
- Analyze the computational complexity and memory footprint of self-attention as a function of sequence length $T$.
- Implement a multi-head self-attention module and inspect attention heatmaps across Kashmiri phonemes.

### Document Directory
Every lecture folder contains the 11 standard educational documents:
1. [`lecture_notes.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_04_deep_learning_for_asr/lecture_12_self_attention_and_transformers/lecture_notes.md): Comprehensive textbook chapter.
2. [`instructor_script.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_04_deep_learning_for_asr/lecture_12_self_attention_and_transformers/instructor_script.md): Minute-by-minute teaching delivery script and blackboard plan.
3. [`mathematical_foundations.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_04_deep_learning_for_asr/lecture_12_self_attention_and_transformers/mathematical_foundations.md): Complete mathematical derivations, proofs, and tensor transformations.
4. [`code_walkthrough.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_04_deep_learning_for_asr/lecture_12_self_attention_and_transformers/code_walkthrough.md): Line-by-line inspection of production code from `src/`.
5. [`practical_lab.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_04_deep_learning_for_asr/lecture_12_self_attention_and_transformers/practical_lab.md): Runnable laboratory exercise with verification scripts.
6. [`exercises.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_04_deep_learning_for_asr/lecture_12_self_attention_and_transformers/exercises.md): Analytical, implementation, and forensic homework problems.
7. [`solutions.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_04_deep_learning_for_asr/lecture_12_self_attention_and_transformers/solutions.md): Fully worked mathematical solutions and reference implementations.
8. [`quiz.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_04_deep_learning_for_asr/lecture_12_self_attention_and_transformers/quiz.md): 5-question comprehension assessment with explanations.
9. [`research_connections.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_04_deep_learning_for_asr/lecture_12_self_attention_and_transformers/research_connections.md): Direct mapping to repository stages, checkpoints, and historical bugs.
10. [`references.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_04_deep_learning_for_asr/lecture_12_self_attention_and_transformers/references.md): Academic bibliography and reading list.
11. [`README.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_04_deep_learning_for_asr/lecture_12_self_attention_and_transformers/README.md): This navigational overview.
