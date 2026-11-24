# Lecture 11: Temporal Convolutions & Receptive Fields
## Strided Convolutions, Receptive Field Mathematics, and the Wav2Vec2 7-Layer CNN Encoder

### Overview
This lecture is part of **Module 04**. It addresses both first-principles speech recognition science and the authentic empirical research findings of the EduCasheer Kashmiri ASR project.

### Learning Objectives
By the conclusion of this lecture, students will be able to:
- Derive the exact analytical formula for receptive field growth in multi-layer strided convolutional networks.
- Deconstruct the architecture of the 7-layer temporal feature encoder used in Wav2Vec 2.0 and XLS-R.
- Verify mathematically how the downsampling factor $R=320$ translates 16 kHz audio into 50 Hz latent speech representations.
- Inspect the PyTorch implementation of `Wav2Vec2FeatureEncoder` in `src/asr/wav2vec2_cnn.py`.

### Document Directory
Every lecture folder contains the 11 standard educational documents:
1. [`lecture_notes.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_04_deep_learning_for_asr/lecture_11_temporal_convolutions_and_receptive_fields/lecture_notes.md): Comprehensive textbook chapter.
2. [`instructor_script.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_04_deep_learning_for_asr/lecture_11_temporal_convolutions_and_receptive_fields/instructor_script.md): Minute-by-minute teaching delivery script and blackboard plan.
3. [`mathematical_foundations.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_04_deep_learning_for_asr/lecture_11_temporal_convolutions_and_receptive_fields/mathematical_foundations.md): Complete mathematical derivations, proofs, and tensor transformations.
4. [`code_walkthrough.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_04_deep_learning_for_asr/lecture_11_temporal_convolutions_and_receptive_fields/code_walkthrough.md): Line-by-line inspection of production code from `src/`.
5. [`practical_lab.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_04_deep_learning_for_asr/lecture_11_temporal_convolutions_and_receptive_fields/practical_lab.md): Runnable laboratory exercise with verification scripts.
6. [`exercises.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_04_deep_learning_for_asr/lecture_11_temporal_convolutions_and_receptive_fields/exercises.md): Analytical, implementation, and forensic homework problems.
7. [`solutions.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_04_deep_learning_for_asr/lecture_11_temporal_convolutions_and_receptive_fields/solutions.md): Fully worked mathematical solutions and reference implementations.
8. [`quiz.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_04_deep_learning_for_asr/lecture_11_temporal_convolutions_and_receptive_fields/quiz.md): 5-question comprehension assessment with explanations.
9. [`research_connections.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_04_deep_learning_for_asr/lecture_11_temporal_convolutions_and_receptive_fields/research_connections.md): Direct mapping to repository stages, checkpoints, and historical bugs.
10. [`references.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_04_deep_learning_for_asr/lecture_11_temporal_convolutions_and_receptive_fields/references.md): Academic bibliography and reading list.
11. [`README.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_04_deep_learning_for_asr/lecture_11_temporal_convolutions_and_receptive_fields/README.md): This navigational overview.
