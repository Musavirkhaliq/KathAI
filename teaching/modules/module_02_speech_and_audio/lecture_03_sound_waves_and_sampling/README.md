# Lecture 03: Sound Waves, Sampling & Quantization
## Continuous Wave Mechanics, The Nyquist-Shannon Theorem, and PCM Quantization Dynamics

### Overview
This lecture is part of **Module 02**. It addresses both first-principles speech recognition science and the authentic empirical research findings of the EduCasheer Kashmiri ASR project.

### Learning Objectives
By the conclusion of this lecture, students will be able to:
- Derive the acoustic wave equation and understand acoustic pressure oscillations.
- Formulate the Nyquist-Shannon Sampling Theorem and the mathematics of aliasing.
- Analyze uniform and non-uniform Pulse Code Modulation (PCM) quantization noise.
- Implement high-fidelity audio tensor loading, dynamic range scaling, and waveform validation in PyTorch.

### Document Directory
Every lecture folder contains the 11 standard educational documents:
1. [`lecture_notes.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_02_speech_and_audio/lecture_03_sound_waves_and_sampling/lecture_notes.md): Comprehensive textbook chapter.
2. [`instructor_script.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_02_speech_and_audio/lecture_03_sound_waves_and_sampling/instructor_script.md): Minute-by-minute teaching delivery script and blackboard plan.
3. [`mathematical_foundations.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_02_speech_and_audio/lecture_03_sound_waves_and_sampling/mathematical_foundations.md): Complete mathematical derivations, proofs, and tensor transformations.
4. [`code_walkthrough.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_02_speech_and_audio/lecture_03_sound_waves_and_sampling/code_walkthrough.md): Line-by-line inspection of production code from `src/`.
5. [`practical_lab.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_02_speech_and_audio/lecture_03_sound_waves_and_sampling/practical_lab.md): Runnable laboratory exercise with verification scripts.
6. [`exercises.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_02_speech_and_audio/lecture_03_sound_waves_and_sampling/exercises.md): Analytical, implementation, and forensic homework problems.
7. [`solutions.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_02_speech_and_audio/lecture_03_sound_waves_and_sampling/solutions.md): Fully worked mathematical solutions and reference implementations.
8. [`quiz.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_02_speech_and_audio/lecture_03_sound_waves_and_sampling/quiz.md): 5-question comprehension assessment with explanations.
9. [`research_connections.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_02_speech_and_audio/lecture_03_sound_waves_and_sampling/research_connections.md): Direct mapping to repository stages, checkpoints, and historical bugs.
10. [`references.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_02_speech_and_audio/lecture_03_sound_waves_and_sampling/references.md): Academic bibliography and reading list.
11. [`README.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_02_speech_and_audio/lecture_03_sound_waves_and_sampling/README.md): This navigational overview.
