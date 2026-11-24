# Lecture 04: Resampling Physics & The 22.05 kHz Defect
## Polyphase Sinc Resampling, Anti-Aliasing Filters, and the Historical Bolbosh Resampling Trap

### Overview
This lecture is part of **Module 02**. It addresses both first-principles speech recognition science and the authentic empirical research findings of the EduCasheer Kashmiri ASR project.

### Learning Objectives
By the conclusion of this lecture, students will be able to:
- Derive the mathematics of digital sample rate conversion (up-sampling, filtering, down-sampling).
- Contrast naive linear/polynomial interpolation against band-limited sinc interpolation.
- Dissect the empirical 22.05 kHz resampling bug discovered in the Bolbosh Kashmiri corpus during the Stage 25 audit.
- Implement a certified band-limited polyphase resampler using PyTorch and `torchaudio`.

### Document Directory
Every lecture folder contains the 11 standard educational documents:
1. [`lecture_notes.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_02_speech_and_audio/lecture_04_resampling_and_22khz_defect/lecture_notes.md): Comprehensive textbook chapter.
2. [`instructor_script.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_02_speech_and_audio/lecture_04_resampling_and_22khz_defect/instructor_script.md): Minute-by-minute teaching delivery script and blackboard plan.
3. [`mathematical_foundations.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_02_speech_and_audio/lecture_04_resampling_and_22khz_defect/mathematical_foundations.md): Complete mathematical derivations, proofs, and tensor transformations.
4. [`code_walkthrough.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_02_speech_and_audio/lecture_04_resampling_and_22khz_defect/code_walkthrough.md): Line-by-line inspection of production code from `src/`.
5. [`practical_lab.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_02_speech_and_audio/lecture_04_resampling_and_22khz_defect/practical_lab.md): Runnable laboratory exercise with verification scripts.
6. [`exercises.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_02_speech_and_audio/lecture_04_resampling_and_22khz_defect/exercises.md): Analytical, implementation, and forensic homework problems.
7. [`solutions.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_02_speech_and_audio/lecture_04_resampling_and_22khz_defect/solutions.md): Fully worked mathematical solutions and reference implementations.
8. [`quiz.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_02_speech_and_audio/lecture_04_resampling_and_22khz_defect/quiz.md): 5-question comprehension assessment with explanations.
9. [`research_connections.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_02_speech_and_audio/lecture_04_resampling_and_22khz_defect/research_connections.md): Direct mapping to repository stages, checkpoints, and historical bugs.
10. [`references.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_02_speech_and_audio/lecture_04_resampling_and_22khz_defect/references.md): Academic bibliography and reading list.
11. [`README.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_02_speech_and_audio/lecture_04_resampling_and_22khz_defect/README.md): This navigational overview.
