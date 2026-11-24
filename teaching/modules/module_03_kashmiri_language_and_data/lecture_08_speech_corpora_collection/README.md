# Lecture 08: Speech Corpora Collection & Curation
## Field Recordings, Web Scraping, SNR Thresholding, Audio Validation, and Silence Trimming

### Overview
This lecture is part of **Module 03**. It addresses both first-principles speech recognition science and the authentic empirical research findings of the EduCasheer Kashmiri ASR project.

### Learning Objectives
By the conclusion of this lecture, students will be able to:
- Design ethical and scalable speech data collection pipelines for low-resource linguistic communities.
- Formulate signal processing filters for automatic speech quality curation (energy SNR, clipping detection, silence trimming).
- Understand the acoustic trade-offs between clean studio speech (IndicVoices) and noisy in-the-wild speech (WAAZ/Bolbosh).
- Implement an automated audio curation and validation script in PyTorch.

### Document Directory
Every lecture folder contains the 11 standard educational documents:
1. [`lecture_notes.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_03_kashmiri_language_and_data/lecture_08_speech_corpora_collection/lecture_notes.md): Comprehensive textbook chapter.
2. [`instructor_script.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_03_kashmiri_language_and_data/lecture_08_speech_corpora_collection/instructor_script.md): Minute-by-minute teaching delivery script and blackboard plan.
3. [`mathematical_foundations.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_03_kashmiri_language_and_data/lecture_08_speech_corpora_collection/mathematical_foundations.md): Complete mathematical derivations, proofs, and tensor transformations.
4. [`code_walkthrough.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_03_kashmiri_language_and_data/lecture_08_speech_corpora_collection/code_walkthrough.md): Line-by-line inspection of production code from `src/`.
5. [`practical_lab.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_03_kashmiri_language_and_data/lecture_08_speech_corpora_collection/practical_lab.md): Runnable laboratory exercise with verification scripts.
6. [`exercises.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_03_kashmiri_language_and_data/lecture_08_speech_corpora_collection/exercises.md): Analytical, implementation, and forensic homework problems.
7. [`solutions.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_03_kashmiri_language_and_data/lecture_08_speech_corpora_collection/solutions.md): Fully worked mathematical solutions and reference implementations.
8. [`quiz.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_03_kashmiri_language_and_data/lecture_08_speech_corpora_collection/quiz.md): 5-question comprehension assessment with explanations.
9. [`research_connections.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_03_kashmiri_language_and_data/lecture_08_speech_corpora_collection/research_connections.md): Direct mapping to repository stages, checkpoints, and historical bugs.
10. [`references.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_03_kashmiri_language_and_data/lecture_08_speech_corpora_collection/references.md): Academic bibliography and reading list.
11. [`README.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_03_kashmiri_language_and_data/lecture_08_speech_corpora_collection/README.md): This navigational overview.
