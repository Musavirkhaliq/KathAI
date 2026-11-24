# Lecture 32: Code-Switching & Dialectal Adaptation
## Kashmiri-Urdu-English Intra-Sentential Switching, Script Mixing, and Acoustic Adapter Fine-Tuning

### Overview
This lecture is part of **Module 10**. It addresses both first-principles speech recognition science and the authentic empirical research findings of the EduCasheer Kashmiri ASR project.

### Learning Objectives
By the conclusion of this lecture, students will be able to:
- Formalize intra-sentential and inter-sentential code-switching in conversational Kashmiri speech.
- Analyze the orthographic and phonetic dilemmas of Kashmiri-Urdu-English code-switching.
- Deconstruct Parameter-Efficient Fine-Tuning (LoRA / Bottleneck Adapters) for dialectal adaptation.
- Implement a code-switching language detection and normalization pre-filter.

### Document Directory
Every lecture folder contains the 11 standard educational documents:
1. [`lecture_notes.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_10_advanced_asr_research/lecture_32_code_switching_and_dialectal_adaptation/lecture_notes.md): Comprehensive textbook chapter.
2. [`instructor_script.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_10_advanced_asr_research/lecture_32_code_switching_and_dialectal_adaptation/instructor_script.md): Minute-by-minute teaching delivery script and blackboard plan.
3. [`mathematical_foundations.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_10_advanced_asr_research/lecture_32_code_switching_and_dialectal_adaptation/mathematical_foundations.md): Complete mathematical derivations, proofs, and tensor transformations.
4. [`code_walkthrough.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_10_advanced_asr_research/lecture_32_code_switching_and_dialectal_adaptation/code_walkthrough.md): Line-by-line inspection of production code from `src/`.
5. [`practical_lab.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_10_advanced_asr_research/lecture_32_code_switching_and_dialectal_adaptation/practical_lab.md): Runnable laboratory exercise with verification scripts.
6. [`exercises.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_10_advanced_asr_research/lecture_32_code_switching_and_dialectal_adaptation/exercises.md): Analytical, implementation, and forensic homework problems.
7. [`solutions.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_10_advanced_asr_research/lecture_32_code_switching_and_dialectal_adaptation/solutions.md): Fully worked mathematical solutions and reference implementations.
8. [`quiz.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_10_advanced_asr_research/lecture_32_code_switching_and_dialectal_adaptation/quiz.md): 5-question comprehension assessment with explanations.
9. [`research_connections.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_10_advanced_asr_research/lecture_32_code_switching_and_dialectal_adaptation/research_connections.md): Direct mapping to repository stages, checkpoints, and historical bugs.
10. [`references.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_10_advanced_asr_research/lecture_32_code_switching_and_dialectal_adaptation/references.md): Academic bibliography and reading list.
11. [`README.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_10_advanced_asr_research/lecture_32_code_switching_and_dialectal_adaptation/README.md): This navigational overview.
