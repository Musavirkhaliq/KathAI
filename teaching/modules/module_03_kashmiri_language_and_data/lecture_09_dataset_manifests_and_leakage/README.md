# Lecture 09: Dataset Manifests & Split Contamination
## JSONL/TSV Manifest Schemas, Graph-Based Disjoint Partitioning, and The Stage 25 Contamination Audit

### Overview
This lecture is part of **Module 03**. It addresses both first-principles speech recognition science and the authentic empirical research findings of the EduCasheer Kashmiri ASR project.

### Learning Objectives
By the conclusion of this lecture, students will be able to:
- Structure standardized, immutable dataset manifests with cryptographic SHA-256 hashes.
- Formulate the mathematical graph-partitioning problem for zero-speaker and zero-sentence data splitting.
- Deconstruct the Stage 25 Contamination Audit that uncovered speaker leakage between train and test splits.
- Implement an automated dataset split validator in Python that detects audio and text contamination.

### Document Directory
Every lecture folder contains the 11 standard educational documents:
1. [`lecture_notes.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_03_kashmiri_language_and_data/lecture_09_dataset_manifests_and_leakage/lecture_notes.md): Comprehensive textbook chapter.
2. [`instructor_script.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_03_kashmiri_language_and_data/lecture_09_dataset_manifests_and_leakage/instructor_script.md): Minute-by-minute teaching delivery script and blackboard plan.
3. [`mathematical_foundations.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_03_kashmiri_language_and_data/lecture_09_dataset_manifests_and_leakage/mathematical_foundations.md): Complete mathematical derivations, proofs, and tensor transformations.
4. [`code_walkthrough.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_03_kashmiri_language_and_data/lecture_09_dataset_manifests_and_leakage/code_walkthrough.md): Line-by-line inspection of production code from `src/`.
5. [`practical_lab.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_03_kashmiri_language_and_data/lecture_09_dataset_manifests_and_leakage/practical_lab.md): Runnable laboratory exercise with verification scripts.
6. [`exercises.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_03_kashmiri_language_and_data/lecture_09_dataset_manifests_and_leakage/exercises.md): Analytical, implementation, and forensic homework problems.
7. [`solutions.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_03_kashmiri_language_and_data/lecture_09_dataset_manifests_and_leakage/solutions.md): Fully worked mathematical solutions and reference implementations.
8. [`quiz.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_03_kashmiri_language_and_data/lecture_09_dataset_manifests_and_leakage/quiz.md): 5-question comprehension assessment with explanations.
9. [`research_connections.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_03_kashmiri_language_and_data/lecture_09_dataset_manifests_and_leakage/research_connections.md): Direct mapping to repository stages, checkpoints, and historical bugs.
10. [`references.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_03_kashmiri_language_and_data/lecture_09_dataset_manifests_and_leakage/references.md): Academic bibliography and reading list.
11. [`README.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_03_kashmiri_language_and_data/lecture_09_dataset_manifests_and_leakage/README.md): This navigational overview.
