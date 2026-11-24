# Lecture 26: Statistical Significance & Bootstrap Tests
## Non-Parametric Bootstrap Confidence Intervals, Paired Wilcoxon Signed-Rank Test, and Scientific Rigor

### Overview
This lecture is part of **Module 08**. It addresses both first-principles speech recognition science and the authentic empirical research findings of the EduCasheer Kashmiri ASR project.

### Learning Objectives
By the conclusion of this lecture, students will be able to:
- Formalize statistical hypothesis testing for speech recognition benchmarks.
- Implement Non-Parametric Bootstrap resampling to compute $95\%$ confidence intervals for WER.
- Formulate the Paired Wilcoxon Signed-Rank Test to determine if an acoustic improvement is statistically significant.
- Analyze the risks of $p$-hacking and ungrounded claims in low-resource speech research.

### Document Directory
Every lecture folder contains the 11 standard educational documents:
1. [`lecture_notes.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_08_benchmarking_and_evaluation/lecture_26_statistical_significance_bootstrap/lecture_notes.md): Comprehensive textbook chapter.
2. [`instructor_script.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_08_benchmarking_and_evaluation/lecture_26_statistical_significance_bootstrap/instructor_script.md): Minute-by-minute teaching delivery script and blackboard plan.
3. [`mathematical_foundations.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_08_benchmarking_and_evaluation/lecture_26_statistical_significance_bootstrap/mathematical_foundations.md): Complete mathematical derivations, proofs, and tensor transformations.
4. [`code_walkthrough.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_08_benchmarking_and_evaluation/lecture_26_statistical_significance_bootstrap/code_walkthrough.md): Line-by-line inspection of production code from `src/`.
5. [`practical_lab.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_08_benchmarking_and_evaluation/lecture_26_statistical_significance_bootstrap/practical_lab.md): Runnable laboratory exercise with verification scripts.
6. [`exercises.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_08_benchmarking_and_evaluation/lecture_26_statistical_significance_bootstrap/exercises.md): Analytical, implementation, and forensic homework problems.
7. [`solutions.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_08_benchmarking_and_evaluation/lecture_26_statistical_significance_bootstrap/solutions.md): Fully worked mathematical solutions and reference implementations.
8. [`quiz.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_08_benchmarking_and_evaluation/lecture_26_statistical_significance_bootstrap/quiz.md): 5-question comprehension assessment with explanations.
9. [`research_connections.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_08_benchmarking_and_evaluation/lecture_26_statistical_significance_bootstrap/research_connections.md): Direct mapping to repository stages, checkpoints, and historical bugs.
10. [`references.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_08_benchmarking_and_evaluation/lecture_26_statistical_significance_bootstrap/references.md): Academic bibliography and reading list.
11. [`README.md`](file:///home/musa/Kashmiri Project/teaching/modules/module_08_benchmarking_and_evaluation/lecture_26_statistical_significance_bootstrap/README.md): This navigational overview.
