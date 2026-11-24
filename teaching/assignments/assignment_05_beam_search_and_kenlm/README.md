# Assignment 5: CTC Prefix Beam Search & KenLM Shallow Fusion
**Curriculum Module**: `Module 07`  
**Grade Weight**: 5.0% of Course Grade  
**Due Date**: End of Week corresponding to `Module 07`

---

## 1. Assignment Overview
Build a CTC Prefix Beam Search decoder tracking $P_b$ and $P_{nb}$, calibrate acoustic temperature $T=1.08$, and integrate 5-gram KenLM rescoring.

---

## 2. Learning Objectives
- Translate mathematical speech recognition equations into clean, vector-accelerated PyTorch code.
- Verify numerical stability and boundary conditions on real Kashmiri speech tensors.
- Adhere to certified production engineering standards matching `src/`.

---

## 3. Directory Structure
- `README.md`: This assignment specification.
- `rubric.md`: Detailed grading rubric (100 points total).
- `starter_code/`: Python template files with `TODO` markers and unit test assertions.
- `solution/`: Reference implementation and worked solutions (restricted to instructors).

---

## 4. Submission Guidelines
Submit your completed Python modules passing all local unit tests via `pytest starter_code/`.
