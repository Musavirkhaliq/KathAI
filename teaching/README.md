# EduCasheer Kashmiri ASR: University & Research Teaching Course
========================================================================

Welcome to the official, authoritative teaching curriculum of the **EduCasheer Kashmiri Automatic Speech Recognition (ASR) Research Initiative**.

This repository transforms 27 stages of cutting-edge speech recognition research on the Kashmiri language (*Kōshur*, ISO 639-3: `kas`) into a **complete, 35-lecture, university-level graduate course**.

---

## The Dual-Narrative Educational Philosophy

This course is designed around two interconnected storylines:

1. **Story 1: How Modern Automatic Speech Recognition Works (First Principles)**
   - Rigorous mathematical formulations of acoustics, sampling, discrete Fourier transforms, temporal convolutions, self-attention, Connectionist Temporal Classification (CTC), beam search, and language model shallow fusion.
   - Theoretical concepts taught from scratch without hand-waving or black-box abstractions.

2. **Story 2: The Real Research Trajectory of Kashmiri ASR (Empirical Evidence)**
   - The authentic, uncensored story of building an end-to-end ASR system for an extreme low-resource language spoken by 7+ million people.
   - Covers 106,487 speech utterances (214.65 hours), 243,701 files, the genesis baseline B-01, foundation model B-02C, production champion E22-01, negative results (synthetic noise, encoder unfreezing, neural candidate rerankers), and forensic reproducibility audits (the 22.05 kHz resampling bug recovery and exact CTC frame slicing).

---

## Course Navigation & Hub Documents

- [`COURSE_MASTER_GUIDE.md`](COURSE_MASTER_GUIDE.md): Central comprehensive course textbook and architectural reference manual.
- [`COURSE_SYLLABUS.md`](COURSE_SYLLABUS.md): 14-week university semester syllabus with lecture schedules, grading rubrics, and reading lists.
- [`LECTURE_INDEX.md`](LECTURE_INDEX.md): Master table of all 35 lectures with links to lecture notes, math proofs, code walkthroughs, labs, and quizzes.
- [`PREREQUISITES.md`](PREREQUISITES.md): Detailed mathematical and software prerequisites roadmap.
- [`RESEARCH_TO_LECTURE_MAP.md`](RESEARCH_TO_LECTURE_MAP.md): Bidirectional cross-reference mapping every repository stage (1 to 27) to lectures.
- [`TEACHING_PROGRESS_TRACKER.md`](TEACHING_PROGRESS_TRACKER.md): Live completion tracker for lectures, case studies, assignments, and labs.

---

## 12 Curricular Modules

1. **Module 01: Introduction to Speech Recognition & The Kashmiri Frontier** (Lectures 01–02)
2. **Module 02: Speech Signal Processing & The Mathematics of Sound** (Lectures 03–05)
3. **Module 03: Kashmiri Linguistics, Orthography & Dataset Engineering** (Lectures 06–09)
4. **Module 04: Deep Learning Foundations for Speech Representation** (Lectures 10–13)
5. **Module 05: Connectionist Temporal Classification & Acoustic Modeling** (Lectures 14–17)
6. **Module 06: Acoustic Fine-Tuning, Scaling & Experimental Optimization** (Lectures 18–20)
7. **Module 07: Decoding, Search Spaces & Language Modeling** (Lectures 21–23)
8. **Module 08: Benchmarking, Error Analysis & Statistical Evaluation** (Lectures 24–26)
9. **Module 09: Complete Kashmiri ASR Research History: Stages 1 to 27** (Lectures 27–30)
10. **Module 10: Advanced Research Directions & Frontier Techniques** (Lectures 31–32)
11. **Module 11: Production Systems & ML Research Engineering** (Lectures 33–34)
12. **Module 12: Capstone: Building Your Own End-to-End Kashmiri ASR System** (Lecture 35)

---

## Directory Architecture

```text
teaching/
├── README.md                           # This document
├── COURSE_MASTER_GUIDE.md              # Master textbook & guide
├── COURSE_SYLLABUS.md                  # 14-week university syllabus
├── LECTURE_INDEX.md                    # 35-lecture index
├── PREREQUISITES.md                    # Math & coding prerequisites
├── RESEARCH_TO_LECTURE_MAP.md          # Research-to-lecture mapping
├── TEACHING_PROGRESS_TRACKER.md        # Live completion tracker
├── modules/                            # 12 Module overview directories
├── lectures/                           # 35 Self-contained lecture packages (11 files each)
├── labs/                               # Hands-on Python & PyTorch lab assignments
├── assignments/                        # 6 Substantial graded problem sets with rubrics
├── assessment/                         # Midterm and Final examinations
├── quizzes/                            # Standalone lecture quizzes & concept checks
├── research_case_studies/              # 15 Detailed forensic research case studies
├── datasets_for_teaching/              # Lightweight teaching audio manifests & samples
├── instructor_resources/               # Slide presentation outlines & teaching notes
├── student_resources/                  # Study guides, cheat-sheets, and reading lists
├── capstone/                           # Complete end-to-end capstone specification
└── tools/                              # Automated course validation & testing suite
```

---

## Target Audience

1. **University Students & Instructors**: Ideal for an advanced undergraduate or graduate course in Speech Processing, Natural Language Processing, or Applied Deep Learning.
2. **AI & Speech Researchers**: A blueprint for building competitive acoustic models and tokenizers for extreme low-resource languages.
3. **ML Engineers**: Practical patterns for robust audio pipeline construction, exact CTC slicing, calibrated beam search decoding, and low-latency production deployment.
