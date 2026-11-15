# Scientific Integrity, Authorship & Submission Checklist

This document provides the mandatory compliance checklist governing ethical standards, authorship confirmation, data usage rights, and submission protocols.

---

## 1. Authorship and Attribution Protocol

In accordance with COPE (Committee on Publication Ethics) guidelines:
- [ ] **Author Names & Order**: Requires human confirmation by the project leadership.
- [ ] **Affiliations**: Formal institutional affiliations to be verified by human authors.
- [ ] **Contribution Tracking**: CRediT (Contributor Roles Taxonomy) statements to be assigned for:
  - Conceptualization
  - Data Curation
  - Formal Analysis & Forensic Auditing
  - Investigation & GPU Training
  - Methodology & Decoding Software
  - Writing - Original Draft & Review
- [ ] **Ethics & Consent Confirmation**: Verification that community recordings comply with non-commercial consent agreements.
- [ ] **Conflict of Interest**: Explicit declaration of zero financial conflicts of interest.

---

## 2. Pre-Submission Verification Gate

- [x] **Zero Fabricated Citations**: 100% of references verified in `literature/citation_verification_log.md`.
- [x] **Zero Altered Historical Results**: All 21 stages preserved chronologically without rewriting experimental history.
- [x] **Negative Results Disclosed**: Complete failure analyses provided for noise augmentation, layer unfreezing, and pseudo-labeling.
- [x] **Code Provenance Traceable**: Script paths, commit hashes, and function implementations documented.
- [x] **Reproduction Script Verified**: Execution of `./reproduce_paper_results.sh --mode all` terminates with exit code 0.
- [x] **Dual CER Definitions Clarified**: CER with spaces (9.01%) and without spaces (9.97%) explicitly distinguished.
- [x] **Diagnostic Limitations Disclosed**: Repeated evaluation on `dev_assessment.jsonl` acknowledged.
