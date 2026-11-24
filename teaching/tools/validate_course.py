#!/usr/bin/env python3
"""
teaching/tools/validate_course.py
Comprehensive automated test suite and integrity validator for the EduCasheer Kashmiri ASR Course.
Validates:
1. All 12 Modules exist and have README.md.
2. All 35 Lectures exist in their respective modules.
3. Every lecture contains all 11 required standard documents and none are empty.
4. All 15 Research Case Studies exist and are nonempty.
5. All 6 Graded Assignments exist with rubrics and starter code.
6. Midterm and Final examinations exist with questions and solution keys.
7. Capstone project specification and rubrics exist.
8. All 12 Module presentation outlines exist in instructor_resources/.
9. Teaching dataset manifests are valid and parseable JSON/JSONL.
10. Internal document links resolve cleanly to existing filesystem paths.
"""

import os
import sys
import json
import re
from pathlib import Path
from urllib.parse import unquote

TEACHING_DIR = Path("/home/musa/Kashmiri Project/teaching")

EXPECTED_MODULES = [
    ("module_01_introduction", 2),
    ("module_02_speech_and_audio", 3),
    ("module_03_kashmiri_language_and_data", 4),
    ("module_04_deep_learning_for_asr", 4),
    ("module_05_ctc_and_acoustic_models", 4),
    ("module_06_training_and_optimization", 3),
    ("module_07_decoding_and_language_models", 3),
    ("module_08_benchmarking_and_evaluation", 3),
    ("module_09_research_experiments", 4),
    ("module_10_advanced_asr_research", 2),
    ("module_11_production_systems", 2),
    ("module_12_future_research", 1),
]

REQUIRED_LECTURE_FILES = [
    "README.md",
    "lecture_notes.md",
    "instructor_script.md",
    "mathematical_foundations.md",
    "code_walkthrough.md",
    "practical_lab.md",
    "exercises.md",
    "solutions.md",
    "quiz.md",
    "research_connections.md",
    "references.md"
]

class CourseValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.total_checks = 0
        self.passed_checks = 0

    def record_check(self, success: bool, msg: str):
        self.total_checks += 1
        if success:
            self.passed_checks += 1
        else:
            self.errors.append(msg)

    def validate_core_guides(self):
        print("=== Checking Core Navigational Guides ===")
        core_files = [
            "README.md",
            "PREREQUISITES.md",
            "TEACHING_PROGRESS_TRACKER.md",
            "COURSE_MASTER_GUIDE.md",
            "COURSE_SYLLABUS.md",
            "LECTURE_INDEX.md",
            "RESEARCH_TO_LECTURE_MAP.md"
        ]
        for cf in core_files:
            path = TEACHING_DIR / cf
            exists = path.exists() and path.stat().st_size > 0
            self.record_check(exists, f"Missing or empty core guide: {cf}")

    def validate_modules_and_lectures(self):
        print("=== Checking 12 Modules and 35 Lectures (11 Files Each) ===")
        modules_dir = TEACHING_DIR / "modules"
        total_lectures_found = 0

        for mod_name, expected_lec_count in EXPECTED_MODULES:
            mod_path = modules_dir / mod_name
            # Check module README
            mod_readme = mod_path / "README.md"
            self.record_check(mod_readme.exists() and mod_readme.stat().st_size > 0, f"Module README missing: {mod_name}/README.md")

            # Check lecture subdirectories
            lec_dirs = [d for d in mod_path.iterdir() if d.is_dir() and d.name.startswith("lecture_")]
            self.record_check(len(lec_dirs) == expected_lec_count, f"Module {mod_name}: expected {expected_lec_count} lectures, found {len(lec_dirs)}")
            total_lectures_found += len(lec_dirs)

            for ldir in lec_dirs:
                for req_file in REQUIRED_LECTURE_FILES:
                    fpath = ldir / req_file
                    is_valid = fpath.exists() and fpath.stat().st_size > 0
                    self.record_check(is_valid, f"Lecture {ldir.name} missing or empty file: {req_file}")

        self.record_check(total_lectures_found == 35, f"Expected 35 total lectures, found {total_lectures_found}")

    def validate_case_studies(self):
        print("=== Checking 15 Research Case Studies ===")
        cs_dir = TEACHING_DIR / "research_case_studies"
        cs_folders = [d for d in cs_dir.iterdir() if d.is_dir()]
        self.record_check(len(cs_folders) == 15, f"Expected 15 case studies, found {len(cs_folders)}")
        for csd in cs_folders:
            rm = csd / "README.md"
            self.record_check(rm.exists() and rm.stat().st_size > 0, f"Case study README missing or empty: {csd.name}")

    def validate_assignments(self):
        print("=== Checking 6 Graded Assignments ===")
        as_dir = TEACHING_DIR / "assignments"
        as_folders = [d for d in as_dir.iterdir() if d.is_dir()]
        self.record_check(len(as_folders) == 6, f"Expected 6 assignments, found {len(as_folders)}")
        for ad in as_folders:
            rm = ad / "README.md"
            rub = ad / "rubric.md"
            self.record_check(rm.exists() and rm.stat().st_size > 0, f"Assignment README missing: {ad.name}")
            self.record_check(rub.exists() and rub.stat().st_size > 0, f"Assignment rubric missing: {ad.name}")
            self.record_check((ad / "starter_code").is_dir(), f"Assignment starter_code missing: {ad.name}")
            self.record_check((ad / "solution").is_dir(), f"Assignment solution missing: {ad.name}")

    def validate_assessments(self):
        print("=== Checking Midterm & Final Exams ===")
        ass_dir = TEACHING_DIR / "assessment"
        for exam in ["midterm_exam", "final_exam"]:
            edir = ass_dir / exam
            paper = edir / "exam_paper.md"
            sol = edir / "solution_key.md"
            self.record_check(paper.exists() and paper.stat().st_size > 0, f"Exam paper missing: {exam}/exam_paper.md")
            self.record_check(sol.exists() and sol.stat().st_size > 0, f"Exam solution key missing: {exam}/solution_key.md")

    def validate_capstone(self):
        print("=== Checking Capstone Project Specification ===")
        cap_dir = TEACHING_DIR / "capstone"
        for doc in ["README.md", "PROJECT_SPECIFICATION.md", "RUBRIC_AND_MILESTONES.md"]:
            path = cap_dir / doc
            self.record_check(path.exists() and path.stat().st_size > 0, f"Capstone document missing: {doc}")

    def validate_presentation_outlines(self):
        print("=== Checking Instructor Presentation Outlines ===")
        pres_dir = TEACHING_DIR / "instructor_resources" / "presentation_outlines"
        outlines = list(pres_dir.glob("*_presentation_outline.md"))
        self.record_check(len(outlines) == 12, f"Expected 12 presentation outlines, found {len(outlines)}")

    def validate_datasets(self):
        print("=== Checking Teaching Dataset Manifests ===")
        d_dir = TEACHING_DIR / "datasets_for_teaching"
        manifest_file = d_dir / "synthetic_audio_manifest.jsonl"
        splits_file = d_dir / "disjoint_speaker_toy_splits.json"
        
        # Check jsonl
        valid_jsonl = False
        if manifest_file.exists():
            with open(manifest_file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                valid_jsonl = len(lines) > 0 and all(json.loads(l) for l in lines)
        self.record_check(valid_jsonl, "Invalid or empty synthetic_audio_manifest.jsonl")

        # Check json
        valid_json = False
        if splits_file.exists():
            with open(splits_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                valid_json = data.get("certified_disjoint") is True
        self.record_check(valid_json, "Invalid or missing disjoint_speaker_toy_splits.json")

    def validate_links(self):
        print("=== Checking Markdown Document Links ===")
        broken_links = []
        link_regex = re.compile(r'\[([^\]]+)\]\((file://[^)]+)\)')

        for root, _, files in os.walk(TEACHING_DIR):
            for file in files:
                if file.endswith(".md"):
                    fpath = Path(root) / file
                    try:
                        text = fpath.read_text(encoding="utf-8")
                        for match in link_regex.finditer(text):
                            raw_uri = match.group(2)
                            # Remove file:// and unquote
                            clean_path = unquote(raw_uri.replace("file://", ""))
                            # Strip line anchor if any
                            clean_path = clean_path.split("#")[0]
                            target_path = Path(clean_path)
                            if not target_path.exists():
                                broken_links.append((str(fpath), raw_uri))
                    except Exception as e:
                        self.warnings.append(f"Error reading {fpath}: {e}")

        self.record_check(len(broken_links) == 0, f"Found {len(broken_links)} broken file:// links in teaching/")
        if broken_links:
            for source, target in broken_links[:10]:
                print(f"  [BROKEN LINK] In {source} -> {target}")

    def run_all(self):
        self.validate_core_guides()
        self.validate_modules_and_lectures()
        self.validate_case_studies()
        self.validate_assignments()
        self.validate_assessments()
        self.validate_capstone()
        self.validate_presentation_outlines()
        self.validate_datasets()
        self.validate_links()

        print("\n" + "=" * 60)
        print("COURSE INTEGRITY VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Total Checks Executed: {self.total_checks}")
        print(f"Passed Checks:         {self.passed_checks}")
        print(f"Failed Checks:         {len(self.errors)}")

        if self.errors:
            print("\nERRORS DETECTED:")
            for err in self.errors:
                print(f"  [FAIL] {err}")
            return False
        else:
            print("\n[PERFECT SUCCESS] All course components verified! The curriculum is complete and self-contained.")
            return True

if __name__ == "__main__":
    validator = CourseValidator()
    success = validator.run_all()
    sys.exit(0 if success else 1)
