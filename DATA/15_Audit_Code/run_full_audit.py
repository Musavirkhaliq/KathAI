#!/usr/bin/env python3
"""
DATA Master Audit Runner
Executes the complete audit pipeline across all data modalities and refreshes documentation.
EduCasheer Kashmiri AI Research Project
"""

import os
import sys
import subprocess
import time

def run_step(step_name, script_path):
    print(f"\n>>> [RUNNING STEP] {step_name} ({script_path})")
    start = time.time()
    res = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    elapsed = time.time() - start
    if res.returncode == 0:
        print(f">>> [SUCCESS] {step_name} completed in {elapsed:.2f}s")
        if res.stdout:
            lines = res.stdout.strip().split("\n")
            print("    " + "\n    ".join(lines[-4:]))
    else:
        print(f">>> [FAILED] {step_name} exited with code {res.returncode}")
        print(res.stderr)
        sys.exit(1)

def main():
    print("======================================================================")
    print("STARTING FULL REPRODUCIBLE DATA AUDIT SUITE")
    print("======================================================================")
    total_start = time.time()

    run_step("1. Physical File Discovery", "DATA/15_Audit_Code/01_discover_files.py")
    run_step("2. Physical & Manifested Speech Inspection", "DATA/15_Audit_Code/02_inspect_audio.py")
    run_step("3. Text Corpus & Script Inspection", "DATA/15_Audit_Code/03_inspect_text.py")
    run_step("4. Parallel Translation & Lexicon Audit", "DATA/15_Audit_Code/04_inspect_translation_and_lexicons.py")
    run_step("5. WAAZ Platform In-Depth Audit", "DATA/15_Audit_Code/05_audit_waaz.py")
    run_step("6. Training Data History Audit", "DATA/15_Audit_Code/06_audit_training_usage.py")
    run_step("7. Benchmark Forensic Audit", "DATA/15_Audit_Code/07_audit_benchmarks.py")
    run_step("8. Unified Data Readiness Matrix", "DATA/15_Audit_Code/08_audit_readiness_and_tiers.py")
    run_step("9. Scientific Visualizations Generator", "DATA/15_Audit_Code/09_generate_visualizations.py")
    run_step("10. Master Documentation Generator", "DATA/15_Audit_Code/10_generate_documentation.py")

    total_elapsed = time.time() - total_start
    print("\n======================================================================")
    print(f"FULL DATA AUDIT COMPLETED IN {total_elapsed:.2f} SECONDS!")
    print("Canonical DATA directory is fully refreshed and up-to-date.")
    print("======================================================================")

if __name__ == "__main__":
    main()
