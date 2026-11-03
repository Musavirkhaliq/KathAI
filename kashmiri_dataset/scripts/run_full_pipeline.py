#!/usr/bin/env python3
"""
Master Execution Pipeline for Kashmiri Speech & Multimodal Dataset
==================================================================
Sequentially executes:
  1. audit_dataset.py
  2. build_manifest.py (Audio standardization, conversion, VAD & SNR estimation)
  3. create_splits.py (Speaker-isolated Train/Val/Test + evaluation benchmark slices)
  4. dataset_statistics.py (Full quantification report generation)
  5. discover_external_datasets.py (Cataloging and registry sync)
"""

import sys
import subprocess
import time

SCRIPTS = [
    "audit_dataset.py",
    "build_manifest.py",
    "create_splits.py",
    "dataset_statistics.py",
    "discover_external_datasets.py"
]

def run():
    print("=" * 60)
    print("STARTING KASHMIRI RESEARCH DATASET FULL PIPELINE")
    print("=" * 60)
    start_total = time.time()

    for script in SCRIPTS:
        print(f"\n>>> Running {script}...")
        t0 = time.time()
        res = subprocess.run([sys.executable, f"/home/musa/Kashmiri Project/kashmiri_dataset/scripts/{script}"])
        if res.returncode != 0:
            print(f"ERROR: {script} failed with exit code {res.returncode}")
            sys.exit(res.returncode)
        print(f"Finished {script} in {time.time() - t0:.2f}s.")

    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {time.time() - start_total:.2f}s")
    print("=" * 60)

if __name__ == '__main__':
    run()
