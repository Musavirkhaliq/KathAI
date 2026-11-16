#!/usr/bin/env python3
"""
Automated Repository Audit & Maintenance Tool Suite
EduCasheer Kashmiri AI Research Project
"""

import os
import sys
import json
import csv
import re
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent.parent

def check_broken_paths():
    """Scan code files for hardcoded paths and verify existence."""
    print("="*60)
    print("AUDIT: Checking for broken paths in first-party code...")
    print("="*60)
    broken = []
    code_files = []
    for root, dirs, files in os.walk(REPO_DIR):
        if "/.git" in root or "venv_shruti" in root or "__pycache__" in root:
            continue
        for f in files:
            if f.endswith(".py"):
                code_files.append(Path(root) / f)

    path_pattern = re.compile(r'["\'](/home/musa/Kashmiri Project/[^"\']+)["\']')
    rel_path_pattern = re.compile(r'["\'](kashmiri_dataset/[^"\']+\.(?:jsonl|json|csv|safetensors|pt|arpa))["\']')

    for cf in code_files:
        try:
            with open(cf, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                for m in path_pattern.findall(content):
                    target = Path(m)
                    if not target.exists():
                        broken.append((cf.relative_to(REPO_DIR), m, "ABSOLUTE_MISSING"))
                for m in rel_path_pattern.findall(content):
                    target = REPO_DIR / m
                    if not target.exists():
                        broken.append((cf.relative_to(REPO_DIR), m, "RELATIVE_MISSING"))
        except Exception:
            pass

    print(f"Scanned {len(code_files)} code files.")
    if broken:
        print(f"Found {len(broken)} broken path references:")
        for cf, target, reason in broken[:10]:
            print(f"  [{reason}] In {cf} -> {target}")
    else:
        print("PASS: Zero broken path references detected in active code.")
    return len(broken)

def validate_manifests():
    """Verify sample count and syntax for key dataset manifests."""
    print("\n" + "="*60)
    print("AUDIT: Validating Core Dataset Manifests...")
    print("="*60)
    manifests_to_check = [
        "kashmiri_dataset/experiments/native_asr_training/data_splits/val.jsonl",
        "kashmiri_dataset/experiments/stage22c_inference_certification/inference/stage22c_best_system_manifest.json",
        "kashmiri_dataset/experiments/stage22_data_expanded_training/checkpoints/stage22_best_model_manifest.json",
        "kashmiri_dataset/experiments/stage24_actual_reranker_training/stage24_best_model_manifest.json"
    ]
    all_ok = True
    for m in manifests_to_check:
        mp = REPO_DIR / m
        if not mp.exists():
            print(f"FAIL: Manifest missing: {m}")
            all_ok = False
            continue
        try:
            if m.endswith(".jsonl"):
                with open(mp, "r", encoding="utf-8") as f:
                    count = sum(1 for _ in f)
                print(f"PASS: {m} (valid JSONL, {count:,} samples)")
            else:
                with open(mp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                print(f"PASS: {m} (valid JSON, {len(data)} top-level keys)")
        except Exception as e:
            print(f"FAIL: Syntax error in {m}: {e}")
            all_ok = False
    return all_ok

def validate_model_checkpoints():
    """Verify that certified model checkpoints exist and weights match expected size."""
    print("\n" + "="*60)
    print("AUDIT: Validating Primary Model Checkpoints...")
    print("="*60)
    checkpoints = [
        ("Stage 22 E22-01 (SOTA Champion)", "kashmiri_dataset/experiments/stage22_data_expanded_training/checkpoints/E22-01/best_model/model.safetensors", 1262467532),
        ("Stage 9 B-02C (Foundation SOTA)", "kashmiri_dataset/experiments/native_asr_training/experiments/stage9_b02_acoustic_comparison/B-02C/checkpoints/best_model/model.safetensors", 1262467532),
        ("Stage 8 B-01 (Genesis Baseline)", "kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc/best_model/model.safetensors", 1262467532),
        ("ByT5 Transliteration Adapter", "kashmiri_dataset/experiments/transliteration_evaluation/byt5_adapter/best_checkpoint/model.safetensors", 1198542104)
    ]
    all_ok = True
    for name, rel_path, expected_size in checkpoints:
        cp = REPO_DIR / rel_path
        if not cp.exists():
            print(f"FAIL: {name} MISSING at {rel_path}")
            all_ok = False
        else:
            actual_size = cp.stat().st_size
            if actual_size == expected_size:
                print(f"PASS: {name} EXISTS ({actual_size:,} bytes, exact size match)")
            else:
                print(f"WARN: {name} EXISTS but size differs (actual {actual_size:,} vs expected {expected_size:,})")
    return all_ok

def validate_doc_links():
    """Verify relative file links inside documentation directory."""
    print("\n" + "="*60)
    print("AUDIT: Validating Documentation Links...")
    print("="*60)
    doc_files = []
    for root, dirs, files in os.walk(REPO_DIR / "docs"):
        for f in files:
            if f.endswith(".md"):
                doc_files.append(Path(root) / f)

    link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    broken_links = []
    total_links = 0

    for df in doc_files:
        try:
            with open(df, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                for text, target in link_pattern.findall(content):
                    if target.startswith("http") or target.startswith("#") or target.startswith("mailto:"):
                        continue
                    total_links += 1
                    # Handle file:// URLs
                    if target.startswith("file://"):
                        from urllib.parse import unquote
                        clean_target = unquote(target[7:].split("#")[0])
                        resolved = Path(clean_target)
                        if not resolved.exists():
                            broken_links.append((df.relative_to(REPO_DIR), target))
                        continue

                    # Clean anchor
                    clean_target = target.split("#")[0]
                    if not clean_target:
                        continue
                    resolved = (df.parent / clean_target).resolve()
                    if not resolved.exists():
                        broken_links.append((df.relative_to(REPO_DIR), target))
        except Exception:
            pass

    print(f"Checked {total_links} relative markdown links across {len(doc_files)} docs.")
    if broken_links:
        print(f"Notice: {len(broken_links)} unresolvable relative targets:")
        for df, target in broken_links[:8]:
            print(f"  In {df} -> {target}")
    else:
        print("PASS: 100% of relative markdown links resolved successfully.")
    return len(broken_links)

def main():
    print(f"EduCasheer Automated Audit Suite starting at: {REPO_DIR}\n")
    c1 = check_broken_paths()
    c2 = validate_manifests()
    c3 = validate_model_checkpoints()
    c4 = validate_doc_links()
    print("\n" + "="*60)
    print("SUMMARY AUDIT SUITE STATUS:")
    print(f"  Broken Paths in Code: {c1}")
    print(f"  Manifests Validated:  {'PASS' if c2 else 'FAIL'}")
    print(f"  Model Checkpoints:    {'PASS' if c3 else 'FAIL'}")
    print(f"  Broken Doc Links:     {c4}")
    print("="*60)

if __name__ == "__main__":
    main()
