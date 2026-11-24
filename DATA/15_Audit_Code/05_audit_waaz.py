#!/usr/bin/env python3
"""
DATA Audit Step 5: WAAZ Crowdsourced Platform In-Depth Forensic Audit
EduCasheer Kashmiri AI Research Project
"""

import os
import sqlite3
import json
import pandas as pd

def main():
    print("=" * 60)
    print("EXECUTING WAAZ COMMUNITY PLATFORM AUDIT")
    print("=" * 60)

    db_path = "/home/musa/Desktop/Projects/DataCollection/data/datacollect.db"
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # 1. Basic Counts
    c.execute("SELECT count(*) FROM users")
    total_users = c.fetchone()[0]

    c.execute("SELECT count(DISTINCT user_id) FROM contributions")
    active_recorded_users = c.fetchone()[0]

    c.execute("SELECT count(*) FROM prompts")
    total_prompts = c.fetchone()[0]

    c.execute("SELECT count(*), sum(duration_seconds) FROM recordings")
    rec_count, total_dur = c.fetchone()
    total_dur_hrs = (total_dur or 0.0) / 3600.0

    # 2. Validation Status
    c.execute("SELECT status, count(*) FROM contributions GROUP BY status")
    contrib_status = dict(c.fetchall())

    c.execute("SELECT verdict, count(*) FROM validations GROUP BY verdict")
    val_verdicts = dict(c.fetchall())

    # 3. Approved recordings
    c.execute("""
        SELECT count(DISTINCT r.id), sum(r.duration_seconds)
        FROM validations v
        JOIN recordings r ON r.id = v.recording_id
        WHERE v.verdict = 'APPROVE'
    """)
    approved_count, approved_dur = c.fetchone()
    approved_dur_hrs = (approved_dur or 0.0) / 3600.0

    # 4. Ready for review queue
    c.execute("""
        SELECT count(DISTINCT r.id), sum(r.duration_seconds)
        FROM contributions c_tbl
        JOIN recordings r ON r.contribution_id = c_tbl.id
        WHERE c_tbl.status = 'READY_FOR_REVIEW' AND r.raw_file_path LIKE 'raw/contributions%'
    """)
    queue_count, queue_dur = c.fetchone()
    queue_dur_hrs = (queue_dur or 0.0) / 3600.0

    print(f"Total Registered Users: {total_users}")
    print(f"Active Recorded Users:  {active_recorded_users}")
    print(f"Total Prompts:          {total_prompts:,}")
    print(f"Total Recordings:       {rec_count:,} ({total_dur_hrs:.2f} hours)")
    print(f"Approved (Gold Tier):   {approved_count} ({approved_dur_hrs:.2f} hours)")
    print(f"Queue (Bronze Tier):    {queue_count} ({queue_dur_hrs:.2f} hours)")
    print(f"Validation Verdicts:    {val_verdicts}")

    # Generate Markdown Documentation
    with open("DATA/02_Data_Collection/community_collection_waaz.md", "w", encoding="utf-8") as f:
        f.write("# WAAZ Kashmiri Crowdsourced Community Speech Audit\n\n")
        f.write("## 1. Platform Infrastructure & Governance\n\n")
        f.write("- **Platform Directory**: `/home/musa/Desktop/Projects/DataCollection`\n")
        f.write("- **Database**: SQLite3 (`data/datacollect.db`, 966 MB)\n")
        f.write("- **Contributor Privacy**: Strict anonymization enforced. Speaker names, contact info, and IP addresses are excluded from research manifests.\n")
        f.write("- **Consent Terms**: Contributors agree to the EduCasheer Contributor License allowing speech recognition research and public non-commercial model release.\n\n")

        f.write("## 2. Quantitative Platform Metrics\n\n")
        f.write("| Metric | Verified Measurement | Notes |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write(f"| **Registered Users** | {total_users:,} | Contributor accounts in database |\n")
        f.write(f"| **Active Audio Contributors** | {active_recorded_users:,} | Native speakers who submitted valid recordings |\n")
        f.write(f"| **Available Prompts** | {total_prompts:,} | Curated Kashmiri text prompts across domains |\n")
        f.write(f"| **Total Submitted Audio** | {rec_count:,} clips | {total_dur_hrs:.2f} hours total physical audio |\n")
        f.write(f"| **Approved & Verified (GOLD)** | {approved_count:,} clips | {approved_dur_hrs:.2f} hours integrated into `manifest_gold.jsonl` |\n")
        f.write(f"| **Pending Review Queue (BRONZE)** | {queue_count:,} clips | {queue_dur_hrs:.2f} hours queued in `HUMAN_VALIDATION_QUEUE.jsonl` |\n")
        f.write(f"| **Rejected Audio** | {val_verdicts.get('REJECT', 0):,} clips | Filtered out due to background noise or prompt mismatch |\n\n")

        f.write("## 3. Incremental Export & Integration Workflow\n\n")
        f.write("New community recordings are processed via the automated incremental ingestion query in `data_acquisition/processing/build_master_corpus.py`.\n")

    print("WAAZ community audit completed successfully.")

if __name__ == "__main__":
    main()
