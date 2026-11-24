#!/usr/bin/env python3
import os
import json
import sqlite3

def main():
    print("Building Human Validation & Review Queue...")
    out_queue_path = "data_acquisition/reports/HUMAN_VALIDATION_QUEUE.jsonl"
    
    queue_records = []

    # 1. WAAZ Platform Review Queue (READY_FOR_REVIEW contributions)
    db_path = "/home/musa/Desktop/Projects/DataCollection/data/datacollect.db"
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("""
            SELECT r.id, r.raw_file_path, r.duration_seconds, t.text, c_tbl.created_at
            FROM contributions c_tbl
            JOIN recordings r ON r.contribution_id = c_tbl.id
            LEFT JOIN transcripts t ON t.id = r.id
            WHERE c_tbl.status = 'READY_FOR_REVIEW' AND r.raw_file_path LIKE 'raw/contributions%'
        """)
        rows = c.fetchall()
        for rid, rpath, rdur, rtxt, sub_date in rows:
            rec = {
                "queue_id": f"queue_waaz_{rid}",
                "source_dataset": "WAAZ_Community_Platform",
                "audio_path": rpath,
                "duration_seconds": float(rdur or 0.0),
                "current_transcript": rtxt,
                "priority": "HIGH",
                "validation_task": "VERIFY_COMMUNITY_READING",
                "instructions": "Listen to the recording and verify if spoken Kashmiri exactly matches the prompt text.",
                "review_status": "PENDING"
            }
            queue_records.append(rec)
        print(f"Added {len(rows)} WAAZ community contributions to validation queue.")

    # 2. Dr. War Broadcast Commentary (ASR Hypotheses needing Human Polish)
    war_manifest = "kashmiri_dataset/manifests/source_manifests/manifest_quran_dr_war.jsonl"
    if os.path.exists(war_manifest):
        war_cnt = 0
        with open(war_manifest, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                # Sample 1,000 top clips for human review
                if war_cnt < 1000:
                    rec = {
                        "queue_id": f"queue_war_{r.get('sample_id')}",
                        "source_dataset": "Dr_War_Broadcast_Series",
                        "audio_path": r.get("canonical_audio_path") or r.get("original_audio_path"),
                        "duration_seconds": float(r.get("duration_seconds") or 0.0),
                        "current_transcript": r.get("transcript_normalized"),
                        "priority": "MEDIUM",
                        "validation_task": "TRANSCRIBE_SPONTANEOUS_BROADCAST",
                        "instructions": "Transcribe the spontaneous Kashmiri broadcast commentary verbatim in standard Perso-Arabic orthography.",
                        "review_status": "PENDING"
                    }
                    queue_records.append(rec)
                    war_cnt += 1
        print(f"Added {war_cnt} Dr. War broadcast clips to validation queue.")

    # Write output
    with open(out_queue_path, "w", encoding="utf-8") as f:
        for r in queue_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Total human validation queue items: {len(queue_records):,}. Written to {out_queue_path}")

if __name__ == "__main__":
    main()
