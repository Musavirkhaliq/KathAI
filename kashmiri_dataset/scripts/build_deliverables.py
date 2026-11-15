import json
import csv
import os
import random
from collections import defaultdict, Counter

os.makedirs("QURAN_HUMAN_REVIEW_PACKAGE", exist_ok=True)

kf_manifest = "kashmiri_dataset/manifests/source_manifests/manifest_quran_king_fahd.jsonl"
war_manifest = "kashmiri_dataset/manifests/source_manifests/manifest_quran_dr_war.jsonl"

candidates = []
uncertain = []

# Process King Fahd
with open(kf_manifest, "r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        cat = rec.get("category", "")
        # Set proper verification status
        if cat == "kashmiri_translation":
            rec["verification_status"] = "tier2_machine_aligned"
            rec["candidate_pool"] = "kashmiri_translation"
            candidates.append(rec)
        elif cat == "kashmiri_commentary":
            rec["verification_status"] = "tier1_machine_detected"
            rec["candidate_pool"] = "kashmiri_commentary"
            candidates.append(rec)
        elif cat in ["uncertain_mixed_speech", "arabic_recitation"]:
            rec["uncertainty_reason"] = "arabic_recitation" if cat == "arabic_recitation" else "uncertain_mixed_speech_boundary"
            uncertain.append(rec)
        else:
            rec["uncertainty_reason"] = "unclassified"
            uncertain.append(rec)

# Process Dr. War
with open(war_manifest, "r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        cat = rec.get("category", "")
        if cat == "kashmiri_commentary":
            rec["verification_status"] = "tier1_machine_detected"
            rec["candidate_pool"] = "kashmiri_commentary"
            candidates.append(rec)
        elif cat in ["uncertain_mixed_speech", "mixed_kashmiri_arabic", "rejected"]:
            rec["uncertainty_reason"] = f"{cat}_{rec.get('flags', 'none')}"
            uncertain.append(rec)
        else:
            rec["uncertainty_reason"] = "unclassified"
            uncertain.append(rec)

print(f"Total Kashmiri Candidates: {len(candidates)}")
print(f"Total Uncertain / Recitation Segments: {len(uncertain)}")

# Write QURAN_NEW_ASR_CANDIDATES.jsonl
with open("QURAN_NEW_ASR_CANDIDATES.jsonl", "w", encoding="utf-8") as f:
    for c in candidates:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

# Write QURAN_UNCERTAIN_SPEECH.jsonl
with open("QURAN_UNCERTAIN_SPEECH.jsonl", "w", encoding="utf-8") as f:
    for u in uncertain:
        f.write(json.dumps(u, ensure_ascii=False) + "\n")

# Compute Durations
cand_dur = sum(c["duration"] for c in candidates)
unc_dur = sum(u["duration"] for u in uncertain)
print(f"Candidates Duration: {cand_dur/3600:.4f} hrs ({cand_dur:.2f} s)")
print(f"Uncertain Duration: {unc_dur/3600:.4f} hrs ({unc_dur:.2f} s)")

# Prepare Stratified Sample of >= 400 clips for Human Review Package
random.seed(42)

kf_trans_high = [c for c in candidates if c.get("source") == "SRC_QURAN_KING_FAHD" and c.get("category") == "kashmiri_translation" and c.get("alignment_sim", 0) > 0.5]
kf_trans_mod = [c for c in candidates if c.get("source") == "SRC_QURAN_KING_FAHD" and c.get("category") == "kashmiri_translation" and c.get("alignment_sim", 0) <= 0.5]
kf_comm_short = [c for c in candidates if c.get("source") == "SRC_QURAN_KING_FAHD" and c.get("category") == "kashmiri_commentary" and c["duration"] < 5.0]
kf_comm_long = [c for c in candidates if c.get("source") == "SRC_QURAN_KING_FAHD" and c.get("category") == "kashmiri_commentary" and c["duration"] >= 5.0]

kf_arabic = [u for u in uncertain if u.get("source") == "SRC_QURAN_KING_FAHD" and u.get("category") == "arabic_recitation"]
kf_unc = [u for u in uncertain if u.get("source") == "SRC_QURAN_KING_FAHD" and u.get("category") == "uncertain_mixed_speech"]

war_comm_high_snr = [c for c in candidates if c.get("source") == "SRC_QURAN_DR_WAR" and c.get("snr", 0) >= 15.0]
war_comm_med_snr = [c for c in candidates if c.get("source") == "SRC_QURAN_DR_WAR" and 10.0 <= c.get("snr", 0) < 15.0]
war_comm_low_snr = [c for c in candidates if c.get("source") == "SRC_QURAN_DR_WAR" and c.get("snr", 0) < 10.0]

war_unc = [u for u in uncertain if u.get("source") == "SRC_QURAN_DR_WAR" and u.get("category") in ["uncertain_mixed_speech", "mixed_kashmiri_arabic"]]
war_rej = [u for u in uncertain if u.get("source") == "SRC_QURAN_DR_WAR" and u.get("category") == "rejected"]

sample_groups = [
    ("KF_Translation_HighAlign", kf_trans_high, 60),
    ("KF_Translation_ModAlign", kf_trans_mod, 40),
    ("KF_Commentary_Short", kf_comm_short, 40),
    ("KF_Commentary_Long", kf_comm_long, 40),
    ("KF_Arabic_Recitation_Control", kf_arabic, 30),
    ("KF_Uncertain_Speech", kf_unc, 30),
    ("War_Commentary_HighSNR", war_comm_high_snr, 50),
    ("War_Commentary_MedSNR", war_comm_med_snr, 50),
    ("War_Commentary_LowSNR", war_comm_low_snr, 40),
    ("War_Uncertain_Mixed", war_unc, 30),
    ("War_Rejected_Noise", war_rej, 30),
]

sampled_clips = []
for stratum_name, pool, target_k in sample_groups:
    k = min(len(pool), target_k)
    chosen = random.sample(pool, k)
    for item in chosen:
        item_copy = dict(item)
        item_copy["review_stratum"] = stratum_name
        item_copy["review_priority"] = "high" if "Translation" in stratum_name else ("medium" if "Commentary" in stratum_name else "audit_check")
        sampled_clips.append(item_copy)

print(f"Total Stratified Human Review Clips Sampled: {len(sampled_clips)}")

with open("QURAN_HUMAN_REVIEW_PACKAGE/pilot_review_samples.jsonl", "w", encoding="utf-8") as f:
    for s in sampled_clips:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

# Consolidated Coverage Audit CSV
cov_headers = [
    "source", "recording_id", "filename", "total_duration_s", 
    "speech_duration_s", "silence_duration_s", "num_segments", 
    "speech_ratio", "kashmiri_duration_s", "arabic_duration_s", 
    "uncertain_duration_s", "mean_segment_dur_s"
]

cov_rows = []

# King Fahd Coverage
with open("kashmiri_dataset/quality_reports/quran_king_fahd_coverage_audit.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        s_num = r["surah_number"]
        fname = r["filename"]
        file_segs = [c for c in candidates if c.get("source_file") == fname] + [u for u in uncertain if u.get("source_file") == fname]
        kas_dur = sum(x["duration"] for x in file_segs if x.get("category") in ["kashmiri_translation", "kashmiri_commentary"])
        ara_dur = sum(x["duration"] for x in file_segs if x.get("category") == "arabic_recitation")
        unc_dur = sum(x["duration"] for x in file_segs if x.get("category") == "uncertain_mixed_speech")
        mean_dur = sum(x["duration"] for x in file_segs) / len(file_segs) if file_segs else 0.0
        
        cov_rows.append({
            "source": "SRC_QURAN_KING_FAHD",
            "recording_id": f"s{int(s_num):03d}",
            "filename": fname,
            "total_duration_s": r["total_duration_s"],
            "speech_duration_s": r["speech_duration_s"],
            "silence_duration_s": r["silence_duration_s"],
            "num_segments": r["num_segments"],
            "speech_ratio": r["speech_ratio"],
            "kashmiri_duration_s": f"{kas_dur:.3f}",
            "arabic_duration_s": f"{ara_dur:.3f}",
            "uncertain_duration_s": f"{unc_dur:.3f}",
            "mean_segment_dur_s": f"{mean_dur:.3f}"
        })

# Dr War Coverage
with open("kashmiri_dataset/quality_reports/quran_dr_war_coverage_audit.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        ep_num = r["episode_number"]
        fname = r["filename"]
        file_segs = [c for c in candidates if c.get("source_file") == fname] + [u for u in uncertain if u.get("source_file") == fname]
        kas_dur = sum(x["duration"] for x in file_segs if x.get("category") == "kashmiri_commentary")
        ara_dur = 0.0
        unc_dur = sum(x["duration"] for x in file_segs if x.get("category") in ["uncertain_mixed_speech", "mixed_kashmiri_arabic", "rejected"])
        mean_dur = sum(x["duration"] for x in file_segs) / len(file_segs) if file_segs else 0.0
        
        cov_rows.append({
            "source": "SRC_QURAN_DR_WAR",
            "recording_id": f"ep{int(ep_num):02d}",
            "filename": fname,
            "total_duration_s": r["total_duration_s"],
            "speech_duration_s": r["speech_duration_s"],
            "silence_duration_s": r["silence_duration_s"],
            "num_segments": r["num_segments"],
            "speech_ratio": r["speech_ratio"],
            "kashmiri_duration_s": f"{kas_dur:.3f}",
            "arabic_duration_s": f"{ara_dur:.3f}",
            "uncertain_duration_s": f"{unc_dur:.3f}",
            "mean_segment_dur_s": f"{mean_dur:.3f}"
        })

with open("QURAN_INDEPENDENT_COVERAGE_AUDIT.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=cov_headers)
    writer.writeheader()
    writer.writerows(cov_rows)

print(f"Generated QURAN_INDEPENDENT_COVERAGE_AUDIT.csv with {len(cov_rows)} recording rows.")
