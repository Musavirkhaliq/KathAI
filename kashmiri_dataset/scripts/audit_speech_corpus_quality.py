#!/usr/bin/env python3
"""
Workstream 8 Stage 7 Phase H: Speech Corpus Quality Audit & Excluded Utterance Recovery
Audits the 3,057 excluded utterances from B-01 training (13.73 hours).
Formulates an engineering recovery plan, domain balance analysis, transcript review guidelines,
and a field collection plan for oratorical/religious Kashmiri speech (WAAZ).
"""

import json
import csv
from pathlib import Path
from collections import defaultdict, Counter

# Input paths
B01_EXCLUDED_PATH = Path("kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc/b01_excluded_samples.csv")
MASTER_MANIFEST = Path("kashmiri_dataset/manifests/manifest_master_consolidated.jsonl")
STAGE7_DIR = Path("kashmiri_dataset/experiments/native_asr_training/experiments/stage7_ctc_lm_decoding")
REPORTS_DIR = STAGE7_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print("=== Stage 7 Phase H: Speech Corpus Quality Audit & Recovery Plan ===")

# 1. Audit Excluded Utterances
excluded_samples = []
reason_counts = Counter()
source_counts = Counter()
total_excluded_duration = 0.0

with open(B01_EXCLUDED_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        dur = float(row["duration"])
        reason = row["exclusion_reason"]
        src = row["source"]
        
        excluded_samples.append(row)
        reason_counts[reason] += 1
        source_counts[src] += 1
        total_excluded_duration += dur

total_excluded = len(excluded_samples)
print(f"Total excluded samples: {total_excluded:,} ({total_excluded_duration/3600:.2f} hours)")

# 2. Formulate Utterance-by-Utterance Recovery Plan
recovery_rows = []
recovery_stats = Counter()
recovered_duration_by_action = defaultdict(float)

for row in excluded_samples:
    sid = row["sample_id"]
    src = row["source"]
    dur = float(row["duration"])
    reason = row["exclusion_reason"]
    
    # Categorize action
    if "duration_too_long" in reason:
        # Long utterances (>16.0s): can be split at silence pauses (VAD) and forced aligned
        action = "VAD_SPLIT_AND_REALIGN"
        feasibility = "HIGH"
        expected_segments = max(2, int(round(dur / 10.0)))
        note = f"Segment at >300ms pause into ~{expected_segments} sub-utterances"
        est_recovered_dur = dur * 0.95  # 95% recoverable after trimming silence
    elif "duration_too_short" in reason and "ctc_length_violation" in reason:
        action = "REJECT_MISALIGNED_TRANSCRIPT"
        feasibility = "ZERO"
        expected_segments = 0
        note = "Catastrophic text-audio mismatch (text length >> acoustic frames)"
        est_recovered_dur = 0.0
    elif "duration_too_short" in reason:
        # Short utterances (<0.5s): inspect duration and target length
        t_len = int(row.get("target_len", 0))
        if dur >= 0.35 and t_len <= 5:
            action = "RETAIN_IF_SNR_ADEQUATE"
            feasibility = "MEDIUM"
            expected_segments = 1
            note = f"Single-word or short interjection ({dur:.2f}s, {t_len} chars)"
            est_recovered_dur = dur
        else:
            action = "DISCARD_MICRO_FRAGMENT"
            feasibility = "LOW"
            expected_segments = 0
            note = f"Acoustic fragment (<0.35s) prone to boundary clipping"
            est_recovered_dur = 0.0
    elif "ctc_length_violation" in reason:
        action = "REJECT_MISALIGNED_TRANSCRIPT"
        feasibility = "ZERO"
        expected_segments = 0
        note = "CTC infeasible: transcript length exceeds downsampled frame count"
        est_recovered_dur = 0.0
    else:
        action = "MANUAL_INSPECTION"
        feasibility = "UNKNOWN"
        expected_segments = 1
        note = "Unclassified reason"
        est_recovered_dur = 0.0
        
    recovery_stats[action] += 1
    recovered_duration_by_action[action] += est_recovered_dur
    
    recovery_rows.append({
        "sample_id": sid,
        "source": src,
        "duration": dur,
        "original_reason": reason,
        "recovery_action": action,
        "feasibility": feasibility,
        "expected_segments": expected_segments,
        "estimated_recovered_seconds": round(est_recovered_dur, 2),
        "notes": note
    })

# Save Recovery Plan CSV
rec_csv_path = REPORTS_DIR / "excluded_utterance_recovery_plan.csv"
with open(rec_csv_path, "w", encoding="utf-8", newline="") as f:
    fieldnames = ["sample_id", "source", "duration", "original_reason", "recovery_action", "feasibility", "expected_segments", "estimated_recovered_seconds", "notes"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(recovery_rows)

print(f"Saved {len(recovery_rows):,} rows to {rec_csv_path}")

# 3. Domain Balance Analysis across Entire Master Manifest
domain_stats = defaultdict(lambda: {"samples": 0, "duration": 0.0, "speakers": set(), "domain": "Unknown"})

# Domain mapping
source_to_domain = {
    "GAASH-Lab/Bolbosh": "Read / Literary Prose",
    "humair025/KashmiriSpeech-IndicVoices": "Conversational / Field Interview",
    "xkaab_asr_4hrs": "Conversational / Crowdsourced",
    "muneebharoon/whisper-kashmiri": "Broadcast / Monologue",
    "openslr_slr122": "Read Speech / Clean Standard",
    "programindz_audio_corpus": "Single-Speaker Read Speech",
    "programindz/kashmiri-tts-single-speaker": "Studio TTS (Female)",
    "omarrran/kashmiri_single_speaker_ttd_hf_dataset_2024": "Studio TTS (Male)",
    "mendeley_ktts": "Studio TTS (Male)"
}

with open(MASTER_MANIFEST, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        src = item.get("source") or item.get("dataset_source") or "unknown"
        dur = float(item.get("duration", 0.0))
        spk = item.get("speaker_id")
        
        domain_stats[src]["samples"] += 1
        domain_stats[src]["duration"] += dur
        domain_stats[src]["domain"] = source_to_domain.get(src, "General Speech")
        if spk:
            domain_stats[src]["speakers"].add(spk)

total_corpus_dur = sum(d["duration"] for d in domain_stats.values())
total_corpus_samples = sum(d["samples"] for d in domain_stats.values())

domain_balance_rows = []
for src, d in sorted(domain_stats.items(), key=lambda x: x[1]["duration"], reverse=True):
    hrs = d["duration"] / 3600
    pct = (d["duration"] / total_corpus_dur) * 100.0
    domain_balance_rows.append({
        "source": src,
        "domain": d["domain"],
        "num_samples": d["samples"],
        "duration_hours": round(hrs, 2),
        "percentage_of_corpus": round(pct, 2),
        "speaker_count": len(d["speakers"])
    })

# Save Domain Balance CSV
dom_csv_path = REPORTS_DIR / "domain_balance_analysis.csv"
with open(dom_csv_path, "w", encoding="utf-8", newline="") as f:
    fieldnames = ["source", "domain", "num_samples", "duration_hours", "percentage_of_corpus", "speaker_count"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(domain_balance_rows)

print(f"Saved Domain Balance Analysis to {dom_csv_path}")

# 4. Generate Comprehensive Audit Report
audit_report_path = REPORTS_DIR / "speech_corpus_quality_audit.md"
with open(audit_report_path, "w", encoding="utf-8") as f:
    f.write("# Stage 7 Phase H: Speech Corpus Quality Audit & Recovery Plan\n\n")
    f.write("- **Date**: 2026-11-11\n")
    f.write(f"- **Total Excluded Utterances Audited**: {total_excluded:,} samples ({total_excluded_duration/3600:.2f} hours)\n")
    f.write(f"- **Total Master Corpus Analyzed**: {total_corpus_samples:,} samples ({total_corpus_dur/3600:.2f} hours)\n\n")
    
    f.write("## 1. Excluded Utterances Analysis & Recovery Potential\n\n")
    f.write("| Exclusion Category | Samples | Duration (hrs) | Action | Recoverable Duration |\n")
    f.write("|---|:---:|:---:|---|:---:|\n")
    for act, cnt in recovery_stats.most_common():
        rec_dur_hrs = recovered_duration_by_action[act] / 3600.0
        f.write(f"| `{act}` | {cnt:,} | {sum(float(r['duration']) for r in recovery_rows if r['recovery_action']==act)/3600:.2f} hrs | {act.replace('_', ' ')} | **{rec_dur_hrs:.2f} hrs** |\n")
    
    f.write(f"\n**Total Recoverable Speech**: **{sum(recovered_duration_by_action.values())/3600:.2f} hours** ({sum(recovered_duration_by_action.values())/total_excluded_duration*100:.1f}% recovery rate).\n\n")
    
    f.write("## 2. Recovery Engineering Protocol\n\n")
    f.write("### A. Long Utterances (> 16.0s, 2,496 samples, 13.67 hours)\n")
    f.write("- **Root Cause**: A 16.0s sequence cutoff was applied during B-01 fine-tuning to prevent GPU out-of-memory under Wav2Vec2's quadratic self-attention memory scaling.\n")
    f.write("- **Recovery Mechanism**: Implement a two-pass segmentation pipeline:\n")
    f.write("  1. Apply WebRTC VAD (or energy silence detection at -35 dB threshold with $\\ge 300$ ms silence gap) to identify natural breath and acoustic clause boundaries.\n")
    f.write("  2. Apply CTC-segmentation (Kurzinger et al.) using the frozen B-01 acoustic model to align ground-truth character sub-sequences to the acoustic segments.\n")
    f.write("  3. Retain split segments between 2.0s and 14.0s. This will yield an estimated ~5,500 new high-quality training utterances (~12.98 hours).\n\n")
    
    f.write("### B. Short Utterances (< 0.5s, 558 samples, 0.065 hours)\n")
    f.write("- **Root Cause**: Strict duration threshold discarded micro-utterances.\n")
    f.write("- **Recovery Mechanism**: Filter for utterances between 0.35s and 0.50s with SNR $\\ge 15$ dB containing complete monosyllabic words (e.g. `نہ`, `آ`, `تہٕ`, `چھُ`). Discard truncated fragments (<0.35s) where phoneme onsets or codas are clipped.\n\n")
    
    f.write("### C. CTC Infeasible Utterances (3 samples, 0.000 hours)\n")
    f.write("- **Root Cause**: Complete metadata/audio misalignment where an entire paragraph (e.g. 393 characters) was attached to a 0.44s audio clip. In CTC loss, target length $U$ cannot exceed acoustic frame count $T$.\n")
    f.write("- **Action**: Permanently purge these 3 misaligned utterances from the speech corpus.\n\n")
    
    f.write("## 3. Domain Balance Analysis\n\n")
    f.write("| Source | Domain | Samples | Hours | Share (%) | Native Speakers |\n")
    f.write("|---|---|:---:|:---:|:---:|:---:|\n")
    for r in domain_balance_rows:
        f.write(f"| `{r['source']}` | {r['domain']} | {r['num_samples']:,} | {r['duration_hours']} h | {r['percentage_of_corpus']}% | {r['speaker_count']} |\n")
        
    f.write("\n## 4. Key Corpus Imbalances Identified\n\n")
    f.write("1. **Literary Read Speech Dominance**: Bolbosh constitutes 56.4% of total speech duration (79.9 hours) across only 40 speakers (average ~2.0 hours per speaker). This creates a heavy acoustic prior toward broadcast/standard read speech.\n")
    f.write("2. **Conversational Speaker Breadth**: IndicVoices provides the widest demographic diversity (482 speakers, 37.18 hours, 26.3% of corpus), capturing authentic spontaneous Kashmiri field dialogue.\n")
    f.write("3. **Severe Gap in Oratorical / Religious Kashmiri (WAAZ)**: The corpus has 0.0 hours of traditional Kashmiri religious sermons, community majalis, and classical oratorical speech (WAAZ), which feature unique elevated vocabulary, archaic Persian/Arabic loanwords, and rhetorical prosody.\n")

print(f"Saved speech quality audit report to {audit_report_path}")

# 5. Generate Transcript Review Guidelines
guidelines_path = REPORTS_DIR / "transcript_review_guidelines.md"
with open(guidelines_path, "w", encoding="utf-8") as f:
    f.write("# Kashmiri ASR Transcript Review & Orthographic Guidelines\n\n")
    f.write("## Purpose\n")
    f.write("Standardized protocols for human reviewers, annotators, and computational linguists auditing and correcting Kashmiri speech transcripts for EduCasheer Kashmiri AI.\n\n")
    f.write("## 1. Perso-Arabic Orthographic Standardization Rules\n\n")
    f.write("1. **Kaf Normalization**: Always transcribe Kashmiri /k/ using Kashmiri Keheh (`\u06a9`). Never use Arabic Kaf (`\u0643`) or Swash Kaf (`\u06aa`).\n")
    f.write("2. **Heh Goal Normalization**: Transcribe Kashmiri /h/ using Heh Goal (`\u06c1`) or Do-Chashmi Heh (`\u06be`) for aspirated consonants (`کھ`, `پھ`, `تھ`, `چھ`, `جھ`). Never use Arabic Heh (`\u0647`) or Teh Marbuta (`\u0629`).\n")
    f.write("3. **Vocalic Diacritics**: Kashmiri vowels must follow standard Academy orthography:\n")
    f.write("   - Zer (`\u0650`): short /i/\n")
    f.write("   - Pesh (`\u064f`): short /u/\n")
    f.write("   - Zabar (`\u064e`): short /a/\n")
    f.write("   - Kashmiri Palatalized Yeh (`\u06cd` / `ۍ`): word-final Kashmiri palatalization.\n")
    f.write("   - Kashmiri Waw with Circle (`\u06c4` / `ۄ`): open-mid back vowel /ɔ/.\n")
    f.write("   - Kashmiri Hamza above (`\u0654`): used on vowels like `ٲ`, `ٚ`, `ٛ`.\n\n")
    f.write("## 2. Spoken Disfluency & Code-Switching Conventions\n\n")
    f.write("1. **Verbatim Spoken Representation**: Transcribe what the speaker actually articulated, including hesitations (`اَہ`, `مَطلب`), repetitions, and false starts.\n")
    f.write("2. **Filler Words**: Standardize ubiquitous Kashmiri fillers:\n")
    f.write("   - `مطلب` (matlab / I mean)\n")
    f.write("   - `تہٕ بیٚیہِ` (te beye / and then)\n")
    f.write("   - `اَچھا` (achha / okay)\n")
    f.write("3. **English Borrowings & Code-Switching**:\n")
    f.write("   - Common loanwords must be phonetically transcribed in Kashmiri Perso-Arabic script:\n")
    f.write("     - 'doctor' -> `ڈاکٹَر`\n")
    f.write("     - 'mobile' -> `موبٲیِل`\n")
    f.write("     - 'school' -> `سکول`\n")
    f.write("     - 'hospital' -> `ہَسپتال`\n")
    f.write("   - Do NOT insert Latin characters (`doctor`, `mobile`) into Kashmiri Perso-Arabic transcripts.\n\n")
    f.write("## 3. Exclusion Flags for Transcribers\n\n")
    f.write("- `[INCOMPREHENSIBLE]`: Speech masked by severe acoustic noise (SNR < 5 dB).\n")
    f.write("- `[OVERLAPPING_SPEECH]`: Two or more speakers talking simultaneously.\n")
    f.write("- `[NON_KASHMIRI]`: Utterances completely spoken in Urdu, Hindi, or English.\n")

print(f"Saved transcript review guidelines to {guidelines_path}")

# 6. Generate WAAZ Collection Improvement Plan
waaz_plan_path = REPORTS_DIR / "waaz_collection_improvement_plan.md"
with open(waaz_plan_path, "w", encoding="utf-8") as f:
    f.write("# Field Collection & Expansion Plan: Kashmiri Oratorical & Religious Speech (WAAZ)\n\n")
    f.write("## 1. Motivation & Linguistic Importance\n\n")
    f.write("Kashmiri *WAAZ* (traditional religious lectures, Friday khutbahs, and spiritual discourses) represents one of the richest and most culturally salient oral traditions in Kashmir. WAAZ speech features:\n")
    f.write("- Elevated literary registers with classical Persian, Arabic, and Sanskrit loanwords.\n")
    f.write("- Dynamic acoustic prosody, high vocal projection, rhythmic cadence, and oratorical passion.\n")
    f.write("- Public address acoustic environments (mosques, shrines, outdoor community gatherings) with reverberation.\n\n")
    f.write("Currently, WAAZ is represented with **0.0 hours** in our speech corpus, representing a critical domain gap.\n\n")
    f.write("## 2. Collection Target & Milestones\n\n")
    f.write("- **Phase 1 Target**: 25.0 hours of high-quality oratorical speech.\n")
    f.write("- **Speakers**: 15–20 distinct prominent Kashmiri orators and scholars representing North Kashmir (Baramulla, Kupwara), Central Kashmir (Srinagar, Budgam), and South Kashmir (Anantnag, Shopian).\n")
    f.write("- **Audio Format**: 48 kHz / 24-bit lossless master WAV, downsampled to 16 kHz mono.\n\n")
    f.write("## 3. Technical Recording & Curation Workflow\n\n")
    f.write("1. **Acoustic Acquisition**:\n")
    f.write("   - Direct digital line-out feed from sound mixing consoles at venues where available (eliminating room reverberation).\n")
    f.write("   - High-sensitivity directional lavalier or cardioid dynamic microphones (Shure SM58 / Rode Wireless PRO) placed 15–20 cm from orator.\n")
    f.write("2. **Segmentation & Alignment**:\n")
    f.write("   - Use Silero VAD to segment 45–60 minute continuous discourses into 3.0s–12.0s acoustic phrases at pause boundaries.\n")
    f.write("   - Discard crowd applause, background chants, and call-and-response sections.\n")
    f.write("3. **Two-Tier Expert Transcription**:\n")
    f.write("   - Tier 1: Initial draft transcription by native Kashmiri linguists.\n")
    f.write("   - Tier 2: Secondary verification by Islamic studies and Kashmiri literary scholars to ensure exact orthographic fidelity for theological terminology and poetic quotations.\n")

print(f"Saved WAAZ collection plan to {waaz_plan_path}")
print("Phase H Execution Complete!")
