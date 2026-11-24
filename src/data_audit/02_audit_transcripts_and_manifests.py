#!/usr/bin/env python3
"""
Step 2: Audit all transcript and manifest representations across the project.
Extracts all candidate audiotext pairs, classifies linguistic correctness,
detects script/language, evaluates alignment categories (A through I),
and records provenance and validation status.
"""

import os
import sys
import json
import re
import tarfile
import sqlite3
import pandas as pd
import pyarrow.parquet as pq

# Kashmiri-specific Perso-Arabic characters (beyond standard Arabic/Urdu):
# ٚ (U+065A vowel sign small v above), ٛ (U+065B inverted v), ؠ (U+0620 Kashmiri yeh),
# ۄ (U+0644 Kashmiri waw with ring/loop), ۆ (U+06C6), ۄ (U+06C4), ے, چ, پ, ژ, گ, ڈ, ٹ, ڑ
KASHMIRI_SPECIFIC_CHARS = set("ٚٛؠۄۆژٹڈڑپچگ")
ARABIC_REC_KEYWORDS = ["بِسْمِ", "الرَّحْمَٰنِ", "الرَّحِيمِ", "الْحَمْدُ", "لِلَّهِ", "رَبِّ", "الْعَالَمِينَ", "صَدَقَ اللَّهُ"]

def detect_script(text):
    if not text or not isinstance(text, str) or not text.strip():
        return "None"
    text = text.strip()
    
    # Check Unicode ranges
    has_arabic = bool(re.search(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', text))
    has_deva = bool(re.search(r'[\u0900-\u097F]', text))
    has_latin = bool(re.search(r'[A-Za-z]', text))
    
    if has_arabic:
        return "kas_Arab" if any(c in KASHMIRI_SPECIFIC_CHARS for c in text) else "Arab"
    elif has_deva:
        return "Deva"
    elif has_latin:
        return "Latn"
    return "Other"

def detect_language(text, script, source_name=""):
    if not text or not isinstance(text, str) or not text.strip():
        return "None"
    text = text.strip()
    
    # Check Arabic recitation indicators
    if any(kw in text for kw in ARABIC_REC_KEYWORDS) and "quran" in source_name.lower():
        # Check if it has genuine Kashmiri translation text or pure Arabic
        if any(c in KASHMIRI_SPECIFIC_CHARS for c in text) or any(w in text for w in ["چھُ", "چھِ", "ہُند", "ہِند", "سُوی", "پزن", "تٔسۍ"]):
            return "kas"
        return "ara"
        
    if script == "kas_Arab":
        return "kas"
    if script == "Deva":
        return "kas_Deva" # Kashmiri in Devanagari or Hindi
    if script == "Latn":
        return "eng"
        
    # Standard Arabic vs Urdu vs Kashmiri without special diacritics
    kashmiri_common_words = ["چھُ", "چھِ", "چھَ", "تہٕ", "تہِ", "ہُند", "ہِند", "یِوان", "کران", "گژھان", "ناو", "میہ", "تُہند", "کٲشُر", "پزَن"]
    if any(w in text for w in kashmiri_common_words):
        return "kas"
    
    # If source is known Kashmiri corpus (Bolbosh, IndicVoices, OpenSLR)
    if any(k in source_name.lower() for k in ["bolbosh", "indicvoices", "openslr", "waaz", "ktts", "programindz", "xkaab"]):
        return "kas"
        
    return "und" # Undetermined

def audit_transcripts_and_manifests():
    print("=" * 60)
    print("STEP 2: TRANSCRIPT & MANIFEST AUDIT")
    print("=" * 60)

    records = []

    # 1. Master Training-Eligible Manifest (106,487 records)
    m_path = "asr_training_eligible_manifest.jsonl"
    if os.path.exists(m_path):
        print(f"\nProcessing {m_path}...")
        with open(m_path) as f:
            for line in f:
                d = json.loads(line)
                sid = d.get("sample_id", "")
                text = d.get("transcript_original", d.get("transcript_normalized", ""))
                norm_text = d.get("transcript_normalized", "")
                src = d.get("source_dataset", "")
                c_audio = d.get("canonical_audio_path", d.get("original_audio_path", ""))
                dur = float(d.get("duration_seconds", 0))
                
                script = detect_script(text)
                lang = detect_language(text, script, src)
                
                # Eligibility and alignment status
                align_status = d.get("alignment_status", "verified")
                elig = d.get("eligibility_status", "training_eligible")
                
                cat = "A" if align_status in ("verified", "canonical_verified", "aligned") else "B"
                if "dr_war" in src.lower() and not text.strip():
                    cat = "G"
                elif "recitation" in sid.lower() or lang == "ara":
                    cat = "F"
                
                records.append({
                    "manifest_record_id": sid,
                    "manifest_name": "asr_training_eligible_manifest.jsonl",
                    "source_dataset": src,
                    "audio_filename": os.path.basename(c_audio),
                    "audio_path": c_audio,
                    "audio_sha256": d.get("audio_sha256", ""),
                    "pcm_sha256": d.get("pcm_sha256", ""),
                    "duration_seconds": dur,
                    "speaker_id": d.get("speaker_id", ""),
                    "text": text,
                    "normalized_text": norm_text,
                    "script": script,
                    "language": lang,
                    "provenance": d.get("provenance", "master_inventory_v1"),
                    "validation_status": "HUMAN_OR_CANONICAL_VERIFIED" if d.get("transcript_verified", False) else "MACHINE_FILTERED",
                    "alignment_category": cat,
                    "in_master_214h": True
                })
        print(f"  Loaded {len(records)} records from {m_path}.")

    # 2. Source Manifests in kashmiri_dataset/manifests/source_manifests/
    sm_dir = "kashmiri_dataset/manifests/source_manifests"
    print(f"\nProcessing source manifests in {sm_dir}...")
    for f in sorted(os.listdir(sm_dir)):
        if f.endswith(".jsonl"):
            sp = os.path.join(sm_dir, f)
            cnt = 0
            with open(sp) as fp:
                for line in fp:
                    try:
                        d = json.loads(line)
                        cnt += 1
                        sid = d.get("sample_id", "")
                        text = d.get("text", d.get("transcript", ""))
                        norm_text = d.get("normalized_text", text)
                        src = d.get("source", os.path.splitext(f)[0])
                        apath = d.get("audio_path", "")
                        dur = float(d.get("duration", d.get("duration_seconds", 0)))
                        
                        script = detect_script(text)
                        lang = detect_language(text, script, src)
                        
                        # Category
                        cat = "A" if d.get("flags") == "ok" else "B"
                        if "dr_war" in f.lower() and not text.strip():
                            cat = "G" # Dr War in raw manifest has 0 text!
                        elif "recitation" in sid.lower() or lang == "ara":
                            cat = "F"
                            
                        records.append({
                            "manifest_record_id": sid,
                            "manifest_name": f"source_manifests/{f}",
                            "source_dataset": src,
                            "audio_filename": os.path.basename(apath),
                            "audio_path": apath,
                            "audio_sha256": d.get("audio_sha256", ""),
                            "pcm_sha256": d.get("pcm_sha256", ""),
                            "duration_seconds": dur,
                            "speaker_id": d.get("speaker_id", ""),
                            "text": text,
                            "normalized_text": norm_text,
                            "script": script,
                            "language": lang,
                            "provenance": f"source_manifest_{f}",
                            "validation_status": "SOURCE_CANONICAL",
                            "alignment_category": cat,
                            "in_master_214h": False # will be cross-indexed
                        })
                    except Exception as e:
                        pass
            print(f"  {f}: {cnt} records")

    # 3. DataCollection platform sqlite (datacollect.db)
    db_path = "/home/musa/Desktop/Projects/DataCollection/data/datacollect.db"
    if os.path.exists(db_path):
        print(f"\nProcessing DataCollection database: {db_path}...")
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("""
            SELECT 
                r.id,
                r.raw_file_path,
                r.processed_file_path,
                r.duration_seconds,
                r.sha256_hash,
                c.speaker_id,
                c.campaign_id,
                t.text,
                t.is_verified,
                t.transcript_type,
                qa.quality_score,
                qa.snr_db,
                qa.is_clipped,
                qa.is_silent,
                qa.is_noisy
            FROM recordings r
            LEFT JOIN contributions c ON r.contribution_id = c.id
            LEFT JOIN transcripts t ON t.recording_id = r.id
            LEFT JOIN quality_assessments qa ON qa.recording_id = r.id
            WHERE r.raw_file_path LIKE '%contributions%'
        """)
        rows = c.fetchall()
        dc_cnt = 0
        for r_id, raw_p, proc_p, dur_s, sha, spk_id, camp_id, text, is_ver, tr_type, q_score, snr, is_clip, is_sil, is_noisy in rows:
            dc_cnt += 1
            script = detect_script(text)
            lang = detect_language(text, script, "waaz_community")
            
            # WAAZ quality tiers
            cat = "A" if is_ver == 1 else "B"
            if is_sil or is_clip:
                cat = "I"
            elif not text or not text.strip():
                cat = "G"
            
            val_stat = "WAAZ_VERIFIED_HUMAN" if is_ver == 1 else "WAAZ_COMMUNITY_PROMPT"
            
            records.append({
                "manifest_record_id": f"waaz_dc_{r_id}",
                "manifest_name": "DataCollection/datacollect.db",
                "source_dataset": "WAAZ_Community",
                "audio_filename": os.path.basename(proc_p if proc_p else raw_p),
                "audio_path": proc_p if proc_p else raw_p,
                "audio_sha256": sha if sha else "",
                "pcm_sha256": "",
                "duration_seconds": float(dur_s) if dur_s else 0.0,
                "speaker_id": spk_id if spk_id else "unknown_waaz",
                "text": text if text else "",
                "normalized_text": text if text else "",
                "script": script,
                "language": lang,
                "provenance": "waaz_live_platform",
                "validation_status": val_stat,
                "alignment_category": cat,
                "in_master_214h": False
            })
        print(f"  Loaded {dc_cnt} live community recordings from DataCollection.")
        conn.close()

    # 4. OpenSLR tar manifest (aazu34)
    openslr_tar = "kashmiri_dataset/external/aazu34_kashmiri-openslr122-segments/openslr122-seg.tar.gz"
    if os.path.exists(openslr_tar):
        print(f"\nProcessing OpenSLR tar manifest...")
        with tarfile.open(openslr_tar, "r:gz") as tar:
            if "openslr122-seg/manifest.jsonl" in tar.getnames():
                f = tar.extractfile("openslr122-seg/manifest.jsonl")
                for line in f:
                    d = json.loads(line)
                    fname = d.get("file", "")
                    text = d.get("text", "")
                    dur = d.get("duration", 0.0)
                    script = detect_script(text)
                    lang = detect_language(text, script, "openslr")
                    records.append({
                        "manifest_record_id": f"openslr122_seg_{fname}",
                        "manifest_name": "openslr122-seg.tar.gz:manifest.jsonl",
                        "source_dataset": "OpenSLR_122_Segments",
                        "audio_filename": fname,
                        "audio_path": f"{openslr_tar}:{fname}",
                        "audio_sha256": "",
                        "pcm_sha256": "",
                        "duration_seconds": float(dur) if dur else 0.0,
                        "speaker_id": "openslr_speaker",
                        "text": text,
                        "normalized_text": text,
                        "script": script,
                        "language": lang,
                        "provenance": "openslr122_hf_segments",
                        "validation_status": "HUMAN_TRANSCRIBED",
                        "alignment_category": "A",
                        "in_master_214h": False
                    })

    # 5. Programindz audio corpus parquet
    prog_parquet = "kashmiri_dataset/external/programindz_kashmiri-audio-corpus/data/train-00000-of-00001.parquet"
    if os.path.exists(prog_parquet):
        print(f"\nProcessing Programindz audio corpus parquet...")
        t = pq.read_table(prog_parquet, columns=["text", "duration"])
        texts = t.column("text").to_pylist()
        durs = t.column("duration").to_pylist()
        for idx, (txt, dur) in enumerate(zip(texts, durs)):
            script = detect_script(txt)
            lang = detect_language(txt, script, "programindz")
            records.append({
                "manifest_record_id": f"programindz_kac_{idx:05d}",
                "manifest_name": "programindz_kashmiri-audio-corpus.parquet",
                "source_dataset": "programindz_audio_corpus",
                "audio_filename": f"kac_{idx:06d}.wav",
                "audio_path": f"{prog_parquet}#row={idx}",
                "audio_sha256": "",
                "pcm_sha256": "",
                "duration_seconds": float(dur),
                "speaker_id": "programindz_speaker",
                "text": txt,
                "normalized_text": txt,
                "script": script,
                "language": lang,
                "provenance": "programindz_hf_corpus",
                "validation_status": "HUMAN_TRANSCRIBED",
                "alignment_category": "A",
                "in_master_214h": False
            })

    # 6. S. Hassan Dictionary CSV
    shassan_csv = "kashmiri_dataset/raw/external/mzmmoazam_dataset/csv_files/S_Hassan_dictionary.csv"
    if os.path.exists(shassan_csv):
        print(f"\nProcessing S_Hassan dictionary CSV...")
        df_sh = pd.read_csv(shassan_csv)
        for _, r in df_sh.iterrows():
            w = str(r.get("word", ""))
            fn = str(r.get("file_name", ""))
            records.append({
                "manifest_record_id": f"uchicago_{fn}",
                "manifest_name": "S_Hassan_dictionary.csv",
                "source_dataset": "UChicago_SHassan_Dictionary",
                "audio_filename": fn if fn.endswith(".mp3") else f"{fn}.mp3",
                "audio_path": f"kashmiri_dataset/raw/external/mzmmoazam_dataset/downloaded_content/content/kashmiri_dataset/downloaded_content/uchicago_S_Hassan/{fn}.mp3",
                "audio_sha256": "",
                "pcm_sha256": "",
                "duration_seconds": 1.0, # estimated dictionary word length
                "speaker_id": "s_hassan_narrator",
                "text": w,
                "normalized_text": w,
                "script": detect_script(w),
                "language": "kas_Latn" if detect_script(w) == "Latn" else "kas",
                "provenance": "uchicago_dictionary_lexicon",
                "validation_status": "LEXICON_HEADWORD",
                "alignment_category": "B", # single word pronunciation
                "in_master_214h": False
            })

    df_out = pd.DataFrame(records)
    out_parquet = "src/data_audit/audit_manifest_pairs.parquet"
    df_out.to_parquet(out_parquet, index=False)
    print(f"\nSaved transcript and manifest audit to {out_parquet} ({len(df_out)} total records).")

    print("\n" + "=" * 60)
    print("MANIFEST & TRANSCRIPT AUDIT SUMMARY")
    print("=" * 60)
    print("\nBreakdown by Source Dataset:")
    print(df_out["source_dataset"].value_counts())
    print("\nBreakdown by Script:")
    print(df_out["script"].value_counts())
    print("\nBreakdown by Language:")
    print(df_out["language"].value_counts())
    print("\nBreakdown by Alignment Category:")
    print(df_out["alignment_category"].value_counts())

if __name__ == "__main__":
    audit_transcripts_and_manifests()
