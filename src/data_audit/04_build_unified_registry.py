#!/usr/bin/env python3
"""
Step 4: Build the Canonical Unified Audio-Text Pair Registry and Deliverable CSVs.
Merges physical audio, manifests, benchmark exclusions, and training memberships.
Generates:
  - docs/data/full_audio_text_inventory/audio_text_pairs.parquet
  - docs/data/full_audio_text_inventory/MASTER_AUDIO_TEXT_INVENTORY.csv
  - docs/data/full_audio_text_inventory/source_registry.csv
  - docs/data/full_audio_text_inventory/source_summary.csv
  - docs/data/full_audio_text_inventory/source_overlap_matrix.csv
  - docs/data/full_audio_text_inventory/new_data_beyond_e22_01.csv
"""

import os
import sys
import json
import hashlib
import pandas as pd
import numpy as np

def build_registry():
    print("=" * 60)
    print("STEP 4: UNIFIED AUDIOTEXT PAIR REGISTRY & CSV GENERATION")
    print("=" * 60)

    # 1. Load physical audio catalog
    phys_df = pd.read_parquet("src/data_audit/audit_physical_audio.parquet")
    print(f"Loaded physical audio catalog: {len(phys_df)} records.")

    # 2. Load manifest pairs catalog
    man_df = pd.read_parquet("src/data_audit/audit_manifest_pairs.parquet")
    print(f"Loaded manifest pairs catalog: {len(man_df)} records.")

    # 3. Load benchmark sets
    benchmarks = {
        "canon_test": "kashmiri_dataset/manifests/test.jsonl",
        "canon_val": "kashmiri_dataset/manifests/val.jsonl",
        "stage9_test": "kashmiri_dataset/experiments/native_asr_training/data_splits/test.jsonl",
        "stage9_val": "kashmiri_dataset/experiments/native_asr_training/data_splits/val.jsonl",
        "dev_b_waaz_val": "kashmiri_dataset/releases/asr_training_v2_waaz/validation_manifests/dev_b_waaz_val.jsonl"
    }
    test_audio = set()
    val_audio = set()
    bench_speakers = set()

    for bname, bpath in benchmarks.items():
        if os.path.exists(bpath):
            with open(bpath) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        af = os.path.basename(d.get("audio_path", d.get("canonical_audio_path", "")))
                        spk = d.get("speaker_id", "")
                        if af:
                            if "test" in bname:
                                test_audio.add(af)
                            if "val" in bname:
                                val_audio.add(af)
                        if spk and spk not in ("unknown", "unspecified", ""):
                            bench_speakers.add(spk)
                    except:
                        pass
    print(f"Loaded benchmarks: {len(test_audio)} test audio, {len(val_audio)} val audio, {len(bench_speakers)} speakers.")

    # 4. Load training sets
    training_sets = {
        "B01": "kashmiri_dataset/experiments/native_asr_training/data_splits/train.jsonl",
        "B02A": "kashmiri_dataset/experiments/native_asr_training/experiments/stage9_b02_acoustic_comparison/manifests/b02_base_manifest.jsonl",
        "B02B": "kashmiri_dataset/experiments/native_asr_training/experiments/stage9_b02_acoustic_comparison/manifests/b02_quran_augmented_manifest.jsonl",
        "B02C": "kashmiri_dataset/experiments/native_asr_training/experiments/stage9_b02_acoustic_comparison/manifests/b02_all_verified_human_manifest.jsonl",
        "E22_01": "kashmiri_dataset/experiments/stage22_data_expanded_training/manifests/data_22a_conservative_tier_a.jsonl",
        "E22_02": "kashmiri_dataset/experiments/stage22_data_expanded_training/manifests/data_22b_full_waaz.jsonl",
        "E22_03": "kashmiri_dataset/experiments/stage22_data_expanded_training/manifests/data_22c_expanded_multisource.jsonl"
    }
    train_audio_map = {k: set() for k in training_sets}
    for tname, tpath in training_sets.items():
        if os.path.exists(tpath):
            with open(tpath) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        af = os.path.basename(d.get("audio_path", d.get("canonical_audio_path", "")))
                        if af:
                            train_audio_map[tname].add(af)
                    except:
                        pass
        print(f"Loaded training run {tname}: {len(train_audio_map[tname])} audio files.")

    # 5. Build Canonical Registry
    # Create canonical record per unique audio item
    # Deduplicate manifest records to get primary transcript
    # Priority for manifest: asr_training_eligible_manifest > source_manifests > datacollect.db > others
    print("\nIndexing primary text annotations...")
    man_df["priority"] = 3
    man_df.loc[man_df["manifest_name"] == "asr_training_eligible_manifest.jsonl", "priority"] = 1
    man_df.loc[man_df["manifest_name"].str.startswith("source_manifests/"), "priority"] = 2
    man_df.loc[man_df["manifest_name"] == "DataCollection/datacollect.db", "priority"] = 2
    
    man_df_sorted = man_df.sort_values(by=["audio_filename", "priority"])
    man_dedup = man_df_sorted.drop_duplicates(subset=["audio_filename"], keep="first").copy()
    print(f"Unique audio files with transcripts: {len(man_dedup)}")

    # Merge physical audio with primary text
    # Filter physical audio: keep real files, plus embedded/tar
    # For symlinks, we already have their real_path
    phys_unique = phys_df[~phys_df["is_symlink"]].drop_duplicates(subset=["filename"], keep="first").copy()
    print(f"Unique physical audio items: {len(phys_unique)}")

    merged = pd.merge(phys_unique, man_dedup, left_on="filename", right_on="audio_filename", how="left")
    if "duration_seconds_x" in merged.columns and "duration_seconds_y" in merged.columns:
        merged["duration_seconds"] = merged["duration_seconds_x"].fillna(merged["duration_seconds_y"])
        merged.drop(columns=["duration_seconds_x", "duration_seconds_y"], inplace=True)
    elif "duration_seconds_x" in merged.columns:
        merged.rename(columns={"duration_seconds_x": "duration_seconds"}, inplace=True)
    print(f"Merged registry entries: {len(merged)}")

    # Assign comprehensive fields
    merged["has_transcript"] = merged["text"].notna() & (merged["text"].str.strip() != "")
    merged["is_kashmiri_script"] = merged["script"].isin(["kas_Arab", "Arab", "Deva"])
    merged["is_kashmiri_lang"] = merged["language"].isin(["kas", "kas_Latn", "kas_Deva"])
    
    # Check training memberships
    for tname in training_sets:
        merged[f"in_{tname}"] = merged["filename"].isin(train_audio_map[tname])

    # Check benchmark collisions
    merged["in_frozen_test"] = merged["filename"].isin(test_audio)
    merged["in_frozen_val"] = merged["filename"].isin(val_audio)
    merged["speaker_collides_benchmark"] = merged["speaker_id"].isin(bench_speakers)
    merged["is_benchmark_contaminated"] = merged["in_frozen_test"] | merged["in_frozen_val"] | merged["speaker_collides_benchmark"]

    # Final Alignment Category
    # A: Verified utterance-level pair
    # B: Likely correct pair
    # C: Segmentable recording
    # D: Alignable recording
    # E: Weak or autogenerated transcription
    # F: Translation or commentary mismatch (Arabic recitation)
    # G: Audio-only (no text)
    # H: Text-only
    # I: Invalid or corrupt (clipped, silent, error)
    final_cats = []
    elig_cats = []
    
    # Load King Fahd detailed category breakdown
    kf_cats = {}
    kf_manifest = "kashmiri_dataset/manifests/source_manifests/manifest_quran_king_fahd.jsonl"
    if os.path.exists(kf_manifest):
        with open(kf_manifest) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    fn = os.path.basename(d.get("audio_path", ""))
                    kf_cats[fn] = d.get("category", "")
                except:
                    pass

    for _, row in merged.iterrows():
        has_txt = row["has_transcript"]
        sub = str(row["subfolder"]).lower()
        fn = str(row["filename"]).lower()
        src = str(row["source_dataset"]).lower()
        val_stat = str(row["validation_status"])
        txt = str(row["text"]) if has_txt else ""
        dur = row["duration_seconds"] if pd.notna(row["duration_seconds"]) else 0.0

        # Check King Fahd explicit categories
        if row["filename"] in kf_cats:
            kfc = kf_cats[row["filename"]]
            if kfc == "arabic_recitation":
                cat = "F"
                elig = "UNUSABLE_ARABIC_RECITATION"
            elif kfc == "kashmiri_commentary":
                cat = "D"
                elig = "RECOVERABLE_ALIGNABLE_COMMENTARY"
            elif kfc == "uncertain_mixed_speech":
                cat = "I"
                elig = "INVALID_BOUNDARY_LEAK"
            elif kfc == "kashmiri_translation":
                cat = "B"
                elig = "VERIFIED_ELIGIBLE_NEW_TRAINING_DATA"
            else:
                cat = "B"
                elig = "VERIFIED_ELIGIBLE_NEW_TRAINING_DATA"
        elif not has_txt:
            if "arabic_recitation" in sub:
                cat = "F"
                elig = "UNUSABLE_ARABIC_RECITATION"
            elif "dr_war" in sub or "dr_war" in fn:
                cat = "G"
                elig = "RECOVERABLE_AUDIO_ONLY_NEEDS_TRANSCRIPTION"
            elif "quran_kashmiri" in sub or "pure_kashmiri" in sub:
                cat = "D"
                elig = "RECOVERABLE_ALIGNABLE_TAFSIR_COMMENTARY"
            else:
                cat = "G"
                elig = "AUDIO_ONLY"
        else:
            if "arabic_recitation" in sub or row["language"] == "ara" or "recitation" in fn:
                cat = "F"
                elig = "UNUSABLE_ARABIC_RECITATION"
            elif "dr_war" in src:
                cat = "E" # ASR hypothesis
                elig = "WEAK_PSEUDO_LABELED_NEEDS_VERIFICATION"
            elif row["source_dataset"] == "quran_kashmiri":
                cat = "D"
                elig = "RECOVERABLE_ALIGNABLE_TAFSIR_COMMENTARY"
            elif row["in_frozen_test"]:
                cat = "A" if val_stat in ("HUMAN_OR_CANONICAL_VERIFIED", "HUMAN_TRANSCRIBED") else "B"
                elig = "EXCLUDED_FROZEN_TEST_BENCHMARK"
            elif row["in_frozen_val"]:
                cat = "A" if val_stat in ("HUMAN_OR_CANONICAL_VERIFIED", "HUMAN_TRANSCRIBED") else "B"
                elig = "EXCLUDED_FROZEN_VAL_BENCHMARK"
            elif row["speaker_collides_benchmark"]:
                cat = "A" if val_stat in ("HUMAN_OR_CANONICAL_VERIFIED", "HUMAN_TRANSCRIBED") else "B"
                elig = "EXCLUDED_BENCHMARK_SPEAKER_COLLISION"
            elif row["in_E22_01"]:
                cat = "A" if val_stat in ("HUMAN_OR_CANONICAL_VERIFIED", "HUMAN_TRANSCRIBED", "WAAZ_VERIFIED_HUMAN") else "B"
                elig = "ALREADY_USED_IN_E22_01"
            elif val_stat in ("HUMAN_OR_CANONICAL_VERIFIED", "HUMAN_TRANSCRIBED", "SOURCE_CANONICAL"):
                cat = "A"
                elig = "VERIFIED_ELIGIBLE_NEW_TRAINING_DATA"
            elif "waaz" in src and val_stat == "WAAZ_COMMUNITY_PROMPT":
                cat = "B"
                elig = "COMMUNITY_PROMPT_NEEDS_VALIDATION"
            elif "uchicago" in src:
                cat = "B"
                elig = "DICTIONARY_LEXICON_ISOLATED_WORD"
            else:
                cat = "B"
                elig = "VERIFIED_ELIGIBLE_NEW_TRAINING_DATA"

        final_cats.append(cat)
        elig_cats.append(elig)

    merged["final_alignment_category"] = final_cats
    merged["final_eligibility_category"] = elig_cats

    # Save audio_text_pairs.parquet
    out_pq = "docs/data/full_audio_text_inventory/audio_text_pairs.parquet"
    merged.to_parquet(out_pq, index=False)
    print(f"\nSaved canonical audio_text_pairs.parquet: {len(merged)} records.")

    # 6. Generate Deliverable CSVs
    # 6a. MASTER_AUDIO_TEXT_INVENTORY.csv
    # Sample or aggregated summary per category & source for readability
    print("\nGenerating MASTER_AUDIO_TEXT_INVENTORY.csv...")
    master_summary = merged.groupby(["location_group", "source_dataset", "final_alignment_category", "final_eligibility_category"]).agg(
        total_records=("filename", "count"),
        total_hours=("duration_seconds", lambda s: round(s.sum() / 3600.0, 3)),
        avg_duration_sec=("duration_seconds", lambda s: round(s.mean(), 2)),
        in_e22_01_count=("in_E22_01", "sum"),
        test_leak_count=("in_frozen_test", "sum"),
        val_leak_count=("in_frozen_val", "sum")
    ).reset_index()
    master_summary.to_csv("docs/data/full_audio_text_inventory/MASTER_AUDIO_TEXT_INVENTORY.csv", index=False)

    # 6b. source_registry.csv
    print("Generating source_registry.csv...")
    source_reg = []
    sources_info = [
        ("SRC_EXT_BOLBOSH", "Bolbosh Kashmiri Speech Corpus", "GAASH-Lab (IUST)", "Multi-speaker read & conversation", "CC-BY 4.0", "kashmiri_dataset/processed/audio/bolbosh/"),
        ("SRC_EXT_INDICVOICES_R", "IndicVoices-R Kashmiri Expansion", "AI4Bharat / IIT Madras", "Spontaneous native dialogue", "CC-BY 4.0", "kashmiri_dataset/processed/audio/indicvoices_r/"),
        ("SRC_EXT_INDICVOICES", "IndicVoices Kashmiri Primary", "AI4Bharat / IIT Madras", "Conversational field recordings", "CC-BY 4.0", "kashmiri_dataset/processed/audio/indicvoices_kashmiri/"),
        ("SRC_QURAN_MIRWAIZ", "Al-Quran Kashmiri Translation (Mirwaiz)", "Mirwaiz Research Foundation", "Spoken religious translation", "Educational / Research", "kashmiri_dataset/processed/audio/quran_kashmiri/"),
        ("SRC_QURAN_DR_WAR", "Al-Quran Kashmiri Commentary (Dr. War)", "Gulistan News Ingestion", "Spoken religious commentary", "Educational / Research", "kashmiri_dataset/processed/audio/quran_dr_war/"),
        ("SRC_QURAN_KING_FAHD", "Al-Quran Kashmiri Ingestion (King Fahd)", "King Fahd Holy Quran Complex", "Studio religious recitation & commentary", "Educational / Research", "kashmiri_dataset/processed/audio/quran_king_fahd/"),
        ("SRC_COMM_WAAZ", "WAAZ Community Crowdsourced Platform", "EduCasheer WAAZ Community", "Crowdsourced native read & spontaneous", "CC-BY-NC 4.0", "DataCollection/data/raw/contributions/"),
        ("SRC_RAW_KTTS", "Kashmiri Text-to-Speech Corpus (KTTS)", "Mendeley / Programindz", "Single-speaker read studio speech", "CC-BY 4.0", "kashmiri_dataset/processed/audio/ktts/"),
        ("SRC_EXT_OPENSLR122", "OpenSLR-122 Kashmiri Speech Corpus", "OpenSLR / LRAC", "Multi-speaker reading & dialogue", "GPL-3.0", "kashmiri_dataset/processed/audio/openslr122/"),
        ("SRC_EXT_XKAAB", "XKaab Kashmiri ASR 4-Hour Corpus", "XKaab Research", "Studio & conversation speech", "CC-BY 4.0", "kashmiri_dataset/processed/audio/xkaab_4hrs/"),
        ("SRC_EXT_AUDIO_CORPUS", "Programindz Spoken Audio Corpus", "Programindz / OpenSLR-122 Slices", "Multi-speaker short segments", "GPL-3.0", "kashmiri_dataset/processed/audio/audio_corpus/"),
        ("SRC_EXT_META_OMNI", "Meta Omnilingual Kashmiri ASR Corpus", "Meta AI Research (FAIR)", "Conversational prompted monologue", "CC-BY 4.0", "kashmiri_dataset/processed/audio/meta_omnilingual/"),
        ("SRC_EXT_WHISPER_1", "Whisper Kashmiri Community Slice 1", "Community Ingestion", "Near-field read speech", "MIT", "kashmiri_dataset/processed/audio/whisper_kashmiri/"),
        ("SRC_EXT_WHISPER_2", "Whisper Kashmiri Fine-tune Part 2", "Muneeb Haroon / HuggingFace", "Community read speech", "Apache 2.0", "kashmiri_dataset/processed/audio/whisper_kashmiri_2/"),
        ("SRC_EXT_UCHICAGO_DICT", "UChicago S. Hassan Kashmiri Dictionary", "University of Chicago / S. Hassan", "Lexicon single-word pronunciations", "Educational / Research", "kashmiri_dataset/raw/external/mzmmoazam_dataset/"),
        ("SRC_BENCH_SARVAM", "Sarvam Indic DiarBench (Kashmiri)", "Sarvam AI", "Multi-speaker diarization benchmark", "CC-BY-SA 4.0", "kashmiri_dataset/processed/audio/sarvam_diarbench/")
    ]
    for sid, sname, org, dom, lic, loc in sources_info:
        # compute matched counts
        src_mask = merged["source_dataset"].str.contains(sname.split()[0], case=False, na=False) | merged["rel_path"].str.contains(loc.split('/')[-2], case=False, na=False)
        sub_df = merged[src_mask]
        source_reg.append({
            "source_id": sid,
            "dataset_name": sname,
            "provider": org,
            "modality": dom,
            "license": lic,
            "local_path": loc,
            "total_physical_clips": len(sub_df),
            "total_hours": round(sub_df["duration_seconds"].sum() / 3600.0, 2),
            "status": "Physically Available" if len(sub_df) > 0 else "External / Cataloged"
        })
    df_src_reg = pd.DataFrame(source_reg)
    df_src_reg.to_csv("docs/data/full_audio_text_inventory/source_registry.csv", index=False)

    # 6c. source_summary.csv
    print("Generating source_summary.csv...")
    src_sum = merged.groupby("source_dataset").agg(
        total_clips=("filename", "count"),
        total_hours=("duration_seconds", lambda s: round(s.sum() / 3600.0, 3)),
        with_transcript_clips=("has_transcript", "sum"),
        with_transcript_hours=("duration_seconds", lambda s: round(merged.loc[s.index, "duration_seconds"][merged.loc[s.index, "has_transcript"]].sum() / 3600.0, 3)),
        in_e22_01_clips=("in_E22_01", "sum"),
        in_e22_01_hours=("duration_seconds", lambda s: round(merged.loc[s.index, "duration_seconds"][merged.loc[s.index, "in_E22_01"]].sum() / 3600.0, 3)),
        benchmark_contaminated_clips=("is_benchmark_contaminated", "sum"),
        benchmark_contaminated_hours=("duration_seconds", lambda s: round(merged.loc[s.index, "duration_seconds"][merged.loc[s.index, "is_benchmark_contaminated"]].sum() / 3600.0, 3))
    ).reset_index()
    src_sum.to_csv("docs/data/full_audio_text_inventory/source_summary.csv", index=False)

    # 6d. source_overlap_matrix.csv
    print("Generating source_overlap_matrix.csv...")
    # Overlap between each dataset and training / benchmarks
    overlap_rows = []
    top_sources = merged["source_dataset"].dropna().unique()
    for s in top_sources:
        s_df = merged[merged["source_dataset"] == s]
        overlap_rows.append({
            "source_dataset": s,
            "total_clips": len(s_df),
            "total_hours": round(s_df["duration_seconds"].sum() / 3600.0, 2),
            "in_E22_01": len(s_df[s_df["in_E22_01"]]),
            "in_E22_03": len(s_df[s_df["in_E22_03"]]),
            "in_test_benchmark": len(s_df[s_df["in_frozen_test"]]),
            "in_val_benchmark": len(s_df[s_df["in_frozen_val"]]),
            "spk_benchmark_overlap": len(s_df[s_df["speaker_collides_benchmark"]])
        })
    df_overlap = pd.DataFrame(overlap_rows)
    df_overlap.to_csv("docs/data/full_audio_text_inventory/source_overlap_matrix.csv", index=False)

    # 6e. new_data_beyond_e22_01.csv
    print("Generating new_data_beyond_e22_01.csv...")
    # Filter for all verified or recoverable records NOT in E22-01 and NOT contaminated
    new_data = merged[~merged["in_E22_01"] & ~merged["in_frozen_test"] & ~merged["in_frozen_val"]].copy()
    new_summary = new_data.groupby(["source_dataset", "final_alignment_category", "final_eligibility_category"]).agg(
        candidate_clips=("filename", "count"),
        candidate_hours=("duration_seconds", lambda s: round(s.sum() / 3600.0, 3)),
        avg_duration_sec=("duration_seconds", lambda s: round(s.mean(), 2)),
        verified_text_count=("has_transcript", "sum")
    ).reset_index()
    new_summary.to_csv("docs/data/full_audio_text_inventory/new_data_beyond_e22_01.csv", index=False)

    print("\n" + "=" * 60)
    print("ALL CANONICAL INVENTORY CSVs SUCCESSFULLY PRODUCED!")
    print("=" * 60)

if __name__ == "__main__":
    build_registry()
