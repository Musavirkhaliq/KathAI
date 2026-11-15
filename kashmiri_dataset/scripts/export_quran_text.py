#!/usr/bin/env python3
"""
Export Complete Kashmiri Quran Translations
===========================================
Source: github.com/izan-majeed/kaeshur-quran-data
Exports:
- Translator 300: Athar Manigami (Perso-Arabic Kashmiri)
- Translator 301: Mirwaiz M. Yousuf Shah (Perso-Arabic Kashmiri)
- Translator 400: Athar Manigami (Roman transliteration)
- Translator 401: Mirwaiz M. Yousuf Shah (Roman transliteration)

Preserves:
- Surah number (1-114)
- Ayah number (1-N)
- Global verse ID (1-6236)
- Verse key (surah:ayah)
- Original verse text
- Normalized verse text
- Arabic reference verse text
- Translator ID & Name
- Script identification
- Source provenance
"""

import os
import sys
import json
import glob

# Add existing scripts directory to path to reuse normalize_kashmiri_text
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
from normalize_transcripts import normalize_kashmiri_text

REPO_DIR = "kashmiri_dataset/raw/external/quran_kashmiri/kaeshur-quran-data"
OUTPUT_DIR = "kashmiri_dataset/raw/external/quran_kashmiri/raw_text"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRANSLATORS = [
    {
        "id": 301,
        "name": "Mirwaiz Mohammad Yousuf Shah",
        "script": "perso_arabic_kashmiri",
        "is_translit": False,
        "filename": "quran_text_mirwaiz_301.jsonl"
    },
    {
        "id": 300,
        "name": "Athar Manigami",
        "script": "perso_arabic_kashmiri",
        "is_translit": False,
        "filename": "quran_text_athar_300.jsonl"
    },
    {
        "id": 401,
        "name": "Mirwaiz Mohammad Yousuf Shah (Transliteration)",
        "script": "latin_transliteration",
        "is_translit": True,
        "filename": "quran_text_mirwaiz_translit_401.jsonl"
    },
    {
        "id": 400,
        "name": "Athar Manigami (Transliteration)",
        "script": "latin_transliteration",
        "is_translit": True,
        "filename": "quran_text_athar_translit_400.jsonl"
    }
]

def load_arabic_verses():
    arabic_map = {}
    verses_dir = os.path.join(REPO_DIR, "kaeshur_quran/verses")
    for f in sorted(glob.glob(os.path.join(verses_dir, "*.json"))):
        s_num = int(os.path.basename(f).split(".")[0])
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            for v in data:
                ayah = v["verse_id"]
                arabic_map[(s_num, ayah)] = {
                    "arabic_text": v["verse_content"],
                    "verse_key": v["verse_key"]
                }
    return arabic_map

def main():
    arabic_map = load_arabic_verses()
    print(f"Loaded {len(arabic_map)} reference Arabic verses.")

    stats = {}

    for tinfo in TRANSLATORS:
        tid = tinfo["id"]
        tname = tinfo["name"]
        script = tinfo["script"]
        is_translit = tinfo["is_translit"]
        out_path = os.path.join(OUTPUT_DIR, tinfo["filename"])

        tdir = os.path.join(REPO_DIR, "kaeshur_quran/translations", str(tid))
        surah_files = sorted(glob.glob(os.path.join(tdir, "*.json")))
        
        records = []
        global_verse_id = 1

        for sf in surah_files:
            s_num = int(os.path.basename(sf).split(".")[0])
            with open(sf, "r", encoding="utf-8") as fp:
                verses = json.load(fp)
            
            for v in verses:
                ayah = v["verse_id"]
                raw_text = v["translation"]
                vkey = f"{s_num}:{ayah}"
                
                ar_info = arabic_map.get((s_num, ayah), {})
                ar_text = ar_info.get("arabic_text", "")

                norm_text = normalize_kashmiri_text(raw_text) if not is_translit else raw_text.strip()

                rec = {
                    "surah_number": s_num,
                    "ayah_number": ayah,
                    "global_verse_id": global_verse_id,
                    "verse_key": vkey,
                    "translator_id": tid,
                    "translator_name": tname,
                    "script": script,
                    "is_transliteration": is_translit,
                    "original_verse_text": raw_text,
                    "normalized_verse_text": norm_text,
                    "arabic_verse_text": ar_text,
                    "source_provenance": "kaeshur-quran-data (github.com/izan-majeed/kaeshur-quran-data)"
                }
                records.append(rec)
                global_verse_id += 1

        with open(out_path, "w", encoding="utf-8") as out_fp:
            for r in records:
                out_fp.write(json.dumps(r, ensure_ascii=False) + "\n")

        stats[tid] = {
            "name": tname,
            "filename": tinfo["filename"],
            "total_verses": len(records),
            "surahs_covered": len(surah_files),
            "output_path": out_path
        }
        print(f"Exported {len(records)} verses for {tname} -> {out_path}")

    summary_file = os.path.join(OUTPUT_DIR, "export_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\nText Export Verification:")
    for tid, info in stats.items():
        print(f"  Translator {tid} ({info['name']}): {info['total_verses']} verses across {info['surahs_covered']} surahs.")

if __name__ == "__main__":
    main()
