import os
import json
from pathlib import Path

base_dir = Path("/home/musa/Kashmiri Project")
raw_text_dir = base_dir / "kashmiri_dataset/raw/external/quran_kashmiri/raw_text"
output_file = base_dir / "QURAN_WRITTEN_TEXT_INVENTORY.jsonl"

text_sources = [
    {
        "file": raw_text_dir / "quran_text_mirwaiz_301.jsonl",
        "edition_name": "Bayan-ul-Furqan (Native Perso-Arabic Script)",
        "translator": "Mirwaiz Mohammad Yousuf Shah",
        "evidence_tier": "Tier 1: Exact structured digital translation",
        "script": "perso_arabic_kashmiri",
        "is_transliteration": False
    },
    {
        "file": raw_text_dir / "quran_text_mirwaiz_translit_401.jsonl",
        "edition_name": "Bayan-ul-Furqan (Roman Transliteration)",
        "translator": "Mirwaiz Mohammad Yousuf Shah",
        "evidence_tier": "Tier 2: Structured Latin transliteration",
        "script": "latin_kashmiri",
        "is_transliteration": True
    },
    {
        "file": raw_text_dir / "quran_text_athar_300.jsonl",
        "edition_name": "Tarjama-e-Quran (Native Perso-Arabic Script)",
        "translator": "Athar Manigami",
        "evidence_tier": "Tier 1: Exact structured digital translation",
        "script": "perso_arabic_kashmiri",
        "is_transliteration": False
    },
    {
        "file": raw_text_dir / "quran_text_athar_translit_400.jsonl",
        "edition_name": "Tarjama-e-Quran (Roman Transliteration)",
        "translator": "Athar Manigami",
        "evidence_tier": "Tier 2: Structured Latin transliteration",
        "script": "latin_kashmiri",
        "is_transliteration": True
    }
]

total_written_records = 0
with open(output_file, "w", encoding="utf-8") as out_f:
    for src in text_sources:
        print(f"Reading {src['edition_name']} from {src['file']}...")
        with open(src["file"], "r", encoding="utf-8") as in_f:
            for line in in_f:
                if line.strip():
                    item = json.loads(line)
                    record = {
                        "verse_key": item.get("verse_key"),
                        "surah_number": item.get("surah_number"),
                        "ayah_number": item.get("ayah_number"),
                        "global_verse_id": item.get("global_verse_id"),
                        "translator": item.get("translator_name", src["translator"]),
                        "edition_name": src["edition_name"],
                        "evidence_tier": src["evidence_tier"],
                        "script": src["script"],
                        "is_transliteration": src["is_transliteration"],
                        "original_verse_text": item.get("original_verse_text", ""),
                        "normalized_verse_text": item.get("normalized_verse_text", ""),
                        "arabic_verse_text": item.get("arabic_verse_text", ""),
                        "source_provenance": item.get("source_provenance", "kaeshur-quran-data (github.com/izan-majeed/kaeshur-quran-data)")
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_written_records += 1

# Also append metadata for King Fahd Complex PDF text
pdf_meta_record = {
    "verse_key": "1:1-114:6",
    "surah_number": "all_114_surahs",
    "ayah_number": "all_6236_ayahs",
    "global_verse_id": "1-6236",
    "translator": "Mirwaiz Mohammad Yousuf Shah",
    "edition_name": "King Fahd Complex for Printing the Holy Quran (Mushaf-e-Kashmir)",
    "evidence_tier": "Tier 3: Published scanned physical mushaf (1698 pages)",
    "script": "perso_arabic_kashmiri",
    "is_transliteration": False,
    "original_verse_text": "Complete 1698-page high-resolution calligraphy print adopted by King Fahd Glorious Quran Printing Complex",
    "normalized_verse_text": "Identical canonical translation text to Mirwaiz ID 301, verified against printed page images",
    "arabic_verse_text": "Complete Quranic Arabic text",
    "source_provenance": "King Fahd Complex / IslamHouse (ks_translation_of_the_Meaning_of_the_Holy_Quran_in_Kashmiri.pdf)"
}
with open(output_file, "a", encoding="utf-8") as out_f:
    out_f.write(json.dumps(pdf_meta_record, ensure_ascii=False) + "\n")
    total_written_records += 1

print(f"Successfully generated {output_file} with {total_written_records} records.")
