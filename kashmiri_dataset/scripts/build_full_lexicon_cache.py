import json
from pathlib import Path

rasa_file = Path("/home/musa/Kashmiri Project/kashmiri_dataset/external/krish187653_rasa-kashmiri-codemixing-transliterations/data/all_rows/train.jsonl")
output_file = Path("/home/musa/Kashmiri Project/kashmiri_dataset/scripts/kashmiri_lexicon_cache.json")

lexicon = {}
with open(rasa_file, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if idx >= 20000: break
        try:
            item = json.loads(line)
        except Exception:
            continue
        ws = item.get("word_sequence") or []
        for w in ws:
            src = w.get("source_span")
            asc = w.get("ascii")
            ph = w.get("phonetic")
            if src and asc and ph and src not in lexicon:
                lexicon[src] = {
                    "ascii": asc.lower(),
                    "phonetic": ph,
                    "lang": w.get("language", "Kashmiri")
                }

print(f"Extracted {len(lexicon):,} verified Kashmiri word pairs from Rasa corpus.")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(lexicon, f, ensure_ascii=False)
print(f"Saved lexicon cache to: {output_file}")
