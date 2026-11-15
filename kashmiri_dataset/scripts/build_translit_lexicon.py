import json
from pathlib import Path
from collections import Counter

all_rows = "/home/musa/Kashmiri Project/kashmiri_dataset/external/krish187653_rasa-kashmiri-codemixing-transliterations/data/all_rows/train.jsonl"
word_pairs = Counter()

with open(all_rows, "r", encoding="utf-8") as f:
    for line in f:
        it = json.loads(line)
        ws = it.get("word_sequence") or []
        for w in ws:
            src = (w.get("source_span") or "").strip()
            asc = (w.get("ascii") or "").strip().lower()
            if src and asc:
                word_pairs[(src, asc)] += 1

lexicon = {}
counts_by_src = {}
for (src, asc), cnt in word_pairs.items():
    if src not in counts_by_src or cnt > counts_by_src[src]:
        counts_by_src[src] = cnt
        lexicon[src] = asc

out_file = "/home/musa/Kashmiri Project/kashmiri_dataset/experiments/romanization_benchmark/lexicon.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(lexicon, f, ensure_ascii=False, indent=2)

print(f"Built Transliteration Lexicon with {len(lexicon):,} verified words!")
