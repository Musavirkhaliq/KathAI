import json
import random
from pathlib import Path

random.seed(42)

BASE_DIR = Path("/home/musa/Kashmiri Project")
RASA_FILE = BASE_DIR / "kashmiri_dataset/external/krish187653_rasa-kashmiri-codemixing-transliterations/data/all_rows/train.jsonl"
OUT_DIR = BASE_DIR / "kashmiri_dataset/experiments/romanization_evaluation/data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

pairs = []
with open(RASA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        it = json.loads(line)
        src = (it.get("text") or "").strip()
        tgt_ascii = (it.get("full_transliteration") or "").strip()
        tgt_diacr = (it.get("full_transliteration_with_diacritics") or "").strip()
        if src and tgt_ascii:
            pairs.append({
                "source": src,
                "target_ascii": tgt_ascii,
                "target_phonetic": tgt_diacr if tgt_diacr else tgt_ascii
            })

random.shuffle(pairs)
total = len(pairs)
train_end = int(total * 0.85)
val_end = int(total * 0.95)

train_pairs = pairs[:train_end]
val_pairs = pairs[train_end:val_end]
test_pairs = pairs[val_end:]

print(f"Total valid pairs: {total}")
print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")

for name, data in [("train.jsonl", train_pairs), ("val.jsonl", val_pairs), ("test.jsonl", test_pairs)]:
    with open(OUT_DIR / name, "w", encoding="utf-8") as out:
        for p in data:
            out.write(json.dumps(p, ensure_ascii=False) + "\n")

print("Saved splits to:", OUT_DIR)
