import json
import unicodedata
from pathlib import Path

BASE_DIR = Path("/home/musa/Kashmiri Project")
TRAIN_MANIFEST = BASE_DIR / "kashmiri_dataset" / "experiments" / "native_asr_training" / "data_splits" / "train.jsonl"
OUT_DIR = BASE_DIR / "kashmiri_dataset" / "experiments" / "native_asr_training" / "training_configurations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

chars = set()

with open(TRAIN_MANIFEST, "r", encoding="utf-8") as f:
    for line in f:
        it = json.loads(line)
        txt = it.get("text", "")
        # Normalize
        txt = unicodedata.normalize("NFKC", txt)
        for ch in txt:
            if not ch.isspace():
                chars.add(ch)

# Standard ASR CTC special tokens
vocab = {
    "[PAD]": 0,
    "[UNK]": 1,
    "|": 2  # word boundary delimiter
}

idx = 3
for ch in sorted(list(chars)):
    vocab[ch] = idx
    idx += 1

print(f"Generated Kashmiri CTC Character Vocabulary: {len(vocab)} tokens.")

vocab_file = OUT_DIR / "kashmiri_ctc_vocab.json"
with open(vocab_file, "w", encoding="utf-8") as f:
    json.dump(vocab, f, indent=2, ensure_ascii=False)

print(f"Saved vocabulary to: {vocab_file}")
