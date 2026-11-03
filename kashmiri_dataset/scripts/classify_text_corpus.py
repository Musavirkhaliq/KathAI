import json
import unicodedata
from collections import Counter
from pathlib import Path

BASE_DIR = Path("/home/musa/Kashmiri Project")
TEXT_JSONL = BASE_DIR / "kashmiri_dataset" / "merged" / "KASHMIRI-TEXT" / "kashmiri_text_corpus.jsonl"
WIKI_JSONL = BASE_DIR / "kashmiri_dataset" / "processed" / "text" / "wikipedia" / "kashmiri_wikipedia_cleaned.jsonl"
REP_DIR = BASE_DIR / "kashmiri_dataset" / "quality_reports"

print("Auditing and Classifying Text Corpus into Distinct Research Taxonomies...")

categories = Counter()
scripts_by_cat = {}
doc_counts = Counter()
word_counts = Counter()
char_counts = Counter()
vocabularies = {}

def get_script(text):
    c = Counter()
    for ch in text[:500]: # sample first 500 chars
        name = unicodedata.name(ch, "")
        if "ARABIC" in name:
            c["Perso-Arabic"] += 1
        elif "DEVANAGARI" in name:
            c["Devanagari"] += 1
        elif "LATIN" in name:
            c["Latin"] += 1
    if not c:
        return "Unknown"
    return c.most_common(1)[0][0]

# 1. Process existing 4.12M records in KASHMIRI-TEXT
with open(TEXT_JSONL, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if idx % 500000 == 0 and idx > 0:
            print(f"  Processed {idx:,} text records...")
        item = json.loads(line)
        src = item.get("source", "")
        item_type = item.get("type", "")
        
        # Determine classification taxonomy
        if "clinical" in src.lower() or "clinical" in item_type.lower():
            cat = "Clinical & Medical Reasoning"
        elif "math" in src.lower() or "math" in item_type.lower():
            cat = "Mathematical & Scientific Reasoning"
        elif "aya" in src.lower() or "instruction" in item_type.lower():
            cat = "Instruction-Response Tuning (Aya)"
        elif "wikipedia" in src.lower():
            cat = "Native Kashmiri Wikipedia"
        else:
            cat = "General Kashmiri Text"
            
        doc_counts[cat] += 1
        
        # sample text length
        txt = ""
        if "instruction" in item and "response" in item:
            txt = f"{item['instruction']} {item['response']}"
        elif "content" in item and isinstance(item["content"], dict):
            txt = " ".join(str(v) for v in item["content"].values())
        elif "text" in item:
            txt = str(item["text"])
            
        w_cnt = len(txt.split())
        c_cnt = len(txt)
        word_counts[cat] += w_cnt
        char_counts[cat] += c_cnt
        
        if cat not in scripts_by_cat:
            scripts_by_cat[cat] = Counter()
        if idx % 100 == 0 and txt:
            sc = get_script(txt)
            scripts_by_cat[cat][sc] += 1

# 2. Add Wikipedia
with open(WIKI_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        cat = "Native Kashmiri Wikipedia"
        doc_counts[cat] += 1
        txt = item.get("text", "")
        word_counts[cat] += len(txt.split())
        char_counts[cat] += len(txt)
        if cat not in scripts_by_cat:
            scripts_by_cat[cat] = Counter()
        sc = get_script(txt)
        scripts_by_cat[cat][sc] += 1

classification_report = {
    "total_classified_documents": sum(doc_counts.values()),
    "categories": {}
}

for cat in doc_counts:
    classification_report["categories"][cat] = {
        "documents": doc_counts[cat],
        "estimated_words": word_counts[cat],
        "estimated_characters": char_counts[cat],
        "primary_scripts": dict(scripts_by_cat.get(cat, {})),
        "source_type": "Synthetic / Model-Generated" if "Reasoning" in cat else ("Crowdsourced / Human Annotated" if "Aya" in cat else "Organic Native Corpus")
    }

out_report = REP_DIR / "text_corpus_classification_audit.json"
with open(out_report, "w", encoding="utf-8") as f:
    json.dump(classification_report, f, indent=2)

print("\n=== Text Classification Audit Complete ===")
for cat, data in classification_report["categories"].items():
    print(f"[{cat}]")
    print(f"  Documents: {data['documents']:,} | Words: {data['estimated_words']:,} | Type: {data['source_type']}")
print(f"\nReport written to: {out_report}")
