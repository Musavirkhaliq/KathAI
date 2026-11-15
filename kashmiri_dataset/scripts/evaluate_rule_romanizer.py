import json
import re
from kashmiri_rule_romanizer import KashmiriRuleRomanizer
from metrics_evaluator import levenshtein_distance

test_path = "/home/musa/Kashmiri Project/kashmiri_dataset/experiments/romanization_benchmark/test.jsonl"
romanizer = KashmiriRuleRomanizer()

total_word_dist = 0
total_ref_words = 0
total_char_dist = 0
total_ref_chars = 0
exact_matches = 0
total_samples = 0

predictions = []

with open(test_path, "r", encoding="utf-8") as f:
    for line in f:
        it = json.loads(line)
        src = it["arabic"]
        ref = it["roman_ascii"].strip().lower()
        hyp = romanizer.romanize(src).strip().lower()

        # Clean punctuation for fair normalized comparison
        ref_clean = re.sub(r'[^\w\s]', '', ref)
        hyp_clean = re.sub(r'[^\w\s]', '', hyp)

        r_words = ref_clean.split()
        h_words = hyp_clean.split()

        w_dist = levenshtein_distance(r_words, h_words)
        total_word_dist += w_dist
        total_ref_words += len(r_words)

        r_chars = list(ref_clean.replace(" ", ""))
        h_chars = list(hyp_clean.replace(" ", ""))

        c_dist = levenshtein_distance(r_chars, h_chars)
        total_char_dist += c_dist
        total_ref_chars += len(r_chars)

        if ref_clean == hyp_clean:
            exact_matches += 1

        total_samples += 1
        if len(predictions) < 5:
            predictions.append({"src": src, "ref": ref_clean, "hyp": hyp_clean})

wer = (total_word_dist / max(total_ref_words, 1)) * 100
cer = (total_char_dist / max(total_ref_chars, 1)) * 100
em = (exact_matches / max(total_samples, 1)) * 100

print(f"=== Rule-Based & Lexicon Romanizer Evaluation on Held-Out Test Set ({total_samples} samples) ===")
print(f"  Word Error Rate (WER)     : {wer:.2f}%")
print(f"  Character Error Rate (CER): {cer:.2f}%")
print(f"  Exact Match Accuracy      : {em:.2f}%")
print("\nSample Comparisons:")
for p in predictions:
    print(f"  Arabic: {p['src']}")
    print(f"  Ref   : {p['ref']}")
    print(f"  Hyp   : {p['hyp']}")
    print("  ---")

results = {
    "model": "KashmiriRuleRomanizer",
    "samples": total_samples,
    "wer": round(wer, 2),
    "cer": round(cer, 2),
    "exact_match_pct": round(em, 2)
}
with open("/home/musa/Kashmiri Project/kashmiri_dataset/experiments/romanization_benchmark/rule_eval_results.json", "w") as f:
    json.dump(results, f, indent=2)
