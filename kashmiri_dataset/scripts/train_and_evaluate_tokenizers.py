import os
import json
import sentencepiece as spm
from pathlib import Path
from transformers import AutoTokenizer

BASE_DIR = Path("/home/musa/Kashmiri Project")
CORPUS_TXT = BASE_DIR / "kashmiri_dataset" / "processed" / "text" / "wikipedia" / "kashmiri_wikipedia_plain.txt"
TOK_DIR = BASE_DIR / "kashmiri_dataset" / "tokenizers"
REP_DIR = BASE_DIR / "kashmiri_dataset" / "quality_reports"

VOCAB_SIZES = [8000, 16000, 24000, 32000]

print("=== Starting Workstream 3: Kashmiri SentencePiece Tokenizer Training ===")

# 1. Train SentencePiece models
sp_models = {}
for vs in VOCAB_SIZES:
    prefix = TOK_DIR / f"spm_{vs//1000}k" / f"kashmiri_spm_{vs//1000}k"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    model_path = f"{prefix}.model"
    
    if not os.path.exists(model_path):
        print(f"\nTraining SentencePiece model with vocab size: {vs}...")
        spm.SentencePieceTrainer.train(
            input=str(CORPUS_TXT),
            model_prefix=str(prefix),
            vocab_size=vs,
            model_type="bpe",
            character_coverage=0.9995,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            user_defined_symbols=["ٲ", "ؠ", "ۆ", "ۄ", "ێ", "ۍ", "ؽ"]
        )
        print(f"  Saved model: {model_path}")
    else:
        print(f"  SentencePiece model {model_path} already exists.")
        
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    sp_models[f"Kashmiri-BPE-{vs//1000}k"] = sp

# 2. Test Corpus for Tokenizer Evaluation
test_sentences = [
    # Organic native Perso-Arabic Kashmiri
    "مےٚ دِیوٗت اکھ ہفتہٕ سمندری جہٲزی سفرس پیٹھ، تہٕ بتھس پیٹھ سمندری ہوا محسوس کرنہٕ کھوتہٕ زیادٕ آرام دہ اوس نہٕ بیٚیہ کہیٖنہٕ۔",
    "کٲشُر زَبان چھِ ہِندوستٲنۍ جۆم تہٕ کٔشیٖر ہٕنٛز سارِوٕے کھۄتہٕ بٔڑ مقٲمی زَبان۔",
    "اَتھ ہَسا چھِ یِوان ترٛیٚیہِ وٕہٕرۍ بعد کَم سے کَم پانٛژھ شےٚ لَچھ رۄپیہِ ؤرِیَس منٛز۔",
    # Code-mixed Kashmiri
    "ڈاکٹر صٲبن دِژ دوایہِ تہٕ وۆنُن کہِ ہسپتالَس منٛز گژھِ ٹیسٹ کراوُن۔",
    # Devanagari Kashmiri
    "कॉशुर ज़बान छि कऺशीर हुन्द प्रान इतिहास बयान करान।"
]

print("\n=== Evaluating Tokenizers against Baseline Foundation LLMs ===")

# We evaluate:
# 1. Tokens per sentence
# 2. Token fertility (Tokens / Word)
# 3. Characters per token

results = {}

for name, sp in sp_models.items():
    total_words = 0
    total_tokens = 0
    total_chars = 0
    for s in test_sentences:
        words = s.split()
        toks = sp.encode_as_pieces(s)
        total_words += len(words)
        total_tokens += len(toks)
        total_chars += len(s)
        
    fertility = round(total_tokens / total_words, 2)
    chars_per_tok = round(total_chars / total_tokens, 2)
    results[name] = {
        "vocab_size": sp.get_piece_size(),
        "total_tokens_on_eval": total_tokens,
        "token_fertility": fertility,
        "chars_per_token": chars_per_tok,
        "sample_tokenization": sp.encode_as_pieces(test_sentences[1])
    }

# Compare against multilingual baseline (e.g. HuggingFace GPT-2 or XLM-R / IndicBERT if available)
try:
    print("Testing against standard multilingual baseline (xlm-roberta-base / indicbert)...")
    from transformers import AutoTokenizer
    # We will test using bert-base-multilingual-cased or xlm-roberta-base
    base_tok = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    total_words = 0
    total_tokens = 0
    total_chars = 0
    for s in test_sentences:
        words = s.split()
        toks = base_tok.tokenize(s)
        total_words += len(words)
        total_tokens += len(toks)
        total_chars += len(s)
    results["mBERT-Multilingual"] = {
        "vocab_size": base_tok.vocab_size,
        "total_tokens_on_eval": total_tokens,
        "token_fertility": round(total_tokens / total_words, 2),
        "chars_per_token": round(total_chars / total_tokens, 2),
        "sample_tokenization": base_tok.tokenize(test_sentences[1])[:10]
    }
except Exception as e:
    print(f"Could not load mBERT baseline: {e}")

out_eval = REP_DIR / "kashmiri_tokenizer_benchmark_results.json"
with open(out_eval, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n=== Tokenizer Benchmark Results ===")
for name, data in results.items():
    print(f"{name:25s} | Vocab: {data['vocab_size']:6d} | Fertility: {data['token_fertility']:4.2f} tokens/word | {data['chars_per_token']:4.2f} chars/token")

print(f"\nSaved detailed evaluation to: {out_eval}")
