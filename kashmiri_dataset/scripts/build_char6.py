import math
from pathlib import Path
from collections import defaultdict, Counter
import kenlm

CORPORA_DIR = Path("kashmiri_dataset/experiments/native_asr_training/experiments/stage7_ctc_lm_decoding/lm_corpora")
LM_DIR = Path("kashmiri_dataset/experiments/native_asr_training/experiments/stage7_ctc_lm_decoding/language_models")

with open(CORPORA_DIR / "corpus_domain_d_balanced.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for idx, line in enumerate(f) if line.strip() and idx < 30000]

order = 6
discount = 0.75
tokenized_corpus = [[c for c in line if c != " "] for line in lines if line]
vocab_counter = Counter(c for tokens in tokenized_corpus for c in tokens)
vocab = {c for c, cnt in vocab_counter.items() if cnt >= 5} | {"<s>", "</s>", "<unk>"}

ngram_counts = [defaultdict(int) for _ in range(order)]
context_counts = [defaultdict(int) for _ in range(order)]

for tokens in tokenized_corpus:
    padded = ["<s>"] + [w if w in vocab else "<unk>" for w in tokens] + ["</s>"]
    for n in range(1, order + 1):
        for i in range(len(padded) - n + 1):
            ngram = tuple(padded[i:i+n])
            ngram_counts[n-1][ngram] += 1
            if n > 1:
                context_counts[n-1][ngram[:-1]] += 1

probs = [{} for _ in range(order)]
backoffs = [{} for _ in range(order)]
total_unigrams = sum(ngram_counts[0].values())
for (w,), count in ngram_counts[0].items():
    probs[0][(w,)] = math.log10(max(count / total_unigrams, 1e-12))
    backoffs[0][(w,)] = 0.0

for n in range(2, order + 1):
    context_extensions = Counter(ng[:-1] for ng in ngram_counts[n-1])
    for ngram, count in ngram_counts[n-1].items():
        context = ngram[:-1]
        c_ctx = context_counts[n-1][context]
        d = min(discount, count * 0.9)
        p_disc = (count - d) / c_ctx
        lower_prob = 10.0 ** probs[n-2].get(ngram[1:], -6.0)
        alpha = (d * context_extensions[context]) / c_ctx
        probs[n-1][ngram] = math.log10(max(p_disc + alpha * lower_prob, 1e-12))
        if n < order:
            backoffs[n-1][ngram] = math.log10(max(alpha, 1e-5))

arpa_lines = ["\\data\\"]
for n in range(1, order + 1):
    arpa_lines.append(f"ngram {n}={len(probs[n-1])}")
arpa_lines.append("")
for n in range(1, order + 1):
    arpa_lines.append(f"\\{n}-grams:")
    for ngram, log_p in sorted(probs[n-1].items()):
        words_str = " ".join(ngram)
        if n < order:
            arpa_lines.append(f"{log_p:.4f}\t{words_str}\t{backoffs[n-1].get(ngram, 0.0):.4f}")
        else:
            arpa_lines.append(f"{log_p:.4f}\t{words_str}")
    arpa_lines.append("")
arpa_lines.append("\\end\\")

out_p = LM_DIR / "lm_char_6gram_domain_d.arpa"
with open(out_p, "w", encoding="utf-8") as f:
    f.write("\n".join(arpa_lines) + "\n")

print(f"Successfully built {out_p.name} ({out_p.stat().st_size / (1024*1024):.2f} MB)!")
m = kenlm.Model(str(out_p))
print("Successfully verified KenLM loading!")
