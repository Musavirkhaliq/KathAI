import re
import unicodedata
import numpy as np

def levenshtein_distance(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y), dtype=int)
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix[x, y] = matrix[x-1, y-1]
            else:
                matrix[x, y] = min(
                    matrix[x-1, y] + 1,     # deletion
                    matrix[x-1, y-1] + 1,   # substitution
                    matrix[x, y-1] + 1      # insertion
                )
    return matrix[size_x - 1, size_y - 1]

def calculate_wer_and_cer(ref, hyp):
    ref_words = ref.strip().split()
    hyp_words = hyp.strip().split()
    
    # Word level
    if len(ref_words) == 0:
        wer = 1.0 if len(hyp_words) > 0 else 0.0
        word_dist = len(hyp_words)
        ref_word_count = 0
    else:
        word_dist = levenshtein_distance(ref_words, hyp_words)
        wer = word_dist / len(ref_words)
        ref_word_count = len(ref_words)
        
    # Char level
    ref_chars = list(ref.replace(" ", ""))
    hyp_chars = list(hyp.replace(" ", ""))
    if len(ref_chars) == 0:
        cer = 1.0 if len(hyp_chars) > 0 else 0.0
        char_dist = len(hyp_chars)
        ref_char_count = 0
    else:
        char_dist = levenshtein_distance(ref_chars, hyp_chars)
        cer = char_dist / len(ref_chars)
        ref_char_count = len(ref_chars)
        
    return {
        "wer": wer,
        "cer": cer,
        "word_dist": word_dist,
        "char_dist": char_dist,
        "ref_words": ref_word_count,
        "ref_chars": ref_char_count,
        "exact_match": (ref.strip() == hyp.strip())
    }

def normalize_perso_arabic_kashmiri(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    # Standardize Arabic Kaf to Kashmiri Keheh
    text = text.replace("\u0643", "\u06a9")
    # Standardize Heh variants (Arabic Heh -> Heh Goal)
    text = text.replace("\u0647", "\u06c1")
    # Standardize Arabic Yeh to Farsi Yeh
    text = text.replace("\u064a", "\u06cc")
    # Remove decorative non-contrastive diacritics for normalized evaluation (Fatha, Damma, Kasra, Shadda, Sukun)
    text = re.sub(r'[\u064b-\u0652\u0653\u0654\u0670]', '', text)
    # Remove punctuation
    text = re.sub(r'[۔،؛!؟\.,:;?"\'\(\)\[\]\{\}\-\_]', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

if __name__ == "__main__":
    t1 = "مےٚ دِیوٗت اکھ ہفتہٕ"
    t2 = "مے دیوت اکھ ہفتہ"
    print("Raw Dist:", calculate_wer_and_cer(t1, t2))
    print("Norm Dist:", calculate_wer_and_cer(normalize_perso_arabic_kashmiri(t1), normalize_perso_arabic_kashmiri(t2)))
