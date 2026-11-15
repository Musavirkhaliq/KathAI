import re
import json
import unicodedata
from pathlib import Path

# Load vocabulary lexicon from Rasa
LEXICON_PATH = Path("/home/musa/Kashmiri Project/kashmiri_dataset/external/krish187653_rasa-kashmiri-codemixing-transliterations/data/all_rows/train.jsonl")

class KashmiriPhoneticRuleTransliteration:
    def __init__(self, build_lexicon=True):
        self.lexicon = {}
        if build_lexicon and LEXICON_PATH.exists():
            self._load_lexicon()
        
        # Phonetic character mapping
        self.char_map = {
            # Consonants
            "ب": "b", "پ": "p", "ت": "t", "ٹ": "t", "ث": "s",
            "ج": "j", "چ": "ch", "ح": "h", "خ": "kh",
            "د": "d", "ڈ": "d", "ذ": "z", "ر": "r", "ڑ": "r",
            "ز": "z", "ژ": "ts", "س": "s", "ش": "sh",
            "ص": "s", "ض": "z", "ط": "t", "ظ": "z",
            "ع": "a", "غ": "gh", "ف": "f", "ق": "q",
            "ک": "k", "گ": "g", "ل": "l", "م": "m",
            "ن": "n", "و": "v", "ہ": "h", "ھ": "h",
            "ی": "y", "ے": "e", "ك": "k", "ك": "k",
            "ي": "y", "ى": "y",
            
            # Kashmiri Special Vowels and Diacritics
            "ٲ": "a", "ؠ": "y", "ۆ": "o", "ۄ": "o", 
            "ێ": "e", "ۍ": "i", "ؽ": "y", "ا": "a",
            "آ": "aa", "أ": "a", "إ": "i", "ؤ": "o",
            "ں": "n", "نٛ": "n", "ںٛ": "n",
            
            # Diacritical marks
            "َ": "a",   # Zabar / Fatha
            "ِ": "i",   # Zer / Kasra
            "ُ": "u",   # Pesh / Damma
            "ٕ": "e",   # Kashmiri neutral vowel (schwa / voru)
            "ٗ": "oo",  # Ulti Pesh
            "ۡ": "",    # Sukun
            "ّ": "",    # Tashdeed
            "ٔ": "",    # Hamza above
            "ء": "",    # Hamza
            
            # Punctuation & digits
            "،": ",", "؛": ";", "؟": "?", "۔": ".",
            "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
            "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9"
        }

    def _load_lexicon(self):
        with open(LEXICON_PATH, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                ws = d.get("word_sequence")
                if isinstance(ws, list):
                    for w in ws:
                        src = w.get("source_span")
                        asc = w.get("ascii")
                        if src and asc:
                            s_clean = unicodedata.normalize("NFKC", src.strip())
                            if s_clean not in self.lexicon:
                                self.lexicon[s_clean] = asc.strip().lower()

    def transliterate_word(self, word):
        w_norm = unicodedata.normalize("NFKC", word.strip())
        if not w_norm:
            return ""
        
        # 1. Lexicon exact lookup
        if w_norm in self.lexicon:
            return self.lexicon[w_norm]
        
        # 2. Rule-based transliteration
        res = []
        i = 0
        n = len(w_norm)
        while i < n:
            ch = w_norm[i]
            # Digraphs
            if ch == "چ" and i + 1 < n and w_norm[i+1] == "ھ":
                res.append("chh")
                i += 2
                continue
            if ch == "ت" and i + 1 < n and w_norm[i+1] == "ھ":
                res.append("th")
                i += 2
                continue
            if ch == "د" and i + 1 < n and w_norm[i+1] == "ھ":
                res.append("dh")
                i += 2
                continue
            if ch == "ک" and i + 1 < n and w_norm[i+1] == "ھ":
                res.append("kh")
                i += 2
                continue
            if ch == "پ" and i + 1 < n and w_norm[i+1] == "ھ":
                res.append("ph")
                i += 2
                continue
            if ch == "ب" and i + 1 < n and w_norm[i+1] == "ھ":
                res.append("bh")
                i += 2
                continue
            if ch == "ج" and i + 1 < n and w_norm[i+1] == "ھ":
                res.append("jh")
                i += 2
                continue
            if ch == "ٹ" and i + 1 < n and w_norm[i+1] == "ھ":
                res.append("th")
                i += 2
                continue
            if ch == "ڈ" and i + 1 < n and w_norm[i+1] == "ھ":
                res.append("dh")
                i += 2
                continue
            
            res.append(self.char_map.get(ch, ch))
            i += 1
            
        trans = "".join(res)
        # Collapse multi-vowels and cleanup
        trans = re.sub(r'([a-zA-Z])\1{2,}', r'\1\1', trans)
        return trans

    def transliterate(self, text):
        words = text.split()
        return " ".join(self.transliterate_word(w) for w in words)

if __name__ == "__main__":
    t = KashmiriPhoneticRuleTransliteration()
    sample = "گاندھی جٮین دِژ تٔمِس تہٕ ارالی کنچنَس منٛز اکھ نیچر تھراپی آشرم"
    print("Input :", sample)
    print("Output:", t.transliterate(sample))
