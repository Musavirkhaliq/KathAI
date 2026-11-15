import re
import unicodedata
import json
from pathlib import Path

class KashmiriRuleRomanizer:
    def __init__(self, lexicon_path="/home/musa/Kashmiri Project/kashmiri_dataset/experiments/romanization_benchmark/lexicon.json"):
        self.word_lexicon = {}
        if lexicon_path and Path(lexicon_path).exists():
            with open(lexicon_path, "r", encoding="utf-8") as f:
                self.word_lexicon = json.load(f)

        # Multi-character mappings
        self.multi_char_rules = [
            ("چھ", "chh"),
            ("تھ", "th"),
            ("دھ", "dh"),
            ("کھ", "kh"),
            ("گھ", "gh"),
            ("پھ", "ph"),
            ("بھ", "bh"),
            ("جھ", "jh"),
            ("آ", "aa"),
            ("ٲ", "aa"),       # Central wavy alef
            ("او", "o"),
            ("ای", "ee"),
            ("ایٚ", "e"),
            ("ییٚ", "ye"),
            ("وٗ", "oo"),
            ("وٚ", "o"),
            ("ۄ", "o"),
            ("ۆ", "o"),
            ("ێ", "e"),
            ("ۍ", "i"),
            ("ؽ", "e"),
            ("سۭتۍ", "seti"),
            ("پؠٹھ", "peth"),
            ("پٮ۪ٹھ", "peth"),
        ]

        self.char_map = {
            'ا': 'a',
            'ب': 'b',
            'پ': 'p',
            'ت': 't',
            'ٹ': 't',
            'ث': 's',
            'ج': 'j',
            'چ': 'ch',
            'ح': 'h',
            'خ': 'kh',
            'د': 'd',
            'ڈ': 'd',
            'ذ': 'z',
            'ر': 'r',
            'ڑ': 'r',
            'ز': 'z',
            'ژ': 'ts',
            'س': 's',
            'ش': 'sh',
            'ص': 's',
            'ض': 'z',
            'ط': 't',
            'ظ': 'z',
            'ع': '',
            'غ': 'gh',
            'ف': 'f',
            'ق': 'q',
            'ک': 'k',
            'گ': 'g',
            'ل': 'l',
            'م': 'm',
            'ن': 'n',
            'و': 'w',
            'ہ': 'h',
            'ۂ': 'h',
            'ۃ': 't',
            'ھ': 'h',
            'ء': '',
            'ی': 'y',
            'ے': 'e',
            'ۓ': 'e',
            'ي': 'y',
            'ك': 'k',
            'ى': 'y',
            '۱': '1', '۲': '2', '۳': '3', '۴': '4', '۵': '5',
            '۶': '6', '۷': '7', '۸': '8', '۹': '9', '۰': '0',
            # Diacritics
            '\u064e': 'a',   # Zabar
            '\u0650': 'i',   # Zer
            '\u064f': 'u',   # Pesh
            '\u0652': '',    # Sukun
            '\u0651': '',    # Tashdid
            '\u0654': '',    # Hamza above
            '\u0655': '',    # Hamza below
            'ٕ': 'i',        # Voru / neutral vowel
            'ٖ': 'i',
            'ٗ': 'u',
            '٘': 'n',
            'ٚ': 'o',
            'ٛ': 'n',
            'ٜ': 'e',
            'ٝ': 'o',
            'ٞ': 'e',
            'ٟ': 'e',
            'ؠ': 'y',
        }

    def transliterate_word(self, word):
        w = word.strip()
        if not w:
            return ""

        # 1. Exact match in lexicon
        if w in self.word_lexicon:
            return self.word_lexicon[w]

        # 2. Match with diacritics stripped
        stripped = re.sub(r'[\u064b-\u065f\u0670\u0652\u064e\u0650\u064fٕ]', '', w)
        if stripped in self.word_lexicon:
            return self.word_lexicon[stripped]

        # 3. Rule-based fallback
        text = w
        for pattern, repl in self.multi_char_rules:
            text = text.replace(pattern, repl)

        res = []
        for ch in text:
            res.append(self.char_map.get(ch, ch))

        out = "".join(res)
        out = re.sub(r'([a-z])\1{2,}', r'\1\1', out)
        return out.lower()

    def romanize(self, text):
        if not text:
            return ""
        # Split tokens
        words = text.strip().split()
        res = [self.transliterate_word(w) for w in words]
        return " ".join(res)

if __name__ == "__main__":
    r = KashmiriRuleRomanizer()
    test_phrases = [
        "یہ چھُ اکھ ٹیسٹ",
        "مےٚ دِیوٗت اکھ ہفتہٕ",
        "طبلہ",
        "آہن حظ",
        "اسلام علیکُم",
        "پلنٛگ",
        "گاندھی جٮین دِژ تٔمِس تہٕ منی بھایی"
    ]
    for p in test_phrases:
        print(f"  {p} -> {r.romanize(p)}")
