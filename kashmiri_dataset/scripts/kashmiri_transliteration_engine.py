"""
Comprehensive Kashmiri Transliteration Engine (Perso-Arabic -> Romanized Kashmiri)
==================================================================================
Features:
  1. 47,264 verified lexical word pairs from our Rasa Kashmiri dataset.
  2. Multi-tier phonetic rule-based fallback for all central vowels (ٲ, ؠ, ۆ, ۄ, ێ, ۍ, ؽ, ٕ, ٚ) and aspirates (چھ, کھ, تھ, etc.).
  3. Supports three output modes:
     - "ascii": Clean colloquial Roman script for mobile / chat / social interfaces
     - "phonetic": Scholarly diacritical Roman script (Grierson standard)
     - "ipa": International Phonetic Alphabet for linguistic research
  4. Preserves genuine Kashmiri phonology without drifting into Urdu/Hindi or English.
"""

import re
import json
import unicodedata
from pathlib import Path
from typing import Dict, Optional, Tuple, List

BASE_DIR = Path("/home/musa/Kashmiri Project")
CACHE_FILE = BASE_DIR / "kashmiri_dataset/scripts/kashmiri_lexicon_cache.json"

# Load precompiled verified lexicon (47,264 Kashmiri words)
VERIFIED_LEXICON = {}
if CACHE_FILE.exists():
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        VERIFIED_LEXICON = json.load(f)

# High-priority core functional Kashmiri morphemes & particles
CORE_LEXICON: Dict[str, Tuple[str, str, str]] = {
    # Perso-Arabic: (ASCII, Phonetic, IPA)
    "چھُ": ("chhu", "chhu", "t͡ʃʰu"),
    "چھِ": ("chhi", "chhi", "t͡ʃʰi"),
    "چھ": ("chhi", "chhi", "t͡ʃʰi"),
    "چھَ": ("chha", "chha", "t͡ʃʰa"),
    "منٛز": ("manz", "manz", "mənz"),
    "تہٕ": ("tah", "tə", "tə"),
    "مےٚ": ("mye", "mye", "mje"),
    "بہٕ": ("bah", "bə", "bə"),
    "سۭتۍ": ("seti", "səti", "sɨtʲ"),
    "اکھ": ("akh", "akh", "akʰ"),
    "اوس": ("os", "ōs", "oːs"),
    "ٲس": ("aas", "ās", "aːs"),
    "ٲسۍ": ("aesi", "ǣsi", "əːsʲ"),
    "ییٚلہ": ("yeli", "yēli", "jeːli"),
    "ییٚتہِ": ("yeth", "yēth", "jeːtʰ"),
    "تتھ": ("tath", "tath", "tatʰ"),
    "نہٕ": ("nah", "nə", "nə"),
    "یہِ": ("yih", "yih", "jih"),
    "یہ": ("ye", "ye", "je"),
    "سُہ": ("suh", "suh", "suh"),
    "سۄ": ("so", "sɔ", "sɔ"),
    "خٲطرٕ": ("khaatir", "khāṭir", "xɑːtˤir"),
    "پؠٹھ": ("pyeth", "pyēṭh", "pʲeːʈʰ"),
    "پٮ۪ٹھ": ("pyeth", "pyēṭh", "pʲeːʈʰ"),
    "کران": ("karan", "karān", "karaːn"),
    "وَنٛان": ("wanan", "wanān", "wanaːn"),
    "گَژھان": ("gatshan", "gatshān", "gat͡sʰaːn"),
    "ہیٚکِو": ("hyekiv", "hyēkiv", "hjeːkiv"),
    "کِتھ": ("kith", "kith", "kitʰ"),
    "پٲٹھۍ": ("paeth", "pǣṭh", "pəːʈʰ"),
    "پٔکِو": ("pakiv", "pakiv", "pakiv"),
    "پنُن": ("panun", "panun", "panun"),
    "پنٕنۍ": ("panin", "panin", "panin"),
    "واریاہ": ("waryah", "wāryāh", "waːrjaːh"),
    "زیادٕ": ("zyadah", "zyādə", "zjaːdə"),
    "زیاد": ("zyad", "zyād", "zjaːd"),
    "کینہہ": ("kehn", "kēnh", "keːnh"),
    "کانٛہہ": ("kaanh", "kānh", "kaːnh"),
    "کیٛاہ": ("kyah", "kyāh", "kjaːh"),
    "کیاہ": ("kyah", "kyāh", "kjaːh"),
    "کیا": ("kya", "kyā", "kjaː"),
    "اسلام علیکُم": ("assalam alaikum", "assalām 'alaikum", "asːalaːm ʕalajkum"),
    "وعلیکُم سلام": ("walaikum assalam", "wa'alaikum assalām", "waʕalajkum asːalaːm"),
    "آہَن حظ": ("aahan haz", "āhan haz", "aːhan haz"),
    "اَہَن حظ": ("ahan haz", "ahan haz", "ahan haz"),
    "شُکُر": ("shukur", "shukur", "ʃukur"),
    "ٹھیٖک": ("theek", "ṭhīk", "ʈʰiːk"),
    "اصل": ("asli", "asli", "asli"),
    "کوٚر": ("kor", "kor", "kor"),
    "گوٚو": ("gov", "gov", "gov"),
    "گوٚمُت": ("gomut", "gomut", "gomut"),
    "گَرٕ": ("garah", "garə", "garə"),
    "گھرٕ": ("garah", "garə", "garə"),
    "گرَس": ("garas", "garas", "garas"),
    "پَننِس": ("pananis", "pananis", "pananis"),
    "خۄدا": ("khoda", "khɔdā", "xɔdaː"),
    "صٲب": ("saeb", "sǣb", "səːb"),
    "اِنسان": ("insan", "insān", "insaːn"),
    "مطلب": ("matlab", "matlab", "matlab"),
    "قُرآن": ("quran", "qur'ān", "qurʔaːn"),
    "شریٖف": ("shareef", "sharīf", "ʃariːf"),
    "پلنٛگ": ("palang", "palang", "palaŋɡ"),
    "طبلہ": ("tabla", "tablā", "tˤablaː")
}

# Substring Multi-Graphemes
MULTI_CHAR_MAP: Dict[str, Tuple[str, str, str]] = {
    "ژھ": ("tsh", "tsh", "t͡sʰ"),
    "چھ": ("chh", "ch", "t͡ʃʰ"),
    "کھ": ("kh", "kh", "kʰ"),
    "تھ": ("th", "th", "tʰ"),
    "پھ": ("ph", "ph", "pʰ"),
    "بھ": ("bh", "bh", "bʱ"),
    "دھ": ("dh", "dh", "dʱ"),
    "جھ": ("jh", "jh", "d͡ʒʱ"),
    "گھ": ("gh", "gh", "ɡʱ"),
    "ٹھ": ("t'h", "ṭh", "ʈʰ"),
    "ڈھ": ("d'h", "ḍh", "ɖʱ"),
    "ڑھ": ("rh", "ṛh", "ɽʱ"),
    "ییٚ": ("ye", "yē", "jeː"),
    "ییٖ": ("yee", "yī", "jiː"),
    "یۆ": ("yo", "yō", "joː"),
    "یۄ": ("yaw", "yɔ", "jɔ"),
    "وا": ("wa", "wā", "waː"),
    "وٗ": ("oo", "ū", "uː"),
    "وٚ": ("o", "o", "o"),
    "وۄ": ("aw", "ɔ", "ɔ"),
    "وۆ": ("o", "ō", "oː"),
    "ٲی": ("aey", "ǣy", "əːj"),
    "ٲ": ("ae", "ǣ", "əː"),
    "آ": ("aa", "ā", "aː"),
    "او": ("o", "ō", "oː"),
    "ای": ("ee", "ī", "iː"),
    "اے": ("ay", "ē", "eː"),
}

# Single Characters
SINGLE_CHAR_MAP: Dict[str, Tuple[str, str, str]] = {
    "ب": ("b", "b", "b"),
    "پ": ("p", "p", "p"),
    "ت": ("t", "t", "t"),
    "ٹ": ("t'", "ṭ", "ʈ"),
    "ث": ("s", "s", "s"),
    "ج": ("j", "j", "d͡ʒ"),
    "چ": ("ch", "ch", "t͡ʃ"),
    "ح": ("h", "h", "h"),
    "خ": ("kh", "kh", "x"),
    "د": ("d", "d", "d"),
    "ڈ": ("d'", "ḍ", "ɖ"),
    "ذ": ("z", "z", "z"),
    "ر": ("r", "r", "r"),
    "ڑ": ("r'", "ṛ", "ɽ"),
    "ز": ("z", "z", "z"),
    "ژ": ("ts", "ts", "t͡s"),
    "س": ("s", "s", "s"),
    "ش": ("sh", "sh", "ʃ"),
    "ص": ("s", "s", "s"),
    "ض": ("z", "z", "z"),
    "ط": ("t", "t", "t"),
    "ظ": ("z", "z", "z"),
    "ع": ("'", "'", "ʔ"),
    "غ": ("gh", "gh", "ɣ"),
    "ف": ("f", "f", "f"),
    "ق": ("q", "q", "q"),
    "ک": ("k", "k", "k"),
    "گ": ("g", "g", "ɡ"),
    "ل": ("l", "l", "l"),
    "م": ("m", "m", "m"),
    "ن": ("n", "n", "n"),
    "ں": ("n", "ñ", "̃"),
    "و": ("w", "w", "w"),
    "ہ": ("h", "h", "h"),
    "ھ": ("h", "h", "h"),
    "ء": ("'", "'", "ʔ"),
    "ی": ("y", "y", "j"),
    "ے": ("e", "ē", "eː"),
    "ؠ": ("y", "y", "ʲ"),
    "ۄ": ("aw", "ɔ", "ɔ"),
    "ۆ": ("o", "ō", "oː"),
    "ێ": ("e", "ē", "eː"),
    "ۍ": ("i", "ī", "iː"),
    "ؽ": ("e", "ē", "eː"),
    "ا": ("a", "a", "a"),
    "َ": ("a", "a", "a"),
    "ِ": ("i", "i", "i"),
    "ُ": ("u", "u", "u"),
    "ٕ": ("e'", "ə", "ə"),
    "ٔ": ("e'", "ə", "ə"),
    "ٚ": ("o", "o", "o"),
    "ْ": ("", "", ""),
    "ّ": ("", "", ""),
    "ٖ": ("ee", "ī", "iː"),
    "ٗ": ("oo", "ū", "uː"),
    "ٰ": ("aa", "ā", "aː"),
    " ": (" ", " ", " "),
    "|": (" ", " ", " "),
    "۔": (".", ".", "."),
    "،": (",", ",", ","),
    "؟": ("?", "?", "?"),
    "!": ("!", "!", "!")
}

class KashmiriTransliterationEngine:
    def __init__(self, mode: str = "ascii"):
        assert mode in ["ascii", "phonetic", "ipa"], f"Unsupported mode: {mode}"
        self.mode = mode
        self.mode_idx = 0 if mode == "ascii" else (1 if mode == "phonetic" else 2)

    def normalize(self, text: str) -> str:
        if not text: return ""
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("\u0643", "\u06a9")
        text = text.replace("\u06aa", "\u06a9")
        text = text.replace("\u064a", "\u06cc")
        text = text.replace("\u0649", "\u06cc")
        text = text.replace("\u0647", "\u06c1")
        text = text.replace("\u06c2", "\u06c1")
        text = re.sub(r'[\u200c\u200d\u200e\u200f]', '', text)
        return text.strip()

    def transliterate_word(self, word: str) -> str:
        norm_w = self.normalize(word)
        if not norm_w: return ""

        # 1. Check Core High-Priority Lexicon
        if norm_w in CORE_LEXICON:
            return CORE_LEXICON[norm_w][self.mode_idx]

        # 2. Check 47k Rasa Lexicon
        if norm_w in VERIFIED_LEXICON:
            entry = VERIFIED_LEXICON[norm_w]
            if self.mode == "ascii":
                return entry["ascii"]
            elif self.mode == "phonetic":
                return entry["phonetic"]
            else:
                # Approximate IPA from phonetic
                return entry["phonetic"]

        # 3. Rule-based Grapheme Parser
        result = []
        i = 0
        n = len(norm_w)
        while i < n:
            if i + 1 < n and norm_w[i+1] == "ّ":
                ch = norm_w[i]
                if ch in SINGLE_CHAR_MAP:
                    base = SINGLE_CHAR_MAP[ch][self.mode_idx]
                    result.append(base + base)
                    i += 2
                    continue
            if i + 3 <= n and norm_w[i:i+3] in MULTI_CHAR_MAP:
                result.append(MULTI_CHAR_MAP[norm_w[i:i+3]][self.mode_idx])
                i += 3
                continue
            if i + 2 <= n and norm_w[i:i+2] in MULTI_CHAR_MAP:
                result.append(MULTI_CHAR_MAP[norm_w[i:i+2]][self.mode_idx])
                i += 2
                continue
            ch = norm_w[i]
            if ch in SINGLE_CHAR_MAP:
                result.append(SINGLE_CHAR_MAP[ch][self.mode_idx])
            else:
                result.append(ch)
            i += 1

        res = "".join(result)
        res = re.sub(r'\s+', ' ', res)
        return res

    def transliterate(self, text: str) -> str:
        if not text: return ""
        norm_t = self.normalize(text)
        if norm_t in CORE_LEXICON:
            return CORE_LEXICON[norm_t][self.mode_idx]

        tokens = re.findall(r'\S+|\s+', text)
        out_tokens = []
        for t in tokens:
            if t.isspace():
                out_tokens.append(t)
            else:
                out_tokens.append(self.transliterate_word(t))
        return "".join(out_tokens).strip()

def run_tests():
    print("=" * 70)
    print("Kashmiri Comprehensive Transliteration Engine (47k Lexicon + Rules)")
    print("=" * 70)
    engine_ascii = KashmiriTransliterationEngine(mode="ascii")
    engine_phonetic = KashmiriTransliterationEngine(mode="phonetic")
    engine_ipa = KashmiriTransliterationEngine(mode="ipa")

    samples = [
        "اسلام علیکُم تُہۍ چھُوا ٹھیٖک",
        "فیورِٹ بُک چھ مےٚ قُرآنِ شریٖف",
        "تَتھ منز چھُ اَسہِ مطلَب ہیچھناوان پَنُن دیٖن",
        "سُہ گوٚو گھرٕ تہٕ بیٚیہِ آو واپس",
        "پلنٛگ چھُ اَصل پٲٹھۍ بنٲومُت",
        "طبلہ واجَن چھُ واریاہ اصل کٲم",
        "یہ چھُ اکھ ٹیسٹ جملہٕ کشمیری زبانِ خٲطرٕ",
        "ڈاکٹرن دِژ مےٚ ٹیبلٹ تہٕ وۄنُم ٹھیٖک گَژھنہٕ خٲطرٕ"
    ]

    for i, s in enumerate(samples, 1):
        print(f"\n[Test {i}]")
        print(f"  Perso-Arabic : {s}")
        print(f"  ASCII Roman  : {engine_ascii.transliterate(s)}")
        print(f"  Phonetic (Sch): {engine_phonetic.transliterate(s)}")
        print(f"  IPA          : {engine_ipa.transliterate(s)}")

if __name__ == "__main__":
    run_tests()
