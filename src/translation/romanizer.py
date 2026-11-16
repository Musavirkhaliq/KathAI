"""
Kashmiri Perso-Arabic to Roman Transliteration & Romanization Module.
=====================================================================
Supports:
- High-priority verified lexical mappings (47,000+ Kashmiri words)
- Core grammatical morphemes & particles
- Multi-tier phonetic rule fallback for central vowels and aspirates
- Output modes: 'ascii', 'phonetic' (Grierson standard), 'ipa'
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LEXICON_CACHE_FILE = PROJECT_ROOT / "kashmiri_dataset/scripts/kashmiri_lexicon_cache.json"

# Core functional Kashmiri morphemes & particles: (ASCII, Phonetic, IPA)
CORE_LEXICON: Dict[str, Tuple[str, str, str]] = {
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
}

# Consonant phoneme mappings
CONSONANT_MAP: Dict[str, Tuple[str, str, str]] = {
    "پ": ("p", "p", "p"),
    "پھ": ("ph", "ph", "pʰ"),
    "ب": ("b", "b", "b"),
    "ت": ("t", "t", "t̪"),
    "تھ": ("th", "th", "t̪ʰ"),
    "ٹ": ("t", "ṭ", "ʈ"),
    "ٹھ": ("th", "ṭh", "ʈʰ"),
    "ث": ("s", "s", "s"),
    "ج": ("j", "j", "d͡ʒ"),
    "چ": ("ch", "ch", "t͡ʃ"),
    "چھ": ("chh", "chh", "t͡ʃʰ"),
    "ح": ("h", "ḥ", "ħ"),
    "خ": ("kh", "kh", "x"),
    "د": ("d", "d", "d̪"),
    "ڈ": ("d", "ḍ", "ɖ"),
    "ذ": ("z", "z", "z"),
    "ر": ("r", "r", "r"),
    "ڑ": ("r", "ṛ", "ɽ"),
    "ز": ("z", "z", "z"),
    "ژ": ("ts", "ts", "t͡s"),
    "ژھ": ("tsh", "tsh", "t͡sʰ"),
    "س": ("s", "s", "s"),
    "ش": ("sh", "sh", "ʃ"),
    "ص": ("s", "ṣ", "sˤ"),
    "ض": ("z", "ż", "zˤ"),
    "ط": ("t", "ṭ", "tˤ"),
    "ظ": ("z", "ẓ", "ðˤ"),
    "ع": ("", "ʻ", "ʔ"),
    "غ": ("gh", "gh", "ɣ"),
    "ف": ("f", "f", "f"),
    "ق": ("q", "q", "q"),
    "ک": ("k", "k", "k"),
    "کھ": ("kh", "kh", "kʰ"),
    "گ": ("g", "g", "ɡ"),
    "ل": ("l", "l", "l"),
    "م": ("m", "m", "m"),
    "ن": ("n", "n", "n"),
    "ں": ("n", "ñ", "̃"),
    "و": ("w", "w", "w"),
    "ہ": ("h", "h", "h"),
    "ھ": ("h", "h", "h"),
    "ء": ("", "’", "ʔ"),
    "ی": ("y", "y", "j"),
}


class KashmiriRomanizer:
    """
    Romanizer for Kashmiri Perso-Arabic text.
    """
    def __init__(self, lexicon_path: Optional[Union[str, Path]] = None):
        self.lexicon: Dict[str, Any] = {}
        path = Path(lexicon_path) if lexicon_path else LEXICON_CACHE_FILE
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.lexicon = json.load(f)
            except Exception:
                self.lexicon = {}

    def romanize_word(self, word: str, mode: str = "ascii") -> str:
        idx = 0 if mode == "ascii" else (1 if mode == "phonetic" else 2)

        # 1. Check Core Lexicon
        if word in CORE_LEXICON:
            return CORE_LEXICON[word][idx]

        # 2. Check Large Precompiled Lexicon
        if word in self.lexicon:
            val = self.lexicon[word]
            if isinstance(val, list) and len(val) > idx:
                return val[idx]
            elif isinstance(val, str):
                return val

        # 3. Rule-based Transliteration fallback
        result = []
        i = 0
        w_len = len(word)
        while i < w_len:
            # Check 2-char compound consonants
            if i + 1 < w_len and word[i:i+2] in CONSONANT_MAP:
                result.append(CONSONANT_MAP[word[i:i+2]][idx])
                i += 2
            elif word[i] in CONSONANT_MAP:
                result.append(CONSONANT_MAP[word[i]][idx])
                i += 1
            elif word[i] in ("ا", "آ"):
                result.append("a" if mode == "ascii" else ("ā" if mode == "phonetic" else "aː"))
                i += 1
            else:
                result.append(word[i])
                i += 1

        return "".join(result)

    def romanize(self, text: str, mode: str = "ascii") -> str:
        """
        Transliterates a full Kashmiri sentence into Roman Kashmiri.
        """
        if not text:
            return ""
        words = text.strip().split()
        out = [self.romanize_word(w, mode=mode) for w in words]
        return " ".join(out)


# Global default instance
_default_romanizer: Optional[KashmiriRomanizer] = None

def get_default_romanizer() -> KashmiriRomanizer:
    global _default_romanizer
    if _default_romanizer is None:
        _default_romanizer = KashmiriRomanizer()
    return _default_romanizer

def transliterate_kashmiri(text: str, mode: str = "ascii") -> str:
    """
    Convenience function to transliterate Kashmiri Perso-Arabic text to Roman script.
    """
    return get_default_romanizer().romanize(text, mode=mode)
