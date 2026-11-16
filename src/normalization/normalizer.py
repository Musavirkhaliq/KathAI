"""
Kashmiri Transcript Normalization & Script Harmonization Module.
================================================================
Implements Perso-Arabic orthographic standardization, diacritic filtering,
zero-width non-joiner (ZWNJ) management, numeral mapping, and code-switching detection.
"""

import re
import unicodedata
from typing import Dict, Any, Optional

# Perso-Arabic Kashmiri canonical character substitutions
CHAR_MAP: Dict[str, str] = {
    '\u0643': '\u06a9',  # Arabic Kaf -> Kashmiri Keheh
    '\u06aa': '\u06a9',  # Swash Kaf -> Kashmiri Keheh
    '\u06b3': '\u06af',  # Gaf variants -> Kashmiri Gaf
    '\u0647': '\u06c1',  # Arabic Heh -> Heh Goal
    '\u0629': '\u06c1',  # Teh Marbuta -> Heh Goal
    '\u06c3': '\u06c2',  # Teh Marbuta Goal -> Heh Goal with Hamza
    '\u0649': '\u06cc',  # Alef Maksura -> Kashmiri Yeh
    '\u064a': '\u06cc',  # Arabic Yeh -> Kashmiri Yeh
    '\u0640': '',        # Tatweel / Kashida
    '\u200b': '',        # Zero-width space
    '\ufeff': '',        # BOM
}

DIACRITICS_REGEX = re.compile(r'[\u064b-\u065f\u0670\u06d6-\u06ed]')

DIGIT_MAP: Dict[str, str] = {
    '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
    '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9',
    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
}

PUNCT_MAP: Dict[str, str] = {
    '،': ' ، ',
    '؛': ' ؛ ',
    '؟': ' ؟ ',
    '۔': ' ۔ ',
    '?': ' ؟ ',
    ',': ' ، ',
    ';': ' ؛ ',
    '!': ' ! ',
    '[': ' ', ']': ' ',
    '(': ' ', ')': ' ',
    '{': ' ', '}': ' ',
    '"': ' ', "'": ' ',
    '—': ' ', '–': ' ',
    '-': ' ', '_': ' ',
}


def normalize_kashmiri_text(
    text: str,
    strip_diacritics: bool = False,
    normalize_digits: bool = True,
    preserve_zwnj: bool = True
) -> str:
    """
    Normalizes Kashmiri Perso-Arabic text to canonical orthographic representation.
    """
    if not text or not isinstance(text, str):
        return ""

    text = unicodedata.normalize('NFC', text)

    for src, tgt in CHAR_MAP.items():
        text = text.replace(src, tgt)

    if normalize_digits:
        for src, tgt in DIGIT_MAP.items():
            text = text.replace(src, tgt)

    if strip_diacritics:
        text = DIACRITICS_REGEX.sub('', text)

    if not preserve_zwnj:
        text = text.replace('\u200c', ' ').replace('\u200d', '')
    else:
        text = re.sub(r'[\u200c\u200d]+', '\u200c', text)
        text = text.replace(' \u200c', '\u200c').replace('\u200c ', '\u200c')

    for p_src, p_tgt in PUNCT_MAP.items():
        text = text.replace(p_src, p_tgt)

    # Filter non-printable ASCII control characters
    text = "".join(ch for ch in text if ord(ch) >= 32 or ch in ('\n', '\t'))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_for_wer_evaluation(text: str) -> str:
    """
    Normalizes reference and hypothesis text for standard Word Error Rate (WER) computation.
    Strips diacritics, maps numerals, removes punctuation, and normalizes whitespace.
    """
    norm = normalize_kashmiri_text(text, strip_diacritics=True, normalize_digits=True, preserve_zwnj=False)
    norm = re.sub(r'[^\w\s]', ' ', norm)
    norm = re.sub(r'\s+', ' ', norm).strip()
    return norm


def detect_code_switching(text: str) -> Dict[str, Any]:
    """
    Analyzes Kashmiri text for multilingual code-switching (Perso-Arabic, Latin, Devanagari).
    """
    if not text:
        return {
            'is_code_switched': False,
            'primary_script': 'none',
            'arabic_ratio': 0.0,
            'latin_ratio': 0.0,
            'devanagari_ratio': 0.0,
            'latin_tokens': []
        }

    total_chars = 0
    arabic_chars = 0
    latin_chars = 0
    deva_chars = 0

    latin_words = []
    tokens = text.split()

    for token in tokens:
        clean_tok = re.sub(r'[^\w]', '', token)
        if re.search(r'[A-Za-z]', clean_tok):
            latin_words.append(clean_tok)

    for ch in text:
        if ch.isspace() or unicodedata.category(ch).startswith('P'):
            continue
        total_chars += 1
        code = ord(ch)
        if (0x0600 <= code <= 0x06FF) or (0x0750 <= code <= 0x077F) or (0x08A0 <= code <= 0x08FF) or (0xFB50 <= code <= 0xFDFF) or (0xFE70 <= code <= 0xFEFF):
            arabic_chars += 1
        elif (0x0041 <= code <= 0x005A) or (0x0061 <= code <= 0x007A):
            latin_chars += 1
        elif 0x0900 <= code <= 0x097F:
            deva_chars += 1

    if total_chars == 0:
        return {
            'is_code_switched': False,
            'primary_script': 'none',
            'arabic_ratio': 0.0,
            'latin_ratio': 0.0,
            'devanagari_ratio': 0.0,
            'latin_tokens': []
        }

    ar_ratio = arabic_chars / total_chars
    lat_ratio = latin_chars / total_chars
    dev_ratio = deva_chars / total_chars

    is_cs = (lat_ratio > 0.05) or (dev_ratio > 0.05) or (len(latin_words) > 0)
    primary = 'perso_arabic'
    if lat_ratio > ar_ratio and lat_ratio > dev_ratio:
        primary = 'latin'
    elif dev_ratio > ar_ratio and dev_ratio > lat_ratio:
        primary = 'devanagari'

    return {
        'is_code_switched': is_cs,
        'primary_script': primary,
        'arabic_ratio': round(ar_ratio, 4),
        'latin_ratio': round(lat_ratio, 4),
        'devanagari_ratio': round(dev_ratio, 4),
        'latin_tokens': latin_words
    }


class KashmiriNormalizer:
    """
    Object-oriented Kashmiri Normalizer wrapper.
    """
    def __init__(self, strip_diacritics: bool = False, normalize_digits: bool = True, preserve_zwnj: bool = True):
        self.strip_diacritics = strip_diacritics
        self.normalize_digits = normalize_digits
        self.preserve_zwnj = preserve_zwnj

    def normalize(self, text: str) -> str:
        return normalize_kashmiri_text(
            text,
            strip_diacritics=self.strip_diacritics,
            normalize_digits=self.normalize_digits,
            preserve_zwnj=self.preserve_zwnj
        )

    def normalize_for_eval(self, text: str) -> str:
        return normalize_for_wer_evaluation(text)

    def analyze_code_switching(self, text: str) -> Dict[str, Any]:
        return detect_code_switching(text)
