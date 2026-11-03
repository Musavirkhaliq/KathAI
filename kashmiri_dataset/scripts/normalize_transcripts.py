#!/usr/bin/env python3
"""
Kashmiri Transcript Normalization & Script Harmonization Module
"""
import re
import unicodedata
from typing import Dict, List

CHAR_MAP = {
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

DIGIT_MAP = {
    '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
    '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9',
    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
}

PUNCT_MAP = {
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

def detect_code_switching(text: str) -> Dict[str, any]:
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

def normalize_for_wer_evaluation(text: str) -> str:
    norm = normalize_kashmiri_text(text, strip_diacritics=True, normalize_digits=True, preserve_zwnj=False)
    norm = re.sub(r'[^\w\s]', ' ', norm)
    norm = re.sub(r'\s+', ' ', norm).strip()
    return norm

if __name__ == '__main__':
    samples = [
        "مِہ چھ ناو توسیف جیلانی",
        "اووَر آل اگر إسِہ پریکٹِکل تٕہ تھیورِٹِکل نالیج MBA وُچھو",
        " لال اوس نپہٕ نپہٕ کران ۱۲۳۴ ",
    ]
    for s in samples:
        print("Raw: ", s)
        print("Norm:", normalize_kashmiri_text(s))
        print("Eval:", normalize_for_wer_evaluation(s))
        print("CS:  ", detect_code_switching(s))
        print("---")
