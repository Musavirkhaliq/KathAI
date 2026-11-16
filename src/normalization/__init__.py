"""
Kashmiri Text Normalization Package.
"""

from src.normalization.normalizer import (
    normalize_kashmiri_text,
    normalize_for_wer_evaluation,
    detect_code_switching,
    KashmiriNormalizer,
    CHAR_MAP,
    DIGIT_MAP,
    PUNCT_MAP,
    DIACRITICS_REGEX
)

__all__ = [
    "normalize_kashmiri_text",
    "normalize_for_wer_evaluation",
    "detect_code_switching",
    "KashmiriNormalizer",
    "CHAR_MAP",
    "DIGIT_MAP",
    "PUNCT_MAP",
    "DIACRITICS_REGEX"
]
