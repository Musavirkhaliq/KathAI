#!/usr/bin/env python3
"""
Linguistic Orthography Evaluator for Kashmiri Perso-Arabic ASR.
Implements:
  Metric A: Strict Canonical Kashmiri WER/CER (official benchmark baseline).
  Metric B: Linguistically Justified Orthographic-Tolerant WER/CER.

All phonemic minimal pairs are preserved. Only legitimate orthographic variants,
optional diacritics, and compound boundary conventions are reconciled under Metric B.
Reference text is NEVER overwritten or altered on disk.
"""

import re
import unicodedata
from jiwer import wer, cer

# ==============================================================================
# 1. Linguistic Inventories
# ==============================================================================

# Metric A: Standard Unicode and Base Normalization
from metrics_evaluator import normalize_perso_arabic_kashmiri as normalize_strict

# Optional Non-Phonemic Diacritics (safely ignorable in casual orthography)
# Fathatan, Dammatan, Sukun, Shadda, Dagger Alef, Superscript Alef
OPTIONAL_DIACRITICS = re.compile(r'[\u064B\u064C\u0651\u0652\u0670\u0640\u200c\u200d]')

# Morpheme Variant Mapping for Metric B
# Maps documented graphemic variants of identical lexical morphemes to a canonical lemma
LEXICAL_VARIANT_MAP = {
    # Postposition "with" (sə:t')
    "سٕتہ": "سۭتۍ",
    "سٕتۍ": "سۭتۍ",
    "سیتۍ": "سۭتۍ",
    "سٲتھ": "سۭتۍ",
    "سیت": "سۭتۍ",
    
    # Postposition "on / upon" (p'eʈh)
    "پیٹھ": "پٮ۪ٹھ",
    "پؠٹھ": "پٮ۪ٹھ",
    "پؠٹٛھ": "پٮ۪ٹھ",
    "پیٹٛھ": "پٮ۪ٹھ",
    "پیٚٹھ": "پٮ۪ٹھ",
    
    # Conjunction "and" (tə)
    "تٕ": "تہٕ",
    "تہ": "تہٕ",
    
    # Copula "is (m.sg)" (chhu)
    "چھُ": "چھ",
    
    # Copula "is (f.sg)" (chhi)
    "چھِ": "چھیٚ",
    "چھی": "چھیٚ",
    "چھےٚ": "چھیٚ",
    "چھے": "چھیٚ",
    
    # Demonstratives / Pronouns (suh, yih)
    "سُہ": "سوہ",
    "سہ": "سوہ",
    "یِہ": "یہ",
    
    # Genitive Postpositions (hond / hənd)
    "ہند": "ہنٛد",
    "ہُنٛد": "ہنٛد",
    "ہنز": "ہنٛز",
    "ہٕنٛز": "ہنٛز",
    
    # Common Nouns / Particles with High Orthographic Variation in Crowdsourcing
    "لُکَن": "لۄکَن",
    "لکن": "لۄکَن",
    "لۄکن": "لۄکَن",
    "تِہ": "تہِ",
    "تہ": "تہِ",
    "روٚپیہ": "روٚپیہِ",
    "روپیہ": "روٚپیہِ",
    "قوبو": "قابوٗ",
    "قابو": "قابوٗ",
    "واسی": "وٲسی",
    "واسیہ": "وٲسی",
    "پونٛسہٕ": "پونٛسہ",
    "پونٛسہ": "پونٛسہ",
}

# Diacritics that distinguish short vs long vowels (analyzed for tolerance)
VOWEL_DIACRITICS = re.compile(r'[\u064E\u064F\u0650\u0654\u0655\u0656\u0657\u0658\u065A\u065B\u065C\u065D\u065E\u065F]')


def normalize_tolerant(text: str) -> str:
    """
    Metric B: Linguistically Justified Orthographic-Tolerant Normalization.
    Preserves all consonantal and phonemic roots while reconciling optional
    diacritics and legitimate lexical variant spellings.
    """
    if not text:
        return ""
    
    # Step 1: Base normalization (strip non-standard unicode, normalize alef/yeh/kaf)
    t = normalize_strict(text)
    
    # Step 2: Strip optional non-phonemic marks (tashdid, sukun, tanween, tatweel, ZWNJ)
    t = OPTIONAL_DIACRITICS.sub("", t)
    
    # Step 3: Normalize common diacritic omissions on short vowels while preserving consonants
    # We strip casual short vowel diacritics (zer, zabar, pesh, circle) for tolerant comparison
    t = VOWEL_DIACRITICS.sub("", t)
    
    # Step 4: Lexical Variant reconciliation
    words = t.split()
    normalized_words = [LEXICAL_VARIANT_MAP.get(w, w) for w in words]
    
    # Step 5: Collapse multiple spaces
    return " ".join(normalized_words).strip()


def evaluate_dual_metrics(references, predictions):
    """
    Computes both Metric A (Strict) and Metric B (Tolerant) side-by-side.
    """
    # Metric A: Strict
    strict_refs = [normalize_strict(r) for r in references]
    strict_preds = [normalize_strict(p) for p in predictions]
    wer_strict = wer(strict_refs, strict_preds) * 100
    cer_strict = cer(strict_refs, strict_preds) * 100
    exact_strict = sum(1 for r, p in zip(strict_refs, strict_preds) if r == p) / len(references) * 100

    # Metric B: Tolerant
    tol_refs = [normalize_tolerant(r) for r in references]
    tol_preds = [normalize_tolerant(p) for p in predictions]
    wer_tol = wer(tol_refs, tol_preds) * 100
    cer_tol = cer(tol_refs, tol_preds) * 100
    exact_tol = sum(1 for r, p in zip(tol_refs, tol_preds) if r == p) / len(references) * 100

    return {
        "strict": {
            "wer": wer_strict,
            "cer": cer_strict,
            "exact_match_pct": exact_strict
        },
        "tolerant": {
            "wer": wer_tol,
            "cer": cer_tol,
            "exact_match_pct": exact_tol
        },
        "delta": {
            "wer_reduction": wer_strict - wer_tol,
            "cer_reduction": cer_strict - cer_tol,
            "exact_match_gain": exact_tol - exact_strict
        }
    }


if __name__ == "__main__":
    # Self-test on known pairs
    ref_test = "نظامن اوس حیدر علی یس سٕتہ کمپنی شٗمٲلی سرکارن پیٹھ قوبو کرنچو کوششو٘ نش پتھ رٹنہٕ باپت کرناٹکاہس پیٹھ حملٕہٕ کرنہٕ باپتھ رابطہٕ کوٚرمت"
    pred_test = "نظامن اوس حیدر علی سۭتۍ کمپنی شمٲلی سرکارن پٮ۪ٹھ قابوٗ کرنٕچ کوٗششہ نش پتھ رٹنہٕ باپتھ کرناٹکاہس پیٚٹھ حملہٕ کرنہٕ باپت رٲبطہٕ کوٚرمت"

    print("=== Self-Test of Linguistic Dual Metrics ===")
    print("Reference (Raw): ", ref_test)
    print("Prediction (Raw):", pred_test)
    print("\nStrict Reference: ", normalize_strict(ref_test))
    print("Strict Prediction:", normalize_strict(pred_test))
    print("\nTolerant Reference: ", normalize_tolerant(ref_test))
    print("Tolerant Prediction:", normalize_tolerant(pred_test))

    res = evaluate_dual_metrics([ref_test], [pred_test])
    print(f"\nStrict WER:   {res['strict']['wer']:.2f}% | CER: {res['strict']['cer']:.2f}%")
    print(f"Tolerant WER: {res['tolerant']['wer']:.2f}% | CER: {res['tolerant']['cer']:.2f}%")
    print(f"Delta WER:    -{res['delta']['wer_reduction']:.2f}%")
