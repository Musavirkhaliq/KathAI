"""
Standardized ASR Evaluation Metrics Module.
===========================================
Computes exact Levenshtein Word Error Rate (WER) and Character Error Rate (CER),
substitutions, insertions, deletions, sentence accuracy, and corpus-level averages.
"""

from typing import List, Dict, Any, Union, Tuple
import numpy as np
from src.normalization.normalizer import normalize_for_wer_evaluation


def levenshtein_distance(seq1: Union[str, List[str]], seq2: Union[str, List[str]]) -> int:
    """
    Computes exact Levenshtein edit distance using dynamic programming.
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y), dtype=np.int32)

    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = matrix[x - 1, y - 1]
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,      # deletion
                    matrix[x - 1, y - 1] + 1,  # substitution
                    matrix[x, y - 1] + 1       # insertion
                )
    return int(matrix[size_x - 1, size_y - 1])


def calculate_wer_and_cer(ref: str, hyp: str, normalize: bool = False) -> Dict[str, Any]:
    """
    Calculates WER and CER for a single reference and hypothesis pair.
    """
    if normalize:
        ref = normalize_for_wer_evaluation(ref)
        hyp = normalize_for_wer_evaluation(hyp)

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

    # Character level
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
        "wer": float(wer),
        "cer": float(cer),
        "word_dist": int(word_dist),
        "char_dist": int(char_dist),
        "ref_words": int(ref_word_count),
        "ref_chars": int(ref_char_count),
        "exact_match": bool(ref.strip() == hyp.strip())
    }


def evaluate_predictions(
    references: List[str],
    hypotheses: List[str],
    normalize: bool = False
) -> Dict[str, Any]:
    """
    Computes corpus-level aggregated WER and CER across aligned lists.
    """
    if len(references) != len(hypotheses):
        raise ValueError(f"Mismatch in references ({len(references)}) and hypotheses ({len(hypotheses)}) lengths.")

    total_word_dist = 0
    total_ref_words = 0
    total_char_dist = 0
    total_ref_chars = 0
    exact_matches = 0

    for ref, hyp in zip(references, hypotheses):
        res = calculate_wer_and_cer(ref, hyp, normalize=normalize)
        total_word_dist += res["word_dist"]
        total_ref_words += res["ref_words"]
        total_char_dist += res["char_dist"]
        total_ref_chars += res["ref_chars"]
        if res["exact_match"]:
            exact_matches += 1

    total_samples = len(references)
    corpus_wer = (total_word_dist / total_ref_words) if total_ref_words > 0 else 0.0
    corpus_cer = (total_char_dist / total_ref_chars) if total_ref_chars > 0 else 0.0
    accuracy = (exact_matches / total_samples) if total_samples > 0 else 0.0

    return {
        "total_samples": total_samples,
        "corpus_wer": round(corpus_wer * 100.0, 2),
        "corpus_cer": round(corpus_cer * 100.0, 2),
        "exact_match_ratio": round(accuracy * 100.0, 2),
        "total_word_errors": total_word_dist,
        "total_reference_words": total_ref_words,
        "total_char_errors": total_char_dist,
        "total_reference_chars": total_ref_chars
    }
