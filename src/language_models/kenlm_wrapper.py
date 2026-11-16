"""
KenLM Language Model Wrapper Module.
====================================
Provides n-gram scoring, sentence probability computation,
and perplexity calculation for Kashmiri language models.
"""

import math
from pathlib import Path
from typing import Optional, Union, List
from src.shared.logging import get_logger

logger = get_logger("kenlm_wrapper")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LM_PATH = PROJECT_ROOT / "kashmiri_dataset/experiments/native_asr_training/experiments/stage7_ctc_lm_decoding/language_models/lm_word_4gram_domain_d.arpa"

try:
    import kenlm
    KENLM_AVAILABLE = True
except ImportError:
    KENLM_AVAILABLE = False


class KenLMWrapper:
    """
    Wrapper for KenLM ARPA / Binary models.
    """
    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self.model_path = Path(model_path) if model_path else DEFAULT_LM_PATH
        self.model = None

        if not self.model_path.exists():
            logger.warning(f"KenLM model file not found at: {self.model_path}")
            return

        if KENLM_AVAILABLE:
            try:
                self.model = kenlm.Model(str(self.model_path))
                logger.info(f"Loaded KenLM model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load KenLM model with kenlm library: {e}")
        else:
            logger.info("kenlm Python package not installed; n-gram perplexity scoring disabled.")

    def score_sentence(self, sentence: str, bos: bool = True, eos: bool = True) -> float:
        """
        Returns log10 probability of sentence.
        """
        if self.model is None:
            return 0.0
        return float(self.model.score(sentence, bos=bos, eos=eos))

    def compute_perplexity(self, sentence: str) -> float:
        """
        Computes sentence perplexity: 10^(-score / num_words).
        """
        if self.model is None:
            return float("inf")
        words = sentence.strip().split()
        if not words:
            return float("inf")
        score = self.score_sentence(sentence)
        # Length including EOS token
        length = len(words) + 1
        return float(10.0 ** (-score / length))
