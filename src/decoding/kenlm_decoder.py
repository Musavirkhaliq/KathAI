"""
Calibrated KenLM CTC Beam Search Decoder Module.
================================================
Wraps pyctcdecode with acoustic temperature scaling (T=1.08),
language model weight (alpha), word insertion bonus (beta), and vocabulary mapping.
"""

from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import numpy as np
import torch

try:
    from pyctcdecode import build_ctcdecoder
    PYCTCDECODE_AVAILABLE = True
except ImportError:
    PYCTCDECODE_AVAILABLE = False


class CalibratedKenLMDecoder:
    """
    KenLM Beam Search Decoder with calibrated acoustic temperature scaling.
    """
    def __init__(
        self,
        vocab_list: List[str],
        kenlm_path: Optional[Union[str, Path]] = None,
        unigrams_path: Optional[Union[str, Path]] = None,
        alpha: float = 0.5,
        beta: float = 1.5,
        temperature: float = 1.08,
        beam_width: int = 100,
        beam_prune_logp: float = -10.0,
        token_min_logp: float = -5.0
    ):
        if not PYCTCDECODE_AVAILABLE:
            raise ImportError("pyctcdecode is required for CalibratedKenLMDecoder. Please install pyctcdecode.")

        self.vocab_list = vocab_list
        self.kenlm_path = str(kenlm_path) if kenlm_path else None
        self.unigrams_path = str(unigrams_path) if unigrams_path else None
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.beam_width = beam_width
        self.beam_prune_logp = beam_prune_logp
        self.token_min_logp = token_min_logp

        unigram_list = None
        if self.unigrams_path and Path(self.unigrams_path).exists():
            with open(self.unigrams_path, "r", encoding="utf-8") as f:
                unigram_list = [line.strip().split("\t")[0] for line in f if line.strip()]

        self.decoder = build_ctcdecoder(
            labels=vocab_list,
            kenlm_model_path=self.kenlm_path,
            unigrams=unigram_list,
            alpha=self.alpha,
            beta=self.beta
        )

    def decode(self, logits: Union[np.ndarray, torch.Tensor]) -> str:
        """
        Decodes logits (T, V) into Kashmiri text string using beam search.
        Applies acoustic temperature scaling: logits / T.
        """
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()

        if logits.ndim == 3:
            logits = logits[0]

        # Apply acoustic temperature scaling
        scaled_logits = logits / self.temperature

        # pyctcdecode expects unnormalized logits or log-probabilities
        transcription = self.decoder.decode(
            scaled_logits,
            beam_width=self.beam_width,
            beam_prune_logp=self.beam_prune_logp,
            token_min_logp=self.token_min_logp
        )
        return transcription.strip()

    def decode_beams(self, logits: Union[np.ndarray, torch.Tensor], n_best: int = 5) -> List[Dict[str, Any]]:
        """
        Returns top N beam candidate transcriptions and scores.
        """
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()

        if logits.ndim == 3:
            logits = logits[0]

        scaled_logits = logits / self.temperature

        beams = self.decoder.decode_beams(
            scaled_logits,
            beam_width=self.beam_width,
            beam_prune_logp=self.beam_prune_logp,
            token_min_logp=self.token_min_logp
        )

        results = []
        for b in beams[:n_best]:
            results.append({
                "text": b[0].strip(),
                "score": float(b[3]) if len(b) > 3 else 0.0,
                "lm_score": float(b[4]) if len(b) > 4 else 0.0
            })
        return results
