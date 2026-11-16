"""
Audio Chunking & Windowing Strategies for Long-Form Speech ASR.
===============================================================
Implements certified Method C2 (Context-Preserving Central Frame Extraction)
and uninterrupted inference to prevent OOM errors and border artifacts.
"""

from typing import Tuple, List, Optional
import numpy as np
import torch
from src.decoding.ctc_slice import compute_exact_ctc_length, slice_exact_logits


def chunk_audio_c2(
    wav: np.ndarray,
    core_samples: int = 18 * 16000,
    context_samples: int = 2 * 16000
) -> List[Tuple[np.ndarray, int, int]]:
    """
    Partitions long audio into windows with context pads before and after.
    Returns list of tuples: (padded_chunk, core_start_idx, core_end_idx)
    where core_start_idx and core_end_idx indicate the valid central region within the chunk.
    """
    total_samples = len(wav)
    if total_samples <= core_samples + context_samples:
        return [(wav, 0, total_samples)]

    chunks = []
    for s_core in range(0, total_samples, core_samples):
        e_core = min(s_core + core_samples, total_samples)
        s_ctx = max(0, s_core - context_samples)
        e_ctx = min(total_samples, e_core + context_samples)

        chunk = wav[s_ctx:e_ctx]
        # Valid region relative to chunk start
        valid_start = s_core - s_ctx
        valid_end = valid_start + (e_core - s_core)
        chunks.append((chunk, valid_start, valid_end))

    return chunks
