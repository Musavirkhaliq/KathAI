"""
Stage 22C Exact CTC Output Length Calculation Module.
=====================================================
Calculates exact mathematical output frame count of the 7-layer Wav2Vec2-XLS-R
convolutional feature extractor without heuristic approximations.
"""

from typing import Union, List
import numpy as np
import torch

# Exact architectural hyperparameters for facebook/wav2vec2-xls-r-300m
WAV2VEC2_CONV_KERNEL: List[int] = [10, 3, 3, 3, 3, 2, 2]
WAV2VEC2_CONV_STRIDE: List[int] = [5, 2, 2, 2, 2, 2, 2]
WAV2VEC2_RECEPTIVE_FIELD: int = 400
WAV2VEC2_TOTAL_STRIDE: int = 320


def compute_exact_ctc_length(input_samples: int) -> int:
    """
    Computes exact output frame length across all 7 Conv1d feature extractor layers.
    Formula per layer: L_out = floor((L_in - kernel) / stride) + 1
    Valid only for input_samples >= 400 (receptive field).
    """
    if input_samples < WAV2VEC2_RECEPTIVE_FIELD:
        return 0
    l = input_samples
    for k, s in zip(WAV2VEC2_CONV_KERNEL, WAV2VEC2_CONV_STRIDE):
        l = (l - k) // s + 1
    return int(l)


def compute_exact_ctc_lengths_batch(input_lengths: Union[List[int], np.ndarray, torch.Tensor]) -> List[int]:
    """
    Computes exact output frame lengths for a batch of sample counts.
    """
    return [compute_exact_ctc_length(int(n)) for n in input_lengths]


def slice_exact_logits(
    logits: Union[np.ndarray, torch.Tensor],
    num_samples: int
) -> Union[np.ndarray, torch.Tensor]:
    """
    Slices logits tensor (T, V) or (B, T, V) to the exact valid frame count,
    strictly removing all padded frames.
    """
    exact_t = compute_exact_ctc_length(num_samples)
    if exact_t == 0:
        raise ValueError(f"Audio sample count {num_samples} is below minimum receptive field (400 samples).")

    if logits.ndim == 2:
        return logits[:exact_t, :]
    elif logits.ndim == 3:
        return logits[:, :exact_t, :]
    else:
        raise ValueError(f"Unsupported logits ndim: {logits.ndim}")
