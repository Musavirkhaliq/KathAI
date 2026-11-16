"""
Audio Preprocessing and Signal Conditioning Module.
===================================================
Provides robust audio loading, 16 kHz resampling, mono conversion,
peak normalization, and dynamic range validation.
"""

from pathlib import Path
from typing import Union, Tuple, Optional
import numpy as np
import soundfile as sf
import scipy.signal

from src.shared.logging import get_logger

logger = get_logger("audio_processor")


def load_audio(
    file_path: Union[str, Path],
    target_sample_rate: int = 16000,
    to_mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Loads audio file from disk, converts to mono if specified, and resamples to target_sample_rate.
    Returns: (audio_array: np.ndarray [float32], sample_rate: int)
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    data, sample_rate = sf.read(str(path))

    # Convert to float32
    if data.dtype != np.float32:
        data = data.astype(np.float32)

    # Convert multi-channel to mono
    if to_mono and data.ndim > 1:
        data = np.mean(data, axis=1)

    # Resample if sample rate differs
    if sample_rate != target_sample_rate:
        num_target_samples = int(round(len(data) * float(target_sample_rate) / float(sample_rate)))
        data = scipy.signal.resample(data, num_target_samples).astype(np.float32)
        sample_rate = target_sample_rate

    return data, sample_rate


def normalize_audio(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """
    Peak normalizes audio array to target_peak (default 0.95) to prevent clipping.
    """
    max_val = np.max(np.abs(audio))
    if max_val > 1e-6:
        return (audio / max_val * target_peak).astype(np.float32)
    return audio
