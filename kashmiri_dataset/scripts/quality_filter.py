#!/usr/bin/env python3
"""
Audio Quality Assessment & SNR Estimation Module for Kashmiri ASR
=================================================================
Calculates:
  - Energy-based Signal-to-Noise Ratio (SNR in dB)
  - Leading and trailing silence duration
  - Active speech ratio (Voice Activity Ratio)
  - Clipping ratio (samples within 1% of full scale)
  - Composite quality score (0.0 - 1.0)
  - Quality flags (too_short, too_long, low_snr, high_silence, clipped, ok)
"""

import math
import numpy as np
import soundfile as sf
from typing import Dict, List, Tuple

def estimate_snr_and_quality(
    audio_path: str,
    frame_ms: float = 25.0,
    min_duration: float = 0.5,
    max_duration: float = 25.0,
    snr_threshold_db: float = 12.0
) -> Dict[str, any]:
    """
    Analyzes an audio file and returns SNR, active speech stats, clipping, and quality flags.
    """
    flags = []
    try:
        data, sr = sf.read(audio_path)
    except Exception as e:
        return {
            'quality_score': 0.0,
            'snr_db': 0.0,
            'duration': 0.0,
            'clipping_ratio': 0.0,
            'speech_ratio': 0.0,
            'flags': ['corrupted_audio', str(e)],
            'is_clean': False
        }

    # Ensure 1D mono
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    duration = len(data) / sr

    if duration < min_duration:
        flags.append('too_short')
    if duration > max_duration:
        flags.append('too_long')

    # Clipping detection (|x| >= 0.99)
    clipping_count = np.sum(np.abs(data) >= 0.99)
    clipping_ratio = clipping_count / len(data) if len(data) > 0 else 0.0
    if clipping_ratio > 0.005:  # more than 0.5% samples clipped
        flags.append('clipped')

    # Frame-level energy analysis
    frame_size = int(sr * (frame_ms / 1000.0))
    if frame_size <= 0 or len(data) < frame_size:
        return {
            'quality_score': 0.1,
            'snr_db': 0.0,
            'duration': round(duration, 3),
            'clipping_ratio': round(clipping_ratio, 4),
            'speech_ratio': 0.0,
            'flags': flags or ['insufficient_frames'],
            'is_clean': False
        }

    # Split into non-overlapping frames
    num_frames = len(data) // frame_size
    frames = data[:num_frames * frame_size].reshape(num_frames, frame_size)
    frame_energies = np.sum(frames ** 2, axis=1) / frame_size + 1e-10

    # Noise floor: bottom 10th percentile energy
    noise_energy = np.percentile(frame_energies, 10)
    # Speech energy: top 20% energy
    speech_energy = np.percentile(frame_energies, 80)

    # Estimate SNR
    if noise_energy > 0:
        snr_linear = speech_energy / noise_energy
        snr_db = 10.0 * math.log10(max(1.0, snr_linear))
    else:
        snr_db = 35.0  # clean studio

    # Voice activity detection threshold: 3x noise floor
    vad_thresh = max(noise_energy * 3.0, 1e-5)
    active_frames = np.sum(frame_energies > vad_thresh)
    speech_ratio = active_frames / num_frames if num_frames > 0 else 0.0

    if snr_db < snr_threshold_db:
        flags.append('low_snr')

    if speech_ratio < 0.3:
        flags.append('high_silence')

    # Composite Quality Score (0.0 to 1.0)
    # Based on SNR (up to 30dB), speech ratio, and clipping penalty
    snr_factor = min(1.0, max(0.0, snr_db / 30.0))
    speech_factor = min(1.0, max(0.0, speech_ratio / 0.8))
    clip_penalty = max(0.0, 1.0 - (clipping_ratio * 20.0))
    duration_factor = 1.0 if (min_duration <= duration <= max_duration) else 0.5

    quality_score = round(0.5 * snr_factor + 0.3 * speech_factor + 0.2 * clip_penalty, 3) * duration_factor
    quality_score = max(0.0, min(1.0, round(quality_score, 3)))

    if not flags:
        flags.append('ok')

    is_clean = ('ok' in flags) and (quality_score >= 0.70) and (snr_db >= 15.0)

    return {
        'quality_score': quality_score,
        'snr_db': round(snr_db, 2),
        'duration': round(duration, 3),
        'clipping_ratio': round(clipping_ratio, 4),
        'speech_ratio': round(speech_ratio, 3),
        'flags': flags,
        'is_clean': is_clean
    }

if __name__ == '__main__':
    print("quality_filter.py module compiled successfully.")
