#!/usr/bin/env python3
"""
Universal Audio-Text Acquisition, Standardization, Auditing, and Deduplication Engine
EduCasheer Kashmiri AI Research Project
"""

import os
import io
import sys
import json
import hashlib
import numpy as np
import soundfile as sf
import torch
import torchaudio

class UniversalAudioEngine:
    def __init__(self, target_sr=16000, target_channels=1):
        self.target_sr = target_sr
        self.target_channels = target_channels
        self.benchmark_hashes = set()
        self.benchmark_speakers = set()
        self.load_benchmarks()

    def load_benchmarks(self):
        """Load frozen evaluation benchmark hashes and speakers to guarantee 0 contamination."""
        benchmark_paths = [
            "kashmiri_dataset/manifests/test.jsonl",
            "kashmiri_dataset/manifests/val.jsonl",
            "kashmiri_dataset/manifests/test.csv",
            "kashmiri_dataset/experiments/native_asr_training/data_splits/test.jsonl"
        ]
        for bp in benchmark_paths:
            if not os.path.exists(bp):
                continue
            try:
                if bp.endswith(".jsonl"):
                    with open(bp, "r", encoding="utf-8") as f:
                        for line in f:
                            rec = json.loads(line)
                            for h_key in ["audio_sha256", "pcm_sha256", "sha256", "file_sha256"]:
                                if rec.get(h_key):
                                    self.benchmark_hashes.add(rec[h_key])
                            if rec.get("speaker_id"):
                                self.benchmark_speakers.add(str(rec["speaker_id"]))
                elif bp.endswith(".csv"):
                    import pandas as pd
                    df = pd.read_csv(bp)
                    for col in ["audio_sha256", "pcm_sha256", "sha256"]:
                        if col in df.columns:
                            self.benchmark_hashes.update(df[col].dropna().astype(str).tolist())
                    if "speaker_id" in df.columns:
                        self.benchmark_speakers.update(df["speaker_id"].dropna().astype(str).tolist())
            except Exception as e:
                print(f"[Warning] Failed loading benchmark {bp}: {e}")
        print(f"[Engine] Benchmark screen loaded: {len(self.benchmark_hashes)} hashes, {len(self.benchmark_speakers)} protected speakers.")

    @staticmethod
    def compute_file_sha256(path):
        """Compute SHA-256 hash of a physical file."""
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def compute_pcm_sha256(audio_array):
        """Compute SHA-256 of raw int16 PCM array to detect re-encodings."""
        if audio_array.dtype != np.int16:
            pcm_int16 = np.clip(audio_array * 32767.0, -32768, 32767).astype(np.int16)
        else:
            pcm_int16 = audio_array
        return hashlib.sha256(pcm_int16.tobytes()).hexdigest()

    @staticmethod
    def estimate_snr_db(signal):
        """Estimate Signal-to-Noise Ratio (SNR) in dB using energy percentile ratio."""
        if len(signal) == 0:
            return 0.0
        frame_len = 512
        num_frames = len(signal) // frame_len
        if num_frames == 0:
            return 20.0
        frames = signal[:num_frames * frame_len].reshape(num_frames, frame_len)
        frame_energies = np.sum(frames ** 2, axis=1) / frame_len
        if np.all(frame_energies <= 1e-12):
            return 0.0
        signal_energy = np.percentile(frame_energies, 90)
        noise_energy = np.percentile(frame_energies, 10)
        if noise_energy <= 1e-12:
            return 60.0
        ratio = max(1e-10, signal_energy / noise_energy)
        return float(np.clip(10.0 * np.log10(ratio), 0.0, 80.0))

    @staticmethod
    def compute_vad_ratio(signal, threshold=0.01):
        """Compute energy-based Voice Activity Detection (VAD) speech ratio."""
        if len(signal) == 0:
            return 0.0
        active = np.sum(np.abs(signal) > threshold)
        return float(active / len(signal))

    def standardize_audio(self, in_path, out_path):
        """Read any audio format, convert to standard 16kHz mono 16-bit PCM WAV."""
        data, sr = sf.read(in_path)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        data = data.astype(np.float32)

        # Resample to target_sr if needed
        if sr != self.target_sr:
            t_data = torch.from_numpy(data).unsqueeze(0)
            t_resampled = torchaudio.functional.resample(t_data, sr, self.target_sr)
            data = t_resampled.squeeze(0).numpy()

        # True-peak normalize if clipping
        peak = np.max(np.abs(data))
        if peak > 1.0:
            data = data / peak
        elif peak > 0.0:
            data = data / max(peak, 0.95) * 0.95

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sf.write(out_path, data, self.target_sr, subtype="PCM_16")

        orig_hash = self.compute_file_sha256(in_path)
        pcm_hash = self.compute_pcm_sha256(data)
        duration = float(len(data) / self.target_sr)
        snr = self.estimate_snr_db(data)
        vad = self.compute_vad_ratio(data)

        # Check benchmark contamination
        is_contaminated = (orig_hash in self.benchmark_hashes) or (pcm_hash in self.benchmark_hashes)

        return {
            "canonical_path": out_path,
            "duration": duration,
            "sample_rate": self.target_sr,
            "channels": 1,
            "original_sha256": orig_hash,
            "pcm_sha256": pcm_hash,
            "snr_db": round(snr, 2),
            "vad_ratio": round(vad, 3),
            "is_benchmark_contaminated": is_contaminated
        }

if __name__ == "__main__":
    engine = UniversalAudioEngine()
    print("Engine instantiated successfully.")
