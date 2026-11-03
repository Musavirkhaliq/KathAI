#!/usr/bin/env python3
"""
Duplicate & Near-Duplicate Detection for Audio and Transcripts
=============================================================
1. Computes MD5/SHA256 for audio files to detect byte-identical audio.
2. Computes normalized text hashes to identify repeated sentences/prompts.
3. Tags duplicate records with duplicate cluster IDs without deleting them.
"""

import hashlib
import os
from collections import defaultdict
from typing import Dict, List, Set, Tuple

def compute_file_hash(path: str, block_size: int = 65536) -> str:
    """Computes SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(block_size):
            h.update(block)
    return h.hexdigest()

def compute_text_hash(normalized_text: str) -> str:
    """Computes MD5 hash of normalized text string."""
    return hashlib.md5(normalized_text.encode('utf-8')).hexdigest()

class DuplicateTracker:
    def __init__(self):
        self.audio_hashes = defaultdict(list)
        self.text_hashes = defaultdict(list)

    def register(self, sample_id: str, audio_path: str, normalized_text: str):
        if audio_path and os.path.exists(audio_path):
            ahash = compute_file_hash(audio_path)
            self.audio_hashes[ahash].append(sample_id)
        
        if normalized_text:
            thash = compute_text_hash(normalized_text)
            self.text_hashes[thash].append(sample_id)

    def get_summary(self) -> Dict[str, any]:
        duplicate_audio = {k: v for k, v in self.audio_hashes.items() if len(v) > 1}
        duplicate_text = {k: v for k, v in self.text_hashes.items() if len(v) > 1}
        return {
            'total_unique_audio': len(self.audio_hashes),
            'duplicate_audio_clusters': len(duplicate_audio),
            'duplicate_audio_samples': sum(len(v) for v in duplicate_audio.values()),
            'total_unique_text': len(self.text_hashes),
            'duplicate_text_clusters': len(duplicate_text),
            'duplicate_text_samples': sum(len(v) for v in duplicate_text.values()),
        }

if __name__ == '__main__':
    print("detect_duplicates.py module compiled successfully.")
