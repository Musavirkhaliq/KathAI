"""
Dataset Manifest Ingestion & Partitioning Module.
=================================================
Parses, audits, and validates JSONL manifests, split partitions,
and audio integrity flags across Kashmiri speech datasets.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Iterator, Optional, Union
from src.shared.logging import get_logger

logger = get_logger("dataset_loader")


def iter_manifest(manifest_path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    """
    Yields parsed records from a JSONL manifest.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                yield json.loads(line_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Error decoding line {line_num} in {path}: {e}")
                continue


def load_manifest(manifest_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Loads all records from a JSONL manifest into memory.
    """
    return list(iter_manifest(manifest_path))


def summarize_manifest(manifest_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Calculates exact aggregate statistics for a given JSONL manifest.
    """
    total_records = 0
    total_duration_sec = 0.0
    audio_paths = set()
    speakers = set()
    sources = set()

    for item in iter_manifest(manifest_path):
        total_records += 1
        dur = item.get("duration_seconds", item.get("duration", item.get("audio_duration", 0.0)))
        if dur is not None:
            try:
                total_duration_sec += float(dur)
            except (ValueError, TypeError):
                pass

        audio = item.get("canonical_audio_path", item.get("audio_path", item.get("audio_filepath", "")))
        if audio:
            audio_paths.add(audio)

        spk = item.get("speaker_id", item.get("speaker", ""))
        if spk:
            speakers.add(spk)

        src = item.get("source_dataset", item.get("source", ""))
        if src:
            sources.add(src)

    return {
        "manifest_path": str(manifest_path),
        "total_records": total_records,
        "total_duration_hours": round(total_duration_sec / 3600.0, 2),
        "total_duration_seconds": round(total_duration_sec, 2),
        "unique_audio_files": len(audio_paths),
        "unique_speakers": len(speakers),
        "distinct_sources": sorted(list(sources))
    }
