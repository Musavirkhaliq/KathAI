"""
Data Collection & Audio Harvesting Base Interface.
==================================================
Defines schemas, ingestion contracts, and quarantine filters
for raw speech and text ingestion into the EduCasheer Kashmiri ecosystem.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from src.shared.logging import get_logger

logger = get_logger("data_collection")


class BaseCollector(ABC):
    """
    Abstract Base Class for data ingestion sources.
    """
    def __init__(self, source_name: str, output_dir: Path):
        self.source_name = source_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def harvest(self) -> List[Dict[str, Any]]:
        """
        Executes harvesting pipeline and yields record dictionaries.
        """
        pass

    def validate_record(self, record: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validates record against ingestion constraints:
        - audio_path exists
        - duration in [0.5s, 35.0s]
        - text non-empty
        """
        audio_p = record.get("audio_path", record.get("canonical_audio_path"))
        if not audio_p or not Path(audio_p).exists():
            return False, "audio_file_missing"

        dur = record.get("duration", record.get("duration_seconds", 0.0))
        try:
            dur = float(dur)
            if dur < 0.5:
                return False, "too_short (<0.5s)"
            if dur > 35.0:
                return False, "too_long (>35.0s)"
        except (ValueError, TypeError):
            return False, "invalid_duration"

        text = record.get("text", record.get("transcript_normalized", ""))
        if not text or not text.strip():
            return False, "empty_transcript"

        return True, None
