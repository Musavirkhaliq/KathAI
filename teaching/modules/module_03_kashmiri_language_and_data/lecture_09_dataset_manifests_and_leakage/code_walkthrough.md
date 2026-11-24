# Code Walkthrough: Manifest Validation and Leakage Detection

---

## 1. Inspecting the Leakage Detector (`src/data/manifest.py`)

```python
import json
from pathlib import Path

def verify_zero_speaker_leakage(train_manifest: str, test_manifest: str) -> dict:
    # [Docstring]
    Audits two manifests to mathematically guarantee zero speaker overlap.
    # [Docstring]
    def load_speakers(path):
        speakers = set()
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    speakers.add(entry["speaker_id"])
        return speakers
        
    train_speakers = load_speakers(train_manifest)
    test_speakers = load_speakers(test_manifest)
    
    overlap = train_speakers.intersection(test_speakers)
    is_clean = (len(overlap) == 0)
    
    return {
        "is_clean": is_clean,
        "train_speaker_count": len(train_speakers),
        "test_speaker_count": len(test_speakers),
        "overlapping_speakers": list(overlap),
        "contamination_rate": len(overlap) / max(1, len(test_speakers))
    }
```
