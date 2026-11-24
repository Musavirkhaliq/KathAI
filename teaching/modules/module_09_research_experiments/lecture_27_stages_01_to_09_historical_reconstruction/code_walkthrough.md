# Code Walkthrough: Forensic Log and Hash Auditing

---

## 1. Checkpoint SHA-256 Hash Auditing Script

```python
import hashlib
from pathlib import Path

def compute_checkpoint_hash(checkpoint_path: str, chunk_size=65536) -> str:
    sha256 = hashlib.sha256()
    with open(checkpoint_path, 'rb') as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()
```
