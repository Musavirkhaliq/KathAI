# Code Walkthrough: Production Logging and ASR Experiment Harness

---

## 1. Production Architecture Overview

In the EduCasheer repository, every experiment is tracked via a deterministic, multi-channel logging and configuration system located in `src/shared/logging.py`.

Before running training or inference, speech systems must establish an auditable trace capturing:
1. Process IDs and hardware device allocations (GPU / CPU).
2. Git commit hashes and branch states to guarantee reproducibility.
3. Audio tensor shapes, sampling rates, and memory layout.

---

## 2. Line-by-Line Code Inspection: `src/shared/logging.py`

```python
# File: src/shared/logging.py
import logging
import sys
import os
from typing import Optional

def setup_logger(
    name: str = "educasheer",
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configures a deterministic, high-resolution research logger.
    
    Args:
        name: Hierarchical module name (e.g. 'asr.trainer', 'data.curator').
        level: Minimum logging threshold (default: INFO).
        log_file: Optional path to an on-disk log artifact for audit trails.
        
    Returns:
        A fully configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate handlers if setup_logger is called repeatedly
    if not logger.handlers:
        # 1. Console Stream Handler (stdout)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Certified forensic formatting string:
        # Includes timestamp, loglevel, process ID, logger name, source file, line number
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [PID:%(process)d] %(name)s (%(filename)s:%(lineno)d): %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 2. File Handler (if audit trail path provided)
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
    return logger
```

### Key Engineering Invariants
- `sys.stdout` vs `sys.stderr`: Line 24 directs logs to standard output rather than stderr, ensuring that batch cluster runners (Slurm/PBS) do not flag informational logs as pipeline errors.
- UTF-8 Encoding: Line 39 explicitly enforces `encoding="utf-8"`. In Kashmiri ASR, loggers frequently print Perso-Arabic text strings (e.g. references and predictions). On non-UTF-8 terminals, printing unencoded Arabic characters causes fatal `UnicodeEncodeError` exceptions.

---

## 3. Audio Tensor Invariant Verification Harness

When an audio batch enters the acoustic pipeline, we execute an automated tensor validation pass:

```python
import torch

def validate_audio_batch_invariants(waveform: torch.Tensor, expected_sr: int = 16000) -> dict:
    """
    Verifies physical and numerical invariants of raw audio tensors before modeling.
    """
    invariants = {
        "is_float32": waveform.dtype == torch.float32,
        "is_finite": torch.isfinite(waveform).all().item(),
        "is_mono": waveform.ndim == 2, # Expected shape: (Batch, Samples)
        "within_dynamic_range": (waveform.abs().max() <= 1.05).item(),
        "no_dc_offset": (waveform.mean().abs() < 0.05).item()
    }
    return invariants
```

If any invariant evaluates to `False`, the audio batch is immediately rejected before reaching the GPU memory buffers.
