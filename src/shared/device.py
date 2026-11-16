"""
PyTorch Device & Hardware Resource Manager.
"""

import torch
from typing import Dict, Any
from src.shared.logging import get_logger

logger = get_logger("device_manager")

def get_optimal_device(preferred_device: str = "auto") -> torch.device:
    """
    Selects optimal computation device (cuda:0, mps, cpu).
    """
    if preferred_device != "auto":
        return torch.device(preferred_device)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Using CUDA GPU: {device_name}")
        return device
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Using Apple Silicon MPS acceleration.")
        return torch.device("mps")
    else:
        logger.info("Using CPU computation.")
        return torch.device("cpu")

def get_gpu_memory_stats(device: torch.device) -> Dict[str, float]:
    """
    Returns GPU memory usage in Megabytes.
    """
    if device.type != "cuda":
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}
    return {
        "allocated_mb": round(torch.cuda.memory_allocated(device) / (1024 ** 2), 2),
        "reserved_mb": round(torch.cuda.memory_reserved(device) / (1024 ** 2), 2),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2),
    }
