"""
Kashmiri ASR Model Loader & Registry.
=====================================
Manages checkpoint resolution across project champions (E22-01, B-02C, B-02A, B-01).
"""

from pathlib import Path
from typing import Union, Tuple, Optional
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from src.shared.logging import get_logger
from src.shared.device import get_optimal_device

logger = get_logger("asr_model")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Checkpoint Registry (ranked in descending order of empirical accuracy)
CHECKPOINT_REGISTRY = {
    "E22-01": PROJECT_ROOT / "kashmiri_dataset/experiments/stage22_data_expanded_training/checkpoints/E22-01/best_model",
    "B-02C": PROJECT_ROOT / "kashmiri_dataset/experiments/native_asr_training/experiments/stage9_b02_acoustic_comparison/B-02C/checkpoints/best_model",
    "B-02A": PROJECT_ROOT / "kashmiri_dataset/experiments/native_asr_training/experiments/stage9_b02_acoustic_comparison/B-02A/checkpoints/best_model",
    "B-01": PROJECT_ROOT / "kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc/best_model",
}

def resolve_default_checkpoint() -> Path:
    """
    Returns path to the highest-ranking existing champion checkpoint.
    """
    for name, p in CHECKPOINT_REGISTRY.items():
        if p.exists() and (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists():
            return p
    # Fallback to first existing directory
    for name, p in CHECKPOINT_REGISTRY.items():
        if p.exists():
            return p
    return CHECKPOINT_REGISTRY["E22-01"]

DEFAULT_MODEL_DIR = resolve_default_checkpoint()


def load_asr_model(
    model_dir: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[Wav2Vec2ForCTC, Wav2Vec2Processor, torch.device]:
    """
    Loads processor and model from checkpoint directory onto optimal device.
    """
    ckpt_path = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ASR Checkpoint directory not found: {ckpt_path}")

    dev = torch.device(device) if device else get_optimal_device()

    logger.info(f"Loading ASR model from {ckpt_path} onto {dev}...")
    processor = Wav2Vec2Processor.from_pretrained(str(ckpt_path))
    model = Wav2Vec2ForCTC.from_pretrained(str(ckpt_path))
    model.to(dev)
    model.eval()
    logger.info("ASR Model loaded successfully.")

    return model, processor, dev
