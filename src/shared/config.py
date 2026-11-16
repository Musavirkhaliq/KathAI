"""
Configuration Parser & Validator.
"""

import json
from pathlib import Path
from typing import Dict, Any, Union
import yaml

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Loads configuration from YAML or JSON file.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        if path.suffix.lower() in (".yaml", ".yml"):
            return yaml.safe_load(f) or {}
        elif path.suffix.lower() == ".json":
            return json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file extension: {path.suffix}")
