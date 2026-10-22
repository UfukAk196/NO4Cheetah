"""Small helpers: seeds, device, config loading.

Checkpoints and parameter counts come from neuraloperator
(``BaseModel.save_checkpoint`` / ``from_checkpoint``, ``neuralop.utils.count_model_params``).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: str | Path) -> dict:
    """Reads a YAML config with ``yaml.safe_load``."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
