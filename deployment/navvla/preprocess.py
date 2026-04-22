"""Preprocessing and config helpers for deployment."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import yaml

# OmniVLA utilities use this std value internally. It is treated as fixed.
FIXED_NORMALIZE_STD = (0.229, 0.224, 0.225)


def load_yaml(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def ensure_image_size(size_like: object, key_name: str) -> Tuple[int, int]:
    if not isinstance(size_like, list) or len(size_like) != 2:
        raise ValueError(f"{key_name} must be [width, height]")
    width, height = int(size_like[0]), int(size_like[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"{key_name} must be positive")
    return width, height


def ensure_normalize_mean(mean_like: object) -> Tuple[float, float, float]:
    if not isinstance(mean_like, list) or len(mean_like) != 3:
        raise ValueError("normalize_mean must be a list of 3 floats")
    return float(mean_like[0]), float(mean_like[1]), float(mean_like[2])


def build_mask(size: Tuple[int, int], use_mask: bool, mask_path: str) -> np.ndarray:
    width, height = size
    if not use_mask:
        return np.ones((height, width, 3), dtype=np.float32)

    if not mask_path:
        raise ValueError("mask_path is empty while use_mask=true")

    mask_file = Path(mask_path)
    if not mask_file.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_file}")

    loaded = np.load(mask_file)
    if loaded.ndim == 2:
        loaded = np.repeat(loaded[:, :, None], 3, axis=2)

    if loaded.shape != (height, width, 3):
        raise ValueError(
            f"Mask shape mismatch: expected {(height, width, 3)}, got {loaded.shape}"
        )

    return loaded.astype(np.float32)
