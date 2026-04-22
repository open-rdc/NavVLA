"""Dataset utilities for OmniVLA-edge fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch.utils.data import Dataset


REQUIRED_KEYS = (
    "obs_images",
    "goal_pose",
    "map_images",
    "goal_image",
    "goal_mask",
    "feat_text",
    "current_img",
    "actions",
)

KEY_ALIASES = {
    "cur_image": "obs_images",
    "goal_image_8": "goal_image",
    "modality_id": "goal_mask",
    "lan_prompt_feature": "feat_text",
}


class EdgeTensorDataset(Dataset):
    """Load preprocessed OmniVLA-edge tensor samples from .pt/.pth files."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")

        self.sample_paths = sorted(
            path
            for pattern in ("*.pt", "*.pth")
            for path in self.data_dir.glob(pattern)
        )
        if not self.sample_paths:
            raise FileNotFoundError(f"No .pt/.pth samples found in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = torch.load(self.sample_paths[index], map_location="cpu")
        if not isinstance(sample, dict):
            raise ValueError(f"Sample must be a dict: {self.sample_paths[index]}")

        normalized = dict(sample)
        for source_key, target_key in KEY_ALIASES.items():
            if source_key in normalized and target_key not in normalized:
                normalized[target_key] = normalized[source_key]

        missing = [key for key in REQUIRED_KEYS if key not in normalized]
        if missing:
            raise ValueError(f"Missing sample keys {missing}: {self.sample_paths[index]}")

        return {key: torch.as_tensor(normalized[key]) for key in REQUIRED_KEYS}


def collate_edge_samples(
    samples: Iterable[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    batch: Dict[str, List[torch.Tensor]] = {key: [] for key in REQUIRED_KEYS}
    for sample in samples:
        for key in REQUIRED_KEYS:
            value = sample[key]
            if value.dim() > 0 and value.shape[0] == 1:
                value = value.squeeze(0)
            batch[key].append(value)
    return {key: torch.stack(values, dim=0) for key, values in batch.items()}
