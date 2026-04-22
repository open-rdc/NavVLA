from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import yaml
from PIL import Image as PILImage

from OmniVLA.inference.utils_policy import transform_images_PIL_mask, transform_images_map

if TYPE_CHECKING:
    from sensor_msgs.msg import Image

# OmniVLA utilities use this std value internally. It is treated as fixed.
FIXED_NORMALIZE_STD = (0.229, 0.224, 0.225)


def load_yaml(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


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
    return loaded.astype(np.float32)


def image_to_cv2(msg: "Image", output_size: Tuple[int, int]) -> np.ndarray:
    frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((int(msg.height), int(msg.width), 3))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    side = min(frame.shape[0], frame.shape[1])
    offset_y = (frame.shape[0] - side) // 2
    offset_x = (frame.shape[1] - side) // 2
    square_image = frame[offset_y : offset_y + side, offset_x : offset_x + side]

    target_w, target_h = int(output_size[0]), int(output_size[1])
    return cv2.resize(square_image, (target_w, target_h), interpolation=cv2.INTER_AREA)


def build_omnivla_edge_inputs(
    context_queue: List[PILImage.Image],
    current_image: PILImage.Image,
    mask_obs: np.ndarray,
    mask_clip: np.ndarray,
    satellite_current: PILImage.Image,
    satellite_goal: PILImage.Image,
    clip_size: Tuple[int, int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    obs_images = transform_images_PIL_mask(list(context_queue), mask_obs).to(device)
    split_obs = torch.split(obs_images, 3, dim=1)
    obs_image_cur = split_obs[-1].to(device)
    obs_images = torch.cat(split_obs, dim=1).to(device)

    cur_large_img = transform_images_PIL_mask(current_image.resize(clip_size), mask_clip).to(device)
    map_images = torch.cat(
        (
            transform_images_map(satellite_current).to(device),
            transform_images_map(satellite_goal).to(device),
            obs_image_cur,
        ),
        axis=1,
    )
    return obs_images, map_images, cur_large_img
