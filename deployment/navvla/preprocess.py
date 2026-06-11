from __future__ import annotations

from pathlib import Path
import sys
from typing import TYPE_CHECKING
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import yaml
from PIL import Image as PILImage

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT_CANDIDATES = [
    _THIS_FILE.parents[2],
    _THIS_FILE.parents[4] / "src" / "NavVLA" if len(_THIS_FILE.parents) > 4 else None,
]
for _repo_root in reversed([path for path in _REPO_ROOT_CANDIDATES if path is not None and (path / "OmniVLA").exists()]):
    for _path in (_repo_root, _repo_root / "OmniVLA", _repo_root / "OmniVLA" / "inference"):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))

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


def image_msg_to_bgr(msg: "Image") -> np.ndarray:
    encoding = msg.encoding.lower()
    channels_by_encoding = {
        "bgr8": 3,
        "rgb8": 3,
        "bgra8": 4,
        "rgba8": 4,
        "mono8": 1,
        "8uc1": 1,
        "8uc3": 3,
        "8uc4": 4,
        "yuv422_yuy2": 2,
        "yuyv": 2,
        "yuy2": 2,
    }
    if encoding not in channels_by_encoding:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")

    channels = channels_by_encoding[encoding]
    row = np.frombuffer(msg.data, dtype=np.uint8).reshape(int(msg.height), int(msg.step))
    image = row[:, : int(msg.width) * channels].reshape(int(msg.height), int(msg.width), channels)

    if encoding in ("bgr8", "8uc3"):
        return image.copy()
    if encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if encoding in ("yuv422_yuy2", "yuyv", "yuy2"):
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUY2)

    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def image_to_cv2(msg: "Image", output_size: Tuple[int, int]) -> np.ndarray:
    frame = image_msg_to_bgr(msg)

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
