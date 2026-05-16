"""Dataset utilities for OmniVLA-edge fine-tuning."""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image

DEFAULT_DUMMY_LANGUAGE = "No language instruction"
DEFAULT_CLIP_MODEL = "ViT-B/32"

REQUIRED_KEYS = (
    "obs_images",
    "goal_pose",
    "map_images",
    "goal_image",
    "goal_mask",
    "feat_text",
    "current_img",
    "actions",
    "dist_to_goal",
)

KEY_ALIASES = {
    "cur_image": "obs_images",
    "goal_image_8": "goal_image",
    "modality_id": "goal_mask",
    "lan_prompt_feature": "feat_text",
}

MODALITY_USES = {
    0: {"satellite"},
    1: {"pose", "satellite"},
    2: {"satellite", "image"},
    3: {"pose", "satellite", "image", "language"},
    4: {"pose"},
    5: {"pose", "image"},
    6: {"image"},
    7: {"language"},
    8: {"language", "pose"},
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


class EdgeNavigationDataset(Dataset):
    """Load raw navigation trajectories for OmniVLA-edge fine-tuning.

    Expected data layout:
        data_folder/traj_name/traj_data.pkl
        data_folder/traj_name/0.jpg
        data_folder/traj_name/1.jpg
        ...
    """

    def __init__(
        self,
        data_folder: str | Path,
        data_split_folder: str | Path,
        dataset_name: str,
        image_size: Tuple[int, int],
        waypoint_spacing: int,
        len_traj_pred: int,
        learn_angle: bool,
        context_size: int,
        context_type: str = "temporal",
        end_slack: int = 0,
        goals_per_obs: int = 1,
        normalize: bool = True,
        modality_id: int = 6,
        metric_waypoint_spacing: float = 1.0,
        clip_image_size: Tuple[int, int] = (224, 224),
        clip_model: str = DEFAULT_CLIP_MODEL,
    ) -> None:
        self.data_folder = Path(data_folder)
        self.data_split_folder = Path(data_split_folder)
        self.dataset_name = dataset_name
        self.image_size = tuple(int(v) for v in image_size)
        self.waypoint_spacing = int(waypoint_spacing)
        self.len_traj_pred = int(len_traj_pred)
        self.learn_angle = bool(learn_angle)
        self.context_size = int(context_size)
        self.context_type = str(context_type)
        self.end_slack = int(end_slack)
        self.goals_per_obs = int(goals_per_obs)
        self.normalize = bool(normalize)
        self.modality_id = int(modality_id)
        self.metric_waypoint_spacing = float(metric_waypoint_spacing)
        self.clip_image_size = tuple(int(v) for v in clip_image_size)
        self.clip_model = str(clip_model)
        self.dummy_text_feature: torch.Tensor | None = None
        self.action_horizon = self.len_traj_pred * self.waypoint_spacing
        self.text_encoder = None
        self.prompt_cache: Dict[str, List[str]] = {}
        self.text_feature_cache: Dict[str, torch.Tensor] = {}

        self.validate_settings()
        self.modality_uses = MODALITY_USES[self.modality_id]
        if "satellite" in self.modality_uses:
            raise NotImplementedError(
                "The current raw trajectory dataset layout does not include satellite map images. "
                "Use modality_id values without satellite input, or extend EdgeNavigationDataset "
                "with explicit map image loading."
            )
        self.traj_names = self.load_traj_names()
        self.trajectory_cache: Dict[str, Mapping[str, np.ndarray]] = {}
        self.index_to_data = self.build_sample_index()
        if not self.index_to_data:
            raise ValueError(
                f"No trainable samples found for dataset={dataset_name} split={self.data_split_folder}"
            )

        self.image_transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ]
        )

    def validate_settings(self) -> None:
        if self.modality_id not in MODALITY_USES:
            raise ValueError(f"Unsupported modality_id={self.modality_id} for {self.dataset_name}")
        if self.context_type != "temporal":
            raise ValueError("Only temporal context_type is supported for OmniVLA-edge datasets.")
        if self.goals_per_obs < 1:
            raise ValueError("goals_per_obs must be >= 1.")
        if self.waypoint_spacing < 1:
            raise ValueError("waypoint_spacing must be >= 1.")

    def uses_modality(self, name: str) -> bool:
        return name in self.modality_uses

    def load_traj_names(self) -> List[str]:
        traj_names_file = (
            self.data_split_folder
            if self.data_split_folder.is_file()
            else self.data_split_folder / "traj_names.txt"
        )
        if not traj_names_file.exists():
            raise FileNotFoundError(f"Missing traj_names.txt: {traj_names_file}")
        names = [line.strip() for line in traj_names_file.read_text().splitlines()]
        return [name for name in names if name]

    def load_trajectory(self, traj_name: str) -> Mapping[str, np.ndarray]:
        if traj_name not in self.trajectory_cache:
            traj_path = self.data_folder / traj_name / "traj_data.pkl"
            if not traj_path.exists():
                raise FileNotFoundError(f"Missing trajectory data: {traj_path}")
            with traj_path.open("rb") as f:
                self.trajectory_cache[traj_name] = pickle.load(f)
        return self.trajectory_cache[traj_name]

    def read_positions(self, traj_data: Mapping[str, np.ndarray]) -> np.ndarray:
        for key in ("position", "positions"):
            if key in traj_data:
                return np.asarray(traj_data[key], dtype=np.float32)
        raise KeyError("traj_data.pkl must contain 'position' or 'positions'.")

    def read_yaw(self, traj_data: Mapping[str, np.ndarray]) -> np.ndarray:
        for key in ("yaw", "yaws", "heading", "headings"):
            if key in traj_data:
                yaw = np.asarray(traj_data[key], dtype=np.float32)
                return yaw.squeeze(-1) if yaw.ndim == 2 and yaw.shape[-1] == 1 else yaw
        raise KeyError("traj_data.pkl must contain yaw/heading data.")

    def build_sample_index(self) -> List[Tuple[str, int, int]]:
        samples = []
        begin_time = self.context_size * self.waypoint_spacing
        action_horizon = self.len_traj_pred * self.waypoint_spacing
        for traj_name in self.traj_names:
            traj_data = self.load_trajectory(traj_name)
            traj_len = len(self.read_positions(traj_data))
            end_time = traj_len - self.end_slack - action_horizon
            for curr_time in range(begin_time, end_time):
                max_goal_time = min(traj_len - self.end_slack - 1, curr_time + action_horizon)
                if max_goal_time > curr_time:
                    samples.append((traj_name, curr_time, max_goal_time))
        return samples

    def __len__(self) -> int:
        return len(self.index_to_data) * self.goals_per_obs

    def load_image(self, traj_name: str, time: int, size: Tuple[int, int]) -> torch.Tensor:
        image_path = self.data_folder / traj_name / f"{time}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")
        with Image.open(image_path) as img:
            img = img.convert("RGB").resize(size)
            image = TF.to_tensor(img)
        if self.normalize:
            image = self.image_transform(image)
        return image

    def convert_to_local_coords(
        self, positions: np.ndarray, curr_pos: np.ndarray, curr_yaw: float
    ) -> np.ndarray:
        rotmat = np.array(
            [
                [math.cos(curr_yaw), -math.sin(curr_yaw)],
                [math.sin(curr_yaw), math.cos(curr_yaw)],
            ],
            dtype=np.float32,
        )
        return (positions - curr_pos).dot(rotmat)

    def compute_actions(
        self, traj_data: Mapping[str, np.ndarray], curr_time: int, goal_time: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        positions_all = self.read_positions(traj_data)
        yaw_all = self.read_yaw(traj_data)
        indices = [
            min(curr_time + i * self.waypoint_spacing, len(positions_all) - 1)
            for i in range(self.len_traj_pred + 1)
        ]
        positions = positions_all[indices]
        yaw = yaw_all[indices]
        goal_pos = positions_all[min(goal_time, len(positions_all) - 1)]

        waypoints = self.convert_to_local_coords(positions, positions[0], float(yaw[0]))
        goal_pos_local = self.convert_to_local_coords(goal_pos[None], positions[0], float(yaw[0]))[0]

        actions_xy = waypoints[1:]
        if self.normalize:
            scale = self.metric_waypoint_spacing * self.waypoint_spacing
            actions_xy = actions_xy / scale
            goal_pos_local = goal_pos_local / scale

        if self.learn_angle:
            yaw_rel = yaw[1:] - yaw[0]
            angle = np.stack([np.cos(yaw_rel), np.sin(yaw_rel)], axis=-1)
            actions = np.concatenate([actions_xy, angle], axis=-1)
        else:
            actions = actions_xy

        return (
            torch.as_tensor(actions, dtype=torch.float32),
            torch.as_tensor(goal_pos_local, dtype=torch.float32),
        )

    def zero_text_feature(self) -> torch.Tensor:
        return torch.zeros(512, dtype=torch.float32)

    def fallback_text_feature(self) -> torch.Tensor:
        if self.uses_modality("language"):
            return self.get_dummy_text_feature()
        return self.zero_text_feature()

    def get_text_feature(self, traj_data: Mapping[str, np.ndarray], curr_time: int) -> torch.Tensor:
        if "feat_text" in traj_data:
            feat = traj_data["feat_text"]
        elif "lan_prompt_feature" in traj_data:
            feat = traj_data["lan_prompt_feature"]
        else:
            return self.fallback_text_feature()

        feat_array = np.asarray(feat, dtype=np.float32)
        if feat_array.size == 0:
            return self.fallback_text_feature()
        if feat_array.ndim > 1:
            if feat_array.shape[0] == 0:
                return self.fallback_text_feature()
            feat_array = feat_array[min(curr_time, feat_array.shape[0] - 1)]
        return torch.as_tensor(feat_array, dtype=torch.float32)

    def get_prompt(self, traj_name: str, curr_time: int) -> str:
        if traj_name not in self.prompt_cache:
            prompt_path = self.data_folder / traj_name / "traj_prompt.txt"
            if not prompt_path.exists():
                raise FileNotFoundError(f"Missing language prompt data: {prompt_path}")
            self.prompt_cache[traj_name] = prompt_path.read_text(encoding="utf-8").splitlines()

        prompts = self.prompt_cache[traj_name]
        if not prompts:
            return DEFAULT_DUMMY_LANGUAGE
        return prompts[min(curr_time, len(prompts) - 1)].strip() or DEFAULT_DUMMY_LANGUAGE

    def encode_text(self, text: str) -> torch.Tensor:
        if text not in self.text_feature_cache:
            import clip

            if self.text_encoder is None:
                text_encoder, _ = clip.load(self.clip_model, device="cpu")
                self.text_encoder = text_encoder.float().eval()
            with torch.no_grad():
                tokens = clip.tokenize(text, truncate=True)
                self.text_feature_cache[text] = (
                    self.text_encoder.encode_text(tokens).squeeze(0).to(dtype=torch.float32).cpu()
                )
        return self.text_feature_cache[text].clone()

    def get_prompt_text_feature(self, traj_name: str, curr_time: int) -> torch.Tensor:
        return self.encode_text(self.get_prompt(traj_name, curr_time))

    def get_dummy_text_feature(self) -> torch.Tensor:
        if self.dummy_text_feature is None:
            self.dummy_text_feature = self.encode_text(DEFAULT_DUMMY_LANGUAGE)
        return self.dummy_text_feature.clone()

    def context_times(self, curr_time: int) -> range:
        start_time = curr_time - self.context_size * self.waypoint_spacing
        return range(start_time, curr_time + 1, self.waypoint_spacing)

    def build_observation_images(self, traj_name: str, curr_time: int) -> torch.Tensor:
        return torch.cat(
            [self.load_image(traj_name, time, self.image_size) for time in self.context_times(curr_time)],
            dim=0,
        )

    def build_map_images(self, obs_images: torch.Tensor, goal_image: torch.Tensor) -> torch.Tensor:
        current_small = obs_images[-3:]
        return torch.cat((current_small, goal_image, current_small), dim=0)

    def build_goal_pose(
        self,
        traj_data: Mapping[str, np.ndarray],
        curr_time: int,
        goal_time: int,
        goal_pos: torch.Tensor,
    ) -> torch.Tensor:
        goal_pose = torch.zeros(4, dtype=torch.float32)
        if self.uses_modality("pose"):
            yaw = self.read_yaw(traj_data)
            goal_pose[:2] = goal_pos
            goal_pose[2] = math.cos(float(yaw[goal_time] - yaw[curr_time]))
            goal_pose[3] = math.sin(float(yaw[goal_time] - yaw[curr_time]))
        return goal_pose

    def build_language_feature(
        self,
        traj_name: str,
        traj_data: Mapping[str, np.ndarray],
        curr_time: int,
    ) -> torch.Tensor:
        if self.uses_modality("language"):
            return self.get_prompt_text_feature(traj_name, curr_time)
        return self.get_text_feature(traj_data, curr_time)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        base_index = index // self.goals_per_obs
        traj_name, curr_time, max_goal_time = self.index_to_data[base_index]
        goal_time = np.random.randint(curr_time + 1, max_goal_time + 1)
        traj_data = self.load_trajectory(traj_name)

        obs_images = self.build_observation_images(traj_name, curr_time)
        goal_image = self.load_image(traj_name, goal_time, self.image_size)
        current_img = self.load_image(traj_name, curr_time, self.clip_image_size)

        actions, goal_pos = self.compute_actions(traj_data, curr_time, goal_time)
        goal_pose = self.build_goal_pose(traj_data, curr_time, goal_time, goal_pos)
        map_images = self.build_map_images(obs_images, goal_image)
        feat_text = self.build_language_feature(traj_name, traj_data, curr_time)

        dist_to_goal = torch.as_tensor(
            (goal_time - curr_time) / self.action_horizon, dtype=torch.float32
        )
        return {
            "obs_images": obs_images,
            "goal_pose": goal_pose,
            "map_images": map_images,
            "goal_image": goal_image,
            "goal_mask": torch.as_tensor(self.modality_id, dtype=torch.long),
            "feat_text": feat_text,
            "current_img": current_img,
            "actions": actions,
            "dist_to_goal": dist_to_goal,
        }


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
