#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
import yaml
from torchvision import transforms


REPO_ROOT = Path(__file__).resolve().parents[2]


class TopomapGenerator:
    def __init__(
        self,
        dataset_path: Path,
        output_dir: Path,
        weight_path: Path,
        device: torch.device,
        saved_step: int = 10,
        crop_size: int = 288,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_image_dir = self.output_dir / "images"
        self.topomap_path = self.output_dir / "topomap.yaml"
        self.weight_path = Path(weight_path)
        self.device = device
        self.saved_step = int(saved_step)
        self.crop_size = int(crop_size)
        if self.saved_step < 1:
            raise ValueError("saved_step must be >= 1")

        self.model = torch.jit.load(str(self.weight_path), map_location=self.device).eval()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((85, 85), antialias=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _image_sort_key(path: Path) -> tuple[int, str]:
        try:
            return int(path.stem), path.name
        except ValueError:
            return 0, path.name

    def _load_trajectory_names(self) -> list[str]:
        traj_names_path = self.dataset_path / "traj_names.txt"
        if not traj_names_path.exists():
            return [path.name for path in sorted(self.dataset_path.glob("traj_*")) if path.is_dir()]

        with traj_names_path.open("r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def _load_trajectory(self, traj_name: str) -> tuple[list[Path], Optional[np.ndarray], Optional[np.ndarray]]:
        traj_dir = self.dataset_path / traj_name
        if not traj_dir.is_dir():
            raise FileNotFoundError(f"Trajectory directory not found: {traj_dir}")

        image_paths = sorted(
            list(traj_dir.glob("*.jpg")) + list(traj_dir.glob("*.png")),
            key=self._image_sort_key,
        )
        if not image_paths:
            raise ValueError(f"No images found in trajectory: {traj_dir}")

        traj_data_path = traj_dir / "traj_data.pkl"
        if not traj_data_path.exists():
            return image_paths, None, None

        with traj_data_path.open("rb") as f:
            traj_data = pickle.load(f)

        positions = traj_data.get("position")
        yaws = traj_data.get("yaw")
        if positions is None or yaws is None:
            return image_paths, None, None

        return image_paths, np.asarray(positions), np.asarray(yaws)

    def _preprocess_image(self, image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        height, width = image.shape[:2]
        side = min(height, width, self.crop_size)
        top = (height - side) // 2
        left = (width - side) // 2
        cropped = image[top : top + side, left : left + side]
        return cv2.resize(cropped, (85, 85), interpolation=cv2.INTER_AREA)

    def _extract_feature(self, image_bgr: np.ndarray) -> Sequence[float]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            feature = self.model(image_tensor)

        feature_np = feature.squeeze(0).detach().cpu().numpy().reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(feature_np))
        if norm > 1e-8:
            feature_np = feature_np / norm
        return feature_np.tolist()

    def generate(self) -> Path:
        self.output_image_dir.mkdir(parents=True, exist_ok=True)
        traj_names = self._load_trajectory_names()
        if not traj_names:
            raise ValueError(f"No trajectories found in dataset: {self.dataset_path}")

        nodes = []
        for traj_name in traj_names:
            image_paths, positions, yaws = self._load_trajectory(traj_name)
            for image_path in image_paths[:: self.saved_step]:
                node_index = len(nodes)
                frame_index = self._image_sort_key(image_path)[0]

                processed_image = self._preprocess_image(image_path)
                output_image_name = f"img{node_index + 1:05d}.png"
                cv2.imwrite(str(self.output_image_dir / output_image_name), processed_image)

                node = {
                    "id": node_index,
                    "image": output_image_name,
                    "feature": self._extract_feature(processed_image),
                    "source": {
                        "trajectory": traj_name,
                        "frame": frame_index,
                        "image": str(image_path.relative_to(self.dataset_path)),
                    },
                }
                if positions is not None and yaws is not None and frame_index < len(positions) and frame_index < len(yaws):
                    node["pose"] = {
                        "x": float(positions[frame_index][0]),
                        "y": float(positions[frame_index][1]),
                        "yaw": float(yaws[frame_index]),
                    }

                nodes.append(node)

        for node_index, node in enumerate(nodes):
            target = node_index + 1 if node_index + 1 < len(nodes) else node_index
            node["edges"] = [{"target": target}]

        with self.topomap_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump({"nodes": nodes}, f, sort_keys=False, allow_unicode=False)
        return self.topomap_path


def resolve_cli_path(raw_path: str, base_path: Path = Path.cwd()) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else base_path / path


def main(args: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", help="Dataset directory generated by NavVLA data_collection.py")
    parser.add_argument(
        "--output-dir",
        default="deployment/config/topomap",
        help="Directory where topomap.yaml and images/ are written, relative to the NavVLA repository root",
    )
    parser.add_argument(
        "--weights",
        default="deployment/weights/placenet.pt",
        help="PlaceNet TorchScript weight path, relative to the NavVLA repository root",
    )
    parser.add_argument("--saved-step", type=int, default=10, help="Use every Nth dataset image as a node")
    parser.add_argument("--crop-size", type=int, default=288, help="Center crop size before resizing to 85x85")
    parsed = parser.parse_args(args)

    dataset_path = resolve_cli_path(parsed.dataset_path)
    output_dir = resolve_cli_path(parsed.output_dir, REPO_ROOT)
    weights_path = resolve_cli_path(parsed.weights, REPO_ROOT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = TopomapGenerator(
        dataset_path=dataset_path,
        output_dir=output_dir,
        weight_path=weights_path,
        device=device,
        saved_step=parsed.saved_step,
        crop_size=parsed.crop_size,
    )
    topomap_path = generator.generate()
    print(f"Topomap saved: {topomap_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
