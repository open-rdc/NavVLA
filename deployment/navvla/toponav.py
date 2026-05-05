from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import yaml
from PIL import Image as PILImage
from torchvision import transforms


@dataclass(frozen=True)
class TopologicalNode:
    node_id: int
    image_name: str
    feature: np.ndarray
    edges: Tuple[Dict[str, object], ...]


class TopologicalNavigator:
    def __init__(
        self,
        topomap_path: Path,
        image_dir: Path,
        weight_path: Path,
        device: torch.device,
        image_size: Tuple[int, int],
        crop_size: int = 288,
    ) -> None:
        self.topomap_path = Path(topomap_path)
        self.image_dir = Path(image_dir)
        self.weight_path = Path(weight_path)
        self.device = device
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.crop_size = int(crop_size)

        self.nodes = self._load_topomap(self.topomap_path)
        self.node_index_by_id = {node.node_id: idx for idx, node in enumerate(self.nodes)}
        self.feature_matrix = np.stack([node.feature for node in self.nodes], axis=0)
        self.model = torch.jit.load(str(self.weight_path), map_location=self.device).eval()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((85, 85), antialias=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _load_topomap(topomap_path: Path) -> List[TopologicalNode]:
        if not topomap_path.exists():
            raise FileNotFoundError(f"Topomap file not found: {topomap_path}")

        with topomap_path.open("r", encoding="utf-8") as f:
            topomap = yaml.safe_load(f) or {}

        raw_nodes = topomap.get("nodes", [])
        if not raw_nodes:
            raise ValueError(f"No nodes found in topomap: {topomap_path}")

        nodes = []
        for raw_node in raw_nodes:
            feature = np.asarray(raw_node["feature"], dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(feature))
            if norm <= 1e-8:
                raise ValueError(f"Topomap feature has zero norm: node_id={raw_node.get('id')}")
            feature = feature / norm

            edges = tuple(raw_node.get("edges", []))
            if not edges:
                raise ValueError(f"Topomap node must have at least one edge: node_id={raw_node.get('id')}")

            nodes.append(
                TopologicalNode(
                    node_id=int(raw_node["id"]),
                    image_name=str(raw_node["image"]),
                    feature=feature,
                    edges=edges,
                )
            )
        return nodes

    def _center_crop(self, image_bgr: np.ndarray) -> np.ndarray:
        height, width = image_bgr.shape[:2]
        side = min(height, width, self.crop_size)
        top = (height - side) // 2
        left = (width - side) // 2
        return image_bgr[top : top + side, left : left + side]

    def extract_feature(self, image_bgr: np.ndarray) -> np.ndarray:
        cropped = self._center_crop(image_bgr)
        image_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            feature = self.model(image_tensor)

        feature_np = feature.squeeze(0).detach().cpu().numpy().reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(feature_np))
        if norm <= 1e-8:
            raise RuntimeError("PlaceNet returned a zero-norm feature.")
        return feature_np / norm

    def estimate_current_node(self, image_bgr: np.ndarray) -> Tuple[int, float]:
        query_feature = self.extract_feature(image_bgr)
        scores = np.dot(self.feature_matrix, query_feature)
        best_index = int(np.argmax(scores))
        return best_index, float(scores[best_index])

    def select_goal_node(self, current_index: int) -> int:
        current_node = self.nodes[current_index]
        target_id = int(current_node.edges[0].get("target", current_node.node_id))
        return self.node_index_by_id.get(target_id, current_index)

    def load_goal_image(self, node_index: int) -> PILImage.Image:
        image_path = self.image_dir / self.nodes[node_index].image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Topomap image not found: {image_path}")
        return PILImage.open(image_path).convert("RGB").resize(self.image_size)


class TopomapGenerator:
    COMMAND_TO_ACTION = {
        0: "roadside",
        1: "straight",
        2: "left",
        3: "right",
    }

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
        self.image_dir = self.dataset_path / "images"
        self.command_dir = self.dataset_path / "commands"
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

    def _load_command(self, image_path: Path) -> int:
        command_path = self.command_dir / f"{image_path.stem}.csv"
        with command_path.open("r", encoding="utf-8") as f:
            line = f.readline().strip()
        return int(float(line.split(",")[0]))

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
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Dataset image directory not found: {self.image_dir}")
        if not self.command_dir.is_dir():
            raise FileNotFoundError(f"Dataset command directory not found: {self.command_dir}")

        self.output_image_dir.mkdir(parents=True, exist_ok=True)
        image_paths = sorted(self.image_dir.glob("*.png"))
        if not image_paths:
            raise ValueError(f"No dataset images found: {self.image_dir}")

        nodes = []
        for node_index, image_path in enumerate(image_paths[:: self.saved_step]):
            command = self._load_command(image_path)
            if command not in self.COMMAND_TO_ACTION:
                raise ValueError(f"Unsupported command value: {command}")

            processed_image = self._preprocess_image(image_path)
            output_image_name = f"img{node_index + 1:05d}.png"
            cv2.imwrite(str(self.output_image_dir / output_image_name), processed_image)
            nodes.append(
                {
                    "id": node_index,
                    "image": output_image_name,
                    "feature": self._extract_feature(processed_image),
                    "action": self.COMMAND_TO_ACTION[command],
                }
            )

        for node_index, node in enumerate(nodes):
            target = node_index + 1 if node_index + 1 < len(nodes) else node_index
            node["edges"] = [{"target": target, "action": node.pop("action")}]

        with self.topomap_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump({"nodes": nodes}, f, sort_keys=False, allow_unicode=False)
        return self.topomap_path
