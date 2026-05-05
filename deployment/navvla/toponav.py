from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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
