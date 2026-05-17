from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
        delta: float = 5.0,
        window_lower: int = -1,
        window_upper: int = 10,
    ) -> None:
        self.topomap_path = Path(topomap_path)
        self.image_dir = Path(image_dir)
        self.weight_path = Path(weight_path)
        self.device = device
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.crop_size = int(crop_size)

        # Bayesian filter パラメータ
        self.delta = float(delta)
        self.window_lower = int(window_lower)
        self.window_upper = int(window_upper)
        # 遷移モデル：ウィンドウ内の移動を等確率とする一様分布
        self.transition = np.ones(self.window_upper - self.window_lower, dtype=np.float32)

        # 信念・lambdaは最初のフレームで初期化
        self.belief: Optional[np.ndarray] = None
        self.lambda1: float = 0.0

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

    def _compute_distances(self, query_feature: np.ndarray) -> np.ndarray:
        # L2正規化済みのためdot積=コサイン類似度 → コサイン距離に変換
        dots = np.clip(np.dot(self.feature_matrix, query_feature), -1.0, 1.0)
        return np.sqrt(2.0 - 2.0 * dots)

    def _observation_likelihood(self, query_feature: np.ndarray) -> np.ndarray:
        return np.exp(-self.lambda1 * self._compute_distances(query_feature))

    def _initialize_belief(self, query_feature: np.ndarray) -> None:
        dists = self._compute_distances(query_feature)
        descriptor_quantiles = np.quantile(dists, [0.025, 0.975])
        self.lambda1 = np.log(self.delta) / (descriptor_quantiles[1] - descriptor_quantiles[0])
        self.belief = np.exp(-self.lambda1 * dists)
        self.belief /= self.belief.sum()

    def _update_belief(self, query_feature: np.ndarray) -> None:
        # ===== Prediction ステップ：遷移モデルで信念を前進方向に伝播 =====
        if self.window_lower < 0:
            conv_ind_l = abs(self.window_lower)
            conv_ind_h = len(self.belief) + abs(self.window_lower)
            bel_ind_l, bel_ind_h = 0, len(self.belief)
        else:
            conv_ind_l, conv_ind_h = 0, len(self.belief) - self.window_lower
            bel_ind_l, bel_ind_h = self.window_lower, len(self.belief)

        belief_pad = np.pad(self.belief, len(self.transition) - 1, mode="symmetric")
        conv = np.convolve(belief_pad, self.transition, mode="valid")
        self.belief[bel_ind_l:bel_ind_h] = conv[conv_ind_l:conv_ind_h]

        if self.window_lower > 0:
            self.belief[: self.window_lower] = 0.0

        # ===== Measurement ステップ：観測尤度でベイズ更新 =====
        self.belief *= self._observation_likelihood(query_feature)
        self.belief /= self.belief.sum()

    def estimate_current_node(self, image_bgr: np.ndarray) -> Tuple[int, float]:
        query_feature = self.extract_feature(image_bgr)

        if self.belief is None:
            self._initialize_belief(query_feature)
        else:
            self._update_belief(query_feature)

        best_index = int(np.argmax(self.belief))
        return best_index, float(self.belief[best_index])

    def reset(self) -> None:
        """信念を初期化する。環境が大きく変わった場合や再スタート時に呼ぶ。"""
        self.belief = None
        self.lambda1 = 0.0

    def select_goal_node(self, current_index: int) -> int:
        current_node = self.nodes[current_index]
        target_id = int(current_node.edges[0].get("target", current_node.node_id))
        return self.node_index_by_id.get(target_id, current_index)

    def load_goal_image(self, node_index: int) -> PILImage.Image:
        image_path = self.image_dir / self.nodes[node_index].image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Topomap image not found: {image_path}")
        return PILImage.open(image_path).convert("RGB").resize(self.image_size)
