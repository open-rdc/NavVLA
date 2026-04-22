#!/usr/bin/env python3
"""ROS2 inference node for OmniVLA-edge navigation."""

from __future__ import annotations

import argparse
import math
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Tuple

import clip  # type: ignore
import numpy as np
import torch
from PIL import Image as PILImage

from OmniVLA.inference.utils_policy import (
    load_model,
    transform_images_PIL_mask,
    transform_images_map,
)
from .preprocess import build_mask, ensure_image_size, ensure_normalize_mean, load_yaml

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path as NavPath
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool


class OmniVLANavigationNode(Node):
    """Inference node publishing velocity and predicted path from camera input."""

    def __init__(
        self,
        nav_config_path: Path,
        preprocess_config_path: Path,
    ) -> None:
        super().__init__("navigation")

        self.repo_root = Path(__file__).resolve().parents[2]
        self.omnivla_root = self.repo_root / "OmniVLA"
        if not self.omnivla_root.exists():
            raise FileNotFoundError(f"OmniVLA submodule not found: {self.omnivla_root}")

        self.nav_cfg = load_yaml(nav_config_path)
        self.preprocess_cfg = load_yaml(preprocess_config_path)

        self.use_goal_pose = bool(self.nav_cfg.get("use_goal_pose", True))
        self.use_goal_image_8 = bool(self.nav_cfg.get("use_goal_image_8", True))
        self.use_lan_prompt = bool(self.nav_cfg.get("use_lan_prompt", True))

        self.context_size = int(self.nav_cfg.get("context_size", 5))
        self.waypoint_spacing = float(self.nav_cfg.get("metric_waypoint_spacing", 0.1))
        self.waypoint_select = int(self.nav_cfg.get("waypoint_select", 4))
        self.linear_max_vel = float(self.nav_cfg.get("linear_max_vel", 0.3))
        self.angular_max_vel = float(self.nav_cfg.get("angular_max_vel", 0.3))
        self.path_frame_id = str(self.nav_cfg.get("path_frame_id", "base_link"))
        self.modality_id = int(self.nav_cfg.get("modality_id", 3))

        self.interval_ms = int(self.nav_cfg.get("interval_ms", 100))
        self.autonomous_enabled = False
        self.last_cmd_active = False

        self.latest_image: Optional[PILImage.Image] = None
        self.context_queue: Deque[PILImage.Image] = deque(maxlen=self.context_size + 1)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._setup_inference_components()

        self.image_sub = self.create_subscription(Image, "/image_raw", self.image_callback, 10)
        self.autonomous_sub = self.create_subscription(
            Bool, "/autonomous", self.autonomous_callback, 10
        )
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.path_pub = self.create_publisher(NavPath, "/path", 10)

        self.timer = self.create_timer(self.interval_ms / 1000.0, self.timer_callback)
        self.get_logger().info("navigation.py node started")

    def _setup_inference_components(self) -> None:
        mean = ensure_normalize_mean(self.preprocess_cfg.get("normalize_mean", [0.485, 0.456, 0.406]))
        self.get_logger().info(
            f"Using normalize_mean={mean}. normalize_std is fixed in OmniVLA implementation."
        )

        self.obs_size = ensure_image_size(self.preprocess_cfg.get("obs_image_size"), "obs_image_size")
        self.goal_size = ensure_image_size(
            self.preprocess_cfg.get("goal_image_size"), "goal_image_size"
        )
        self.clip_size = ensure_image_size(
            self.preprocess_cfg.get("clip_image_size"), "clip_image_size"
        )

        use_mask = bool(self.preprocess_cfg.get("use_mask", False))
        mask_path_value = str(self.preprocess_cfg.get("mask_path", ""))
        if use_mask and mask_path_value:
            mask_path = Path(mask_path_value)
            if not mask_path.is_absolute():
                mask_path = (self.repo_root / mask_path).resolve()
            mask_path_value = str(mask_path)

        self.mask_obs = build_mask(self.obs_size, use_mask, mask_path_value)
        self.mask_goal = build_mask(self.goal_size, use_mask, mask_path_value)
        self.mask_clip = build_mask(self.clip_size, use_mask, mask_path_value)

        weights_path = Path(str(self.nav_cfg.get("weights_path", "")))
        if not weights_path.is_absolute():
            weights_path = (self.repo_root / weights_path).resolve()
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")

        model_params = {
            "model_type": str(self.nav_cfg.get("model_type", "omnivla-edge")),
            "len_traj_pred": int(self.nav_cfg.get("len_traj_pred", 8)),
            "learn_angle": bool(self.nav_cfg.get("learn_angle", True)),
            "context_size": self.context_size,
            "obs_encoder": str(self.nav_cfg.get("obs_encoder", "efficientnet-b0")),
            "obs_encoding_size": int(self.nav_cfg.get("obs_encoding_size", 1024)),
            "late_fusion": bool(self.nav_cfg.get("late_fusion", False)),
            "mha_num_attention_heads": int(self.nav_cfg.get("mha_num_attention_heads", 4)),
            "mha_num_attention_layers": int(self.nav_cfg.get("mha_num_attention_layers", 4)),
            "mha_ff_dim_factor": int(self.nav_cfg.get("mha_ff_dim_factor", 4)),
            "clip_type": str(self.preprocess_cfg.get("clip_model", "ViT-B/32")),
        }

        self.model, self.text_encoder, _ = load_model(str(weights_path), model_params, self.device)
        self.model = self.model.to(self.device).eval()
        self.text_encoder = self.text_encoder.to(self.device).eval()

        if self.use_goal_image_8:
            goal_image_path = Path(str(self.nav_cfg.get("goal_image_path", "OmniVLA/inference/goal_img.jpg")))
            if not goal_image_path.is_absolute():
                goal_image_path = (self.repo_root / goal_image_path).resolve()
            if not goal_image_path.exists():
                raise FileNotFoundError(f"Goal image not found: {goal_image_path}")
            goal_pil = PILImage.open(goal_image_path).convert("RGB").resize(self.goal_size)
        else:
            goal_pil = PILImage.new("RGB", self.goal_size, color=(0, 0, 0))

        self.goal_image_tensor = transform_images_PIL_mask(goal_pil, self.mask_goal).to(self.device)

        if self.use_goal_pose:
            raw_goal_pose = self.nav_cfg.get("goal_pose", [0.0, 0.0, 1.0, 0.0])
            if not isinstance(raw_goal_pose, list) or len(raw_goal_pose) != 4:
                raise ValueError("goal_pose must be [x, y, cos, sin]")
            goal_pose = [float(v) for v in raw_goal_pose]
        else:
            goal_pose = [0.0, 0.0, 1.0, 0.0]

        self.goal_pose_tensor = torch.tensor([goal_pose], dtype=torch.float32, device=self.device)
        self.modality_tensor = torch.tensor([self.modality_id], dtype=torch.long, device=self.device)

        prompt = str(self.nav_cfg.get("lan_prompt", "xxxx")) if self.use_lan_prompt else "xxxx"
        token = clip.tokenize(prompt, truncate=True).to(self.device)
        with torch.no_grad():
            self.feat_text = self.text_encoder.encode_text(token)

        self.satellite_cur = PILImage.new("RGB", (352, 352), color=(0, 0, 0))
        self.satellite_goal = PILImage.new("RGB", (352, 352), color=(0, 0, 0))

    def autonomous_callback(self, msg: Bool) -> None:
        self.autonomous_enabled = bool(msg.data)

    def image_callback(self, msg: Image) -> None:
        try:
            self.latest_image = self._ros_image_to_pil(msg)
        except Exception as exc:
            self.get_logger().warning(f"Failed to decode /image_raw: {exc}")

    def timer_callback(self) -> None:
        if not self.autonomous_enabled:
            self._publish_zero_twist()
            return

        if self.latest_image is None:
            return

        try:
            linear_vel, angular_vel, waypoints = self._infer(self.latest_image)
        except Exception as exc:
            self.get_logger().warning(f"Inference failed: {exc}")
            self._publish_zero_twist()
            return

        twist = Twist()
        twist.linear.x = float(linear_vel)
        twist.angular.z = float(angular_vel)
        self.cmd_pub.publish(twist)

        path_msg = self._build_path_msg(waypoints)
        self.path_pub.publish(path_msg)
        self.last_cmd_active = True

    def _publish_zero_twist(self) -> None:
        if not self.last_cmd_active:
            return
        stop = Twist()
        self.cmd_pub.publish(stop)
        self.last_cmd_active = False

    def _infer(self, current_image: PILImage.Image) -> Tuple[float, float, np.ndarray]:
        obs_image = current_image.resize(self.obs_size)
        if not self.context_queue:
            for _ in range(self.context_size + 1):
                self.context_queue.append(obs_image)
        else:
            self.context_queue.append(obs_image)

        obs_images = transform_images_PIL_mask(list(self.context_queue), self.mask_obs).to(self.device)
        split_obs = torch.split(obs_images, 3, dim=1)
        obs_image_cur = split_obs[-1].to(self.device)
        obs_images = torch.cat(split_obs, dim=1).to(self.device)

        cur_large_img = transform_images_PIL_mask(
            current_image.resize(self.clip_size), self.mask_clip
        ).to(self.device)

        map_images = torch.cat(
            (
                transform_images_map(self.satellite_cur).to(self.device),
                transform_images_map(self.satellite_goal).to(self.device),
                obs_image_cur,
            ),
            axis=1,
        )

        with torch.no_grad():
            action_pred, _, _ = self.model(
                obs_images,
                self.goal_pose_tensor,
                map_images,
                self.goal_image_tensor,
                self.modality_tensor,
                self.feat_text,
                cur_large_img,
            )

        waypoints = action_pred.float().cpu().numpy()[0]
        selected = max(0, min(self.waypoint_select, waypoints.shape[0] - 1))
        linear_vel, angular_vel = self._waypoint_to_velocity(waypoints[selected])
        return linear_vel, angular_vel, waypoints

    def _waypoint_to_velocity(self, waypoint: np.ndarray) -> Tuple[float, float]:
        dx, dy, hx, hy = [float(v) for v in waypoint]
        dx *= self.waypoint_spacing
        dy *= self.waypoint_spacing

        eps = 1e-8
        dt = 1.0 / 3.0

        if abs(dx) < eps and abs(dy) < eps:
            linear_vel = 0.0
            angular_vel = self._clip_angle(math.atan2(hy, hx)) / dt
        elif abs(dx) < eps:
            linear_vel = 0.0
            angular_vel = math.copysign(math.pi / (2.0 * dt), dy)
        else:
            linear_vel = dx / dt
            angular_vel = math.atan(dy / dx) / dt

        linear_vel = float(np.clip(linear_vel, 0.0, self.linear_max_vel))
        angular_vel = float(np.clip(angular_vel, -self.angular_max_vel, self.angular_max_vel))
        return linear_vel, angular_vel

    @staticmethod
    def _clip_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _build_path_msg(self, waypoints: np.ndarray) -> NavPath:
        msg = NavPath()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.path_frame_id

        for wp in waypoints:
            pose = PoseStamped()
            pose.header = msg.header
            x = float(wp[0]) * self.waypoint_spacing
            y = float(wp[1]) * self.waypoint_spacing
            yaw = math.atan2(float(wp[3]), float(wp[2]))

            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            msg.poses.append(pose)

        return msg

    @staticmethod
    def _ros_image_to_pil(msg: Image) -> PILImage.Image:
        if msg.height == 0 or msg.width == 0:
            raise ValueError("Received empty image")

        if msg.encoding not in ("rgb8", "bgr8", "mono8"):
            raise ValueError(f"Unsupported encoding: {msg.encoding}")

        if msg.encoding == "mono8":
            raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
            mono = raw[:, : msg.width]
            rgb = np.repeat(mono[:, :, None], 3, axis=2)
        else:
            channels = 3
            raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
            rgb = raw[:, : msg.width * channels].reshape(msg.height, msg.width, channels)
            if msg.encoding == "bgr8":
                rgb = rgb[:, :, ::-1]

        return PILImage.fromarray(rgb.astype(np.uint8), mode="RGB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nav-config",
        default="deployment/config/nav.yaml",
        help="Path to nav config yaml",
    )
    parser.add_argument(
        "--preprocess-config",
        default="deployment/config/preprocess.yaml",
        help="Path to preprocess config yaml",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    nav_config_path = Path(args.nav_config).expanduser().resolve()
    preprocess_config_path = Path(args.preprocess_config).expanduser().resolve()

    rclpy.init()
    node = OmniVLANavigationNode(nav_config_path, preprocess_config_path)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
