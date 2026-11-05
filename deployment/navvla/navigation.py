#!/usr/bin/env python3

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Tuple
from ament_index_python.packages import get_package_share_directory

import clip
import cv2
import numpy as np
import torch
from PIL import Image as PILImage

from OmniVLA.inference.utils_policy import (
    load_model,
    transform_images_PIL_mask,
)
from .preprocess import build_mask, build_omnivla_edge_inputs, image_to_cv2, load_yaml
from .toponav import TopologicalNavigator
from .visualize import render_debug_image, to_image_msg

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path as NavPath
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Empty, String


STOP_RE = re.compile(r"\bstop\b", re.I)  # 止まれの指示("Stop at the white line" など言い回しは自由)


class OmniVLANavigationNode(Node):
    def __init__(
        self,
        nav_config_path: Path,
        preprocess_config_path: Path,
        package_share_dir: Path,
    ) -> None:
        super().__init__("navigation")

        self.autonomous_flag = False
        self.context_queue = []
        self.frame_history = []
        self.obs_image = None
        self.obs_image_bgr = None
        self.last_linear_vel = None
        # 止まれ: 自律走行中に止まれの指示("stop" を含む指示)が一度出たら /cross までその指示を保持する
        # (停止ノードは見た目がほぼ同じで、止まりきる前に位置推定が区間を抜けると走り抜けてしまうため)。
        # 保持するのは指示の文だけで、動くかどうかはモデルが決める(白線で前のロボットに続いて詰める)
        self.stop_latched = False
        self.latched_prompt = None
        self.latched_end = None    # 保持した止まれの区間の最後のノード(トポマップ)
        self.latched_next = None   # その区間の次の指示
        # /cross: 保持した止まれの区間の次の指示(横断歩道を渡る指示など)に切り替える。
        # 位置推定が止まれの区間を抜けたら(トポマップが無いときは止まれ以外が cross_release_sec 続いたら)地図の指示に戻す
        self.cross_override = False
        self.cross_prompt = None
        self.cross_until = None
        self.crossed_end = -1      # /cross で渡った止まれの区間の最後のノード。推定が戻ってもその区間では保持し直さない
        self.crossed_next = None   # その区間の次の指示(推定が戻ったときもこちらを出す)
        self.non_stop_since = None
        self.encoded_prompt = None
        self.package_share_dir = package_share_dir

        self.nav_cfg = load_yaml(nav_config_path)
        self.preprocess_cfg = load_yaml(preprocess_config_path)

        self.init_params()
        self.init_model()
        self.init_model_modality()
        self.init_toponav()

        self.image_sub = self.create_subscription(Image, "/image_raw", self.image_callback, 10)
        self.autonomous_sub = self.create_subscription(Bool, "/autonomous", self.autonomous_callback, 10)
        self.prompt_sub = self.create_subscription(String, "/prompt", self.prompt_callback, 10)
        self.cross_sub = self.create_subscription(Empty, "/cross", self.cross_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.path_pub = self.create_publisher(NavPath, "/path", 10)
        self.debug_image_pub = self.create_publisher(Image, "/debug_image", 10)
        # トポマップで決めた指示を音声案内(voice_node)にも流す
        self.prompt_pub = self.create_publisher(String, "/prompt", 10)

        self.timer = self.create_timer(self.interval_ms / 1000.0, self.timer_callback)

    def init_params(self) -> None:
        self.context_size = self.nav_cfg.get("context_size", 5)
        self.waypoint_spacing = self.nav_cfg.get("metric_waypoint_spacing", 0.1)
        self.waypoint_select = self.nav_cfg.get("waypoint_select", 4)
        self.linear_max_vel = self.nav_cfg.get("linear_max_vel", 0.3)
        self.angular_max_vel = self.nav_cfg.get("angular_max_vel", 0.3)
        self.path_frame_id = self.nav_cfg.get("path_frame_id", "base_link")
        self.modality_id = self.nav_cfg.get("modality_id", 3)
        self.interval_ms = self.nav_cfg.get("interval_ms", 100)
        # 学習の waypoint_spacing と同じ値。過去画像も予測点も interval_ms の何周期おきかを表す
        self.context_spacing = int(self.nav_cfg.get("context_spacing", 1))
        # 停止以外の指示がこの秒数続いたら横断の上書きを解除する(トポマップを使わないとき)
        self.cross_release_sec = float(self.nav_cfg.get("cross_release_sec", 3.0))
        # 予測点(waypoint_select)までの距離[m]がこれ未満なら止める。0 で無効
        self.stop_distance = float(self.nav_cfg.get("stop_distance", 0.0))
        # 予測した経路(全点)が今の位置からこの距離[m]以内なら止める。0 で無効。
        # 4.8 秒(8 点 x 0.6 秒)で 0.3 m = 平均 0.06 m/s 以下のじわじわ進む経路を採用しない(学習の zero_stop_m と同じ考え方)
        self.stop_path_m = float(self.nav_cfg.get("stop_path_m", 0.0))

        modality_to_flags = {
            0: (False, False, False),  # satellite only
            1: (True, False, False),   # pose + satellite
            2: (False, True, False),   # image + satellite
            3: (True, True, True),     # all
            4: (True, False, False),   # pose only
            5: (True, True, False),    # pose + image
            6: (False, True, False),   # image only
            7: (False, False, True),   # language only
            8: (True, False, True),    # language + pose
        }
        self.use_goal_pose, self.use_goal_image, self.use_prompt = modality_to_flags[self.modality_id]

    def init_model(self) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        obs_size = self.preprocess_cfg.get("obs_image_size", [96, 96])
        goal_size = self.preprocess_cfg.get("goal_image_size", [96, 96])
        clip_size = self.preprocess_cfg.get("clip_image_size", [224, 224])

        self.obs_size = (obs_size[0], obs_size[1])
        self.goal_size = (goal_size[0], goal_size[1])
        self.clip_size = (clip_size[0], clip_size[1])

        use_mask = self.preprocess_cfg.get("use_mask", False)
        raw_mask_path = str(self.preprocess_cfg.get("mask_path", ""))
        if raw_mask_path:
            mask_path_obj = self.package_share_dir / Path(raw_mask_path)
            mask_path = str(mask_path_obj)
        else:
            mask_path = ""

        self.mask_obs = build_mask(self.obs_size, use_mask, mask_path)
        self.mask_goal = build_mask(self.goal_size, use_mask, mask_path)
        self.mask_clip = build_mask(self.clip_size, use_mask, mask_path)

        weights_path = self.package_share_dir / Path(str(self.nav_cfg.get("weights_path", "")))
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")

        model_params = {
            "model_type": self.nav_cfg.get("model_type", "omnivla-edge"),
            "len_traj_pred": self.nav_cfg.get("len_traj_pred", 8),
            "learn_angle": self.nav_cfg.get("learn_angle", True),
            "context_size": self.context_size,
            "obs_encoder": self.nav_cfg.get("obs_encoder", "efficientnet-b0"),
            "obs_encoding_size": self.nav_cfg.get("obs_encoding_size", 1024),
            "late_fusion": self.nav_cfg.get("late_fusion", False),
            "mha_num_attention_heads": self.nav_cfg.get("mha_num_attention_heads", 4),
            "mha_num_attention_layers": self.nav_cfg.get("mha_num_attention_layers", 4),
            "mha_ff_dim_factor": self.nav_cfg.get("mha_ff_dim_factor", 4),
            "clip_type": self.preprocess_cfg.get("clip_model", "ViT-B/32"),
        }

        self.model, self.text_encoder, _ = load_model(str(weights_path), model_params, self.device)
        self.model = self.model.to(self.device).eval()
        self.text_encoder = self.text_encoder.to(self.device).eval()
        self.get_logger().info(f"Weights loaded: {weights_path.name}")

    def init_model_modality(self) -> None:
        if self.use_goal_image:
            goal_image_path = self.package_share_dir / Path(
                str(self.nav_cfg.get("goal_image_path", "OmniVLA/inference/goal_img.jpg"))
            )
            goal_pil = PILImage.open(goal_image_path).convert("RGB").resize(self.goal_size)
        else:
            goal_pil = PILImage.new("RGB", self.goal_size, color=(0, 0, 0))

        if self.use_goal_pose:
            raw_goal_pose = self.nav_cfg.get("goal_pose", [0.0, 0.0, 1.0, 0.0])
            goal_pose = [float(v) for v in raw_goal_pose]
        else:
            goal_pose = [0.0, 0.0, 1.0, 0.0]

        self.latest_prompt = "Stop at the white line"
        self.update_text_feature()

        self.satellite_current = PILImage.new("RGB", (352, 352), color=(0, 0, 0))
        self.satellite_goal = PILImage.new("RGB", (352, 352), color=(0, 0, 0))

        self.goal_image_tensor = transform_images_PIL_mask(goal_pil, self.mask_goal).to(self.device)
        self.goal_pose_tensor = torch.tensor([goal_pose], dtype=torch.float32, device=self.device)
        self.modality_tensor = torch.tensor([self.modality_id], dtype=torch.long, device=self.device)

    def autonomous_callback(self, msg: Bool) -> None:
        self.autonomous_flag = bool(msg.data)
        self.get_logger().info(f"Autonomous mode: {self.autonomous_flag}")
        if not self.autonomous_flag and (self.stop_latched or self.cross_override):
            # 手動に切り替えたら保持・横断をやめる(人が動かすので)
            self.stop_latched = False
            self.cross_override = False
            self.get_logger().info("Stop latch / cross cleared (manual mode)")
            if self.use_prompt:
                self.update_text_feature()

    def init_toponav(self) -> None:
        # トポマップ(PlaceNet 特徴)上のベイズフィルタで現在ノードを推定し、そのノードの指示を言語指示にする
        self.toponav = None
        self.toponav_index = None
        if not bool(self.nav_cfg.get("use_toponav", False)):
            return
        share = self.package_share_dir
        self.toponav = TopologicalNavigator(
            topomap_path=share / str(self.nav_cfg["topomap_path"]),
            image_dir=share / str(self.nav_cfg["topomap_image_dir"]),
            weight_path=share / str(self.nav_cfg["placenet_weight_path"]),
            device=self.device,
            image_size=self.goal_size,
            crop_size=int(self.nav_cfg.get("toponav_crop_size", 0)),
            delta=float(self.nav_cfg.get("toponav_delta", 5.0)),
            window_lower=int(self.nav_cfg.get("toponav_window_lower", 0)),
            window_upper=int(self.nav_cfg.get("toponav_window_upper", 2)),
            window_radius=int(self.nav_cfg.get("toponav_window_radius", 2)),
        )
        self.toponav_min_score = float(self.nav_cfg.get("toponav_min_score", 0.4))
        self.get_logger().info(f"Toponav loaded: nodes={len(self.toponav.nodes)}")

    def update_toponav(self) -> None:
        index, score = self.toponav.estimate_current_node(self.obs_image_bgr)
        if score < self.toponav_min_score:
            return  # 推定が定まるまでは指示を変えない
        if index != self.toponav_index:
            self.toponav_index = index
            self.get_logger().info(f"Toponav node={index} score={score:.2f}", throttle_duration_sec=2.0)
        prompt = self.toponav.nodes[index].prompt
        if prompt != self.latest_prompt:
            self.prompt_pub.publish(String(data=prompt))
        self.set_prompt(prompt)

    def prompt_callback(self, msg: String) -> None:
        if self.toponav is not None:
            return  # 指示はトポマップから決める(自分が流した /prompt も届くので無視する)
        self.set_prompt(str(msg.data))

    @staticmethod
    def is_stop(prompt: str) -> bool:
        return bool(STOP_RE.search(prompt or ""))

    def stop_block(self, index):
        """index のノードから同じ指示が続く区間の最後のノードと、その次の指示。トポマップが無ければ (None, None)"""
        if self.toponav is None or index is None:
            return None, None
        nodes = self.toponav.nodes
        end = index
        while end + 1 < len(nodes) and nodes[end + 1].prompt == nodes[index].prompt:
            end += 1
        return end, (nodes[end + 1].prompt if end + 1 < len(nodes) else None)

    def set_prompt(self, prompt: str) -> None:
        self.latest_prompt = prompt
        if self.cross_override:
            if self.cross_until is not None:
                # 位置推定が保持した止まれの区間を抜けたら地図の指示に戻す
                if self.toponav_index is not None and self.toponav_index > self.cross_until:
                    self.cross_override = False
                    self.get_logger().info("Cross released: passed the stop section")
            else:
                now = self.get_clock().now().nanoseconds * 1e-9
                if self.is_stop(prompt):
                    self.non_stop_since = None
                elif self.non_stop_since is None:
                    self.non_stop_since = now
                elif now - self.non_stop_since >= self.cross_release_sec:
                    self.cross_override = False
                    self.non_stop_since = None
                    self.get_logger().info("Cross released")
        passed = self.toponav_index is not None and self.toponav_index <= self.crossed_end
        if self.is_stop(prompt) and self.autonomous_flag and not self.cross_override and not self.stop_latched and not passed:
            self.stop_latched = True
            self.latched_prompt = prompt
            self.latched_end, self.latched_next = self.stop_block(self.toponav_index)
            self.get_logger().info(f"Stop latched: '{prompt}' until /cross (next: '{self.latched_next}')")
        if self.use_prompt:
            self.update_text_feature()

    def cross_callback(self, msg: Empty) -> None:
        if not self.stop_latched:
            self.get_logger().info("/cross ignored: no stop instruction is held")
            return
        self.cross_override = True
        self.cross_prompt = self.latched_next or "No language instruction"
        self.cross_until = self.latched_end
        if self.latched_end is not None:
            self.crossed_end = self.latched_end
            self.crossed_next = self.cross_prompt
        self.non_stop_since = None
        self.stop_latched = False
        self.get_logger().info(f"Cross: '{self.latched_prompt}' -> '{self.cross_prompt}'")
        if self.use_prompt:
            self.update_text_feature()

    def effective_prompt(self) -> str:
        if self.cross_override:
            return self.cross_prompt
        if self.stop_latched:
            return self.latched_prompt
        if (
            self.crossed_next is not None
            and self.toponav_index is not None
            and self.toponav_index <= self.crossed_end
            and self.is_stop(self.latest_prompt)
        ):
            return self.crossed_next  # 渡った区間に推定が少し戻っても止まれは出さない
        return self.latest_prompt

    def image_callback(self, msg: Image) -> None:
        rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape((int(msg.height), int(msg.width), 3))
        self.obs_image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv_image = image_to_cv2(msg, self.clip_size)
        self.obs_image = PILImage.fromarray(cv_image[:, :, ::-1])
        debug_img = render_debug_image(cv_image, self.effective_prompt())
        self.debug_image_pub.publish(to_image_msg(debug_img, msg.header.stamp))


    def timer_callback(self) -> None:
        # 位置推定は自律走行していない間も回す(スタート地点まで手で動かす間に推定を収束させる)。
        # 自律走行で止めている間は進めない(遷移モデルで信念が前に流れ、停止ノードを抜けて勝手に走り出さないように)
        stopped = self.autonomous_flag and self.last_linear_vel == 0.0
        if self.toponav is not None and self.obs_image_bgr is not None and not stopped:
            self.update_toponav()
        if not self.autonomous_flag or self.obs_image is None:
            return

        # 学習と同じく context_spacing 周期おきの過去画像を使う(5Hz・spacing 3 なら 0.6 秒おき)
        history_len = self.context_size * self.context_spacing + 1
        self.frame_history.append(self.obs_image.resize(self.obs_size))
        if len(self.frame_history) > history_len:
            self.frame_history.pop(0)
        if len(self.frame_history) < history_len:
            return
        self.context_queue = self.frame_history[:: self.context_spacing]

        obs_images, map_images, cur_large_img = build_omnivla_edge_inputs(
            context_queue=self.context_queue,
            current_image=self.obs_image,
            mask_obs=self.mask_obs,
            mask_clip=self.mask_clip,
            satellite_current=self.satellite_current,
            satellite_goal=self.satellite_goal,
            clip_size=self.clip_size,
            device=self.device,
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

        waypoints, linear_vel, angular_vel = self.action_to_waypoints_and_cmd_vel(action_pred)

        self.publisher_path(waypoints)
        self.publisher_command_velocity(linear_vel, angular_vel)


    def publisher_path(self, waypoints: np.ndarray) -> None:
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
        
        self.path_pub.publish(msg)

    def publisher_command_velocity(self, linear_vel: float, angular_vel: float) -> None:
        self.last_linear_vel = float(linear_vel)
        twist = Twist()
        twist.linear.x = float(linear_vel)
        twist.angular.z = float(angular_vel)
        self.cmd_pub.publish(twist)

    def action_to_waypoints_and_cmd_vel(self, action_pred: np.ndarray) -> Tuple[np.ndarray, float, float]:
        waypoints = action_pred.float().cpu().numpy()[0]
        selected = max(0, min(self.waypoint_select, waypoints.shape[0] - 1))

        dx, dy, hx, hy = [float(v) for v in waypoints[selected]]
        dx *= self.waypoint_spacing
        dy *= self.waypoint_spacing

        if math.hypot(dx, dy) < self.stop_distance:
            return waypoints, 0.0, 0.0
        if self.stop_path_m > 0 and float(np.hypot(waypoints[:, 0], waypoints[:, 1]).max()) * self.waypoint_spacing <= self.stop_path_m:
            return waypoints, 0.0, 0.0

        eps = 0.1
        # 予測点の間隔は context_spacing 周期。spacing 1 のときの 0.2 秒と同じ関係を保つ
        dt = (self.interval_ms / 1000.0) * self.context_spacing

        if abs(dx) < eps and abs(dy) < eps:
            linear_vel = 0.0
            angular_vel = 0.0
        elif abs(dx) < eps:
            linear_vel = 0.0
            angular_vel = math.copysign(math.pi / (2.0 * dt), dy)
        else:
            linear_vel = dx / dt
            angular_vel = math.atan(dy / dx) / dt

        linear_vel = float(np.clip(linear_vel, 0.0, 1.0))
        angular_vel = float(np.clip(angular_vel, -1.0, 1.0))

        maxv = float(self.linear_max_vel)
        maxw = float(self.angular_max_vel)
        if abs(linear_vel) <= maxv:
            if abs(angular_vel) <= maxw:
                linear_vel_limit = linear_vel
                angular_vel_limit = angular_vel
            else:
                rd = linear_vel / angular_vel
                linear_vel_limit = maxw * math.copysign(1.0, linear_vel) * abs(rd)
                angular_vel_limit = maxw * math.copysign(1.0, angular_vel)
        else:
            if abs(angular_vel) <= 0.001:
                linear_vel_limit = maxv * math.copysign(1.0, linear_vel)
                angular_vel_limit = 0.0
            else:
                rd = linear_vel / angular_vel
                if abs(rd) >= maxv / maxw:
                    linear_vel_limit = maxv * math.copysign(1.0, linear_vel)
                    angular_vel_limit = maxv * math.copysign(1.0, angular_vel) / abs(rd)
                else:
                    linear_vel_limit = maxw * math.copysign(1.0, linear_vel) * abs(rd)
                    angular_vel_limit = maxw * math.copysign(1.0, angular_vel)

        return waypoints, float(linear_vel_limit), float(angular_vel_limit)

    def update_text_feature(self) -> None:
        prompt = self.effective_prompt() if self.use_prompt else "No language instruction"
        if prompt == self.encoded_prompt:
            return  # 指示が変わっていなければエンコードし直さない
        self.encoded_prompt = prompt
        token = clip.tokenize(prompt, truncate=True).to(self.device)
        with torch.no_grad():
            self.feat_text = self.text_encoder.encode_text(token)


def main() -> int:
    package_share_dir = Path(get_package_share_directory("navvla"))
    nav_config_path = package_share_dir / "config" / "nav.yaml"
    preprocess_config_path = package_share_dir / "config" / "preprocess.yaml"

    rclpy.init()
    node = OmniVLANavigationNode(nav_config_path, preprocess_config_path, package_share_dir)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
